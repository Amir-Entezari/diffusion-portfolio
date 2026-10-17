"""Matched context x denoiser-capacity diagnostics from saved validation forecasts."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffusion_portfolio.data.identity import file_sha256
from diffusion_portfolio.evaluation.forecast_comparison import load_matched_synthetic_forecasts, moving_block_bootstrap
from diffusion_portfolio.evaluation.history_ablation import conditional_mean_diagnostics
from diffusion_portfolio.evaluation.probabilistic import crps_by_date, energy_score_by_date, evaluate_probabilistic_forecast
from diffusion_portfolio.evaluation.scenarios import probabilistic_summary
from diffusion_portfolio.models.diffusion.build import build_diffusion


CAPACITY_CHANNELS = {"Large": (32, 64, 128), "Small": (16, 32, 64)}
CAPACITY_ARMS = {"K1-Large": (1, "Large"), "K60-Large": (60, "Large"),
                 "K1-Small": (1, "Small"), "K60-Small": (60, "Small")}
SCORE_NAMES = ("crps", "energy_score", "log_mean_mse")


def capacity_parameter_counts(cfg, n_assets):
    """Count existing modules without changing the caller's initialization RNG."""
    with torch.random.fork_rng(devices=[]):
        model = build_diffusion(cfg, n_assets=n_assets)
    return {"parameter_count": sum(p.numel() for p in model.parameters()),
            "history_encoder_parameter_count": sum(p.numel() for p in model.history_encoder.parameters()),
            "denoiser_parameter_count": sum(p.numel() for p in model.score_network.parameters())}


def capacity_contrasts(scores, *, pilot=False):
    """Per-date planned effects; the interaction is formed before resampling."""
    effects = {}
    for metric in SCORE_NAMES:
        small_gap = scores["K60-Small"][metric] - scores["K1-Small"][metric]
        effects[f"context_Small_{metric}"] = small_gap
        if not pilot:
            large_gap = scores["K60-Large"][metric] - scores["K1-Large"][metric]
            effects[f"capacity_K1_{metric}"] = scores["K1-Small"][metric] - scores["K1-Large"][metric]
            effects[f"capacity_K60_{metric}"] = scores["K60-Small"][metric] - scores["K60-Large"][metric]
            effects[f"context_Large_{metric}"] = large_gap
            effects[f"interaction_{metric}"] = small_gap - large_gap
    return effects


def _verified_history(matched, forecast_dir, run_dir):
    """Read only history whose file hash was recorded at forecast generation.

    Prefer last.pt over a possibly stale CSV. Original run directories are
    explicit inputs because older forecast archives did not copy histories.
    No legacy metadata is amended and no checkpoint is repaired.
    """
    locations = [Path(run_dir)] if run_dir is not None else []
    locations.append(Path(forecast_dir))
    for name in ("last.pt", "history.csv"):
        for location in locations:
            path = location / name
            if path.is_file() and matched.hashes.get(name) == file_sha256(path):
                history = (pd.DataFrame(torch.load(path, map_location="cpu", weights_only=False)["history"])
                           if name == "last.pt" else pd.read_csv(path))
                required = {"epoch", "train_loss", "val_loss"}
                if (not required.issubset(history.columns)
                        or history.epoch.tolist() != list(range(1, matched.training["completed_epochs"] + 1))
                        or not np.isfinite(history[["train_loss", "val_loss"]].to_numpy()).all()):
                    raise ValueError("Invalid verified training history")
                best = history.loc[history.epoch == matched.training["checkpoint_epoch"]].iloc[0]
                expected = [matched.training["checkpoint_val_loss"], matched.training["checkpoint_val_loss"],
                            matched.training["final_train_loss"], matched.training["final_val_loss"]]
                actual = [best.val_loss, history.val_loss.min(), history.train_loss.iloc[-1], history.val_loss.iloc[-1]]
                if not np.allclose(actual, expected, rtol=0, atol=1e-8):
                    raise ValueError("Verified training history conflicts with forecast checkpoint metadata")
                return history, {"file": str(path), "sha256": file_sha256(path)}
    raise ValueError("Missing matching verified training history; supply original --run-dirs LABEL=path (last.pt or history.csv)")


def plot_capacity_losses(history, selected_epochs, path):
    """Save the actual training/validation curves and selected epochs."""
    import matplotlib.pyplot as plt

    labels = list(selected_epochs)
    fig, axes = plt.subplots(1 if len(labels) == 2 else 2, 2, figsize=(11, 4 if len(labels) == 2 else 7), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        rows = history.loc[history.model == label]
        ax.plot(rows.epoch, rows.train_loss, label="Train Loss")
        ax.plot(rows.epoch, rows.val_loss, label="Validation Loss")
        ax.axvline(selected_epochs[label], color="gray", linestyle="--", label="Selected epoch")
        ax.set(title=label, xlabel="Epoch", ylabel="Diffusion loss")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def diagnose_capacity_ablation(baseline_dir, diffusion_dirs, *, output_dir, run_dirs=None,
                               pilot=False, block_length=20, n_bootstrap=1000, bootstrap_seed=42):
    """Strict four-arm full comparison, or a separately labeled Small-only pilot.

    Full mode requires every available validation target and completed 15-epoch
    runs. Pilot mode accepts exactly two completed epochs in both Small arms;
    it never compares their partial budgets with completed Large models.
    """
    expected = {"K1-Small", "K60-Small"} if pilot else set(CAPACITY_ARMS)
    if set(diffusion_dirs) != expected:
        raise ValueError(f"Expected exactly these {'pilot' if pilot else 'full'} capacity labels: {sorted(expected)}")
    run_dirs = {} if run_dirs is None else run_dirs
    if set(run_dirs) - expected:
        raise ValueError("Unknown capacity run-dir labels")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Capacity output must be empty or absent: {output}")
    first, common_contract = None, None
    forecasts, training, counts, histories, inputs, contracts = {}, {}, {}, {}, {}, {}
    for label in CAPACITY_ARMS:
        if label not in expected:
            continue
        matched = load_matched_synthetic_forecasts(baseline_dir, diffusion_dirs[label])
        keep, capacity = CAPACITY_ARMS[label]
        policy = matched.contract.get("history_mask")
        if (matched.cfg.data.dataset != "synthetic_mean_reverting" or policy is None
                or policy["keep_last"] != keep or tuple(matched.cfg.model.channels) != CAPACITY_CHANNELS[capacity]):
            raise ValueError(f"Capacity label {label} does not match its saved history policy/channels")
        if (matched.cfg.training.epochs != 15 or matched.metadata.get("batch_size") != 16
                or matched.metadata.get("sampling_seed") != matched.cfg.training.validation_seed):
            raise ValueError("Capacity protocol requires a 15-epoch budget, inference batch size 16 and the original validation seed")
        state = matched.training
        if pilot:
            if state["status"] != "preliminary" or state["completed_epochs"] != 2:
                raise ValueError("Pilot comparison requires two-epoch preliminary Small runs only")
        elif (state["status"] != "completed_declared_budget" or state["completed_epochs"] != 15
              or len(matched.dates) != len(matched.canonical.target_dates)):
            raise ValueError("Full capacity comparison requires completed 15-epoch runs and all validation targets")
        contract = deepcopy(matched.contract)
        contract["model"].pop("channels")
        contract["history_mask"].pop("keep_last")
        if first is None:
            first, common_contract = matched, contract
        elif (contract != common_contract or not matched.dates.equals(first.dates)
              or matched.identity != first.identity or not np.array_equal(matched.observed, first.observed)
              or matched.metadata["n_scenarios"] != first.metadata["n_scenarios"]):
            raise ValueError("Capacity arms differ beyond the declared channels/history factors, or have unaligned sources/targets")
        counts[label] = capacity_parameter_counts(matched.cfg, len(matched.identity["columns"]))
        if matched.training.get("parameter_count") != counts[label]["parameter_count"]:
            raise ValueError("Recorded capacity parameter count does not match the saved architecture")
        history, evidence = _verified_history(matched, diffusion_dirs[label], run_dirs.get(label))
        best = history.loc[history.epoch == state["checkpoint_epoch"]].iloc[0]
        histories[label] = history
        training[label] = {**state, **counts[label],
                           "selected_train_loss": float(best.train_loss),
                           "selected_train_val_gap": float(best.val_loss - best.train_loss),
                           "final_train_val_gap": float(history.val_loss.iloc[-1] - history.train_loss.iloc[-1])}
        inputs[label] = {**matched.inputs, "verified_history": evidence,
                         "sampling": {key: matched.metadata.get(key) for key in
                                      ("sampling_seed", "batch_size", "device", "torch_version", "numpy_version")}}
        contracts[label] = matched.contract
        forecasts[label] = matched.forecasts["diffusion"]
    if len({count["history_encoder_parameter_count"] for count in counts.values()}) != 1:
        raise ValueError("History encoder capacity must be identical across arms")
    if not pilot and counts["K1-Small"]["parameter_count"] >= counts["K1-Large"]["parameter_count"]:
        raise ValueError("Small denoiser must have fewer parameters than Large")
    if set(first.forecasts) != {"diffusion", "gaussian", "ar1", "ewma"}:
        raise ValueError("Capacity comparison requires Gaussian, AR(1) and EWMA baseline archives")
    forecasts.update({name: value for name, value in first.forecasts.items() if name != "diffusion"})
    observed, dates = first.observed, first.dates
    last_returns = first.canonical.history[:len(dates), -1]
    rows, scores, calibration = [], {}, []
    per_date = {"date": [str(date.date()) for date in dates]}
    for label, samples in forecasts.items():
        diagnostics = conditional_mean_diagnostics(samples, observed, last_returns)
        metrics = evaluate_probabilistic_forecast(samples, observed)
        scores[label] = {"crps": crps_by_date(samples, observed),
                         "energy_score": energy_score_by_date(samples, observed),
                         "log_mean_mse": diagnostics.pop("mse_by_date")}
        rows.append({"model": label, **diagnostics, "crps": metrics.crps_mean,
                     "energy_score": metrics.energy_score, "mean_abs_ace": float(np.abs(metrics.calibration.ace).mean()),
                     **training.get(label, {})})
        calibration.extend({"model": label, "level": row["level"], "coverage": row["picp"],
                            "absolute_error": abs(row["ace"])} for row in probabilistic_summary(metrics)["calibration"])
        per_date.update({f"{label}_{metric}": values for metric, values in scores[label].items()})
    effects = capacity_contrasts(scores, pilot=pilot)
    # One bootstrap call shares the sampled chronological indices across every
    # planned contrast, including the already formed difference-in-differences.
    differences = np.column_stack(list(effects.values()))
    intervals = moving_block_bootstrap(differences, block_length=block_length,
                                       n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    contrasts = [{"contrast": name, "n_dates": len(dates), "mean_effect": float(series.mean()),
                  "paired_sample_std": float(series.std(ddof=1)),
                  "ci95_lower": float(intervals[0, i]), "ci95_upper": float(intervals[1, i])}
                 for i, (name, series) in enumerate(effects.items())]
    per_date.update(effects)
    report = {
        "experiment": "context_x_denoiser_capacity", "pilot": pilot, "split": "validation",
        "dataset": first.cfg.data.dataset, "source_identity": first.identity["source_identity"],
        "return_units": first.identity["return_units"], "columns": first.identity["columns"],
        "n_dates": len(dates), "target_dates": per_date["date"], "n_scenarios": first.metadata["n_scenarios"],
        "common_contract_except_factors": common_contract, "training_contracts": contracts,
        "models": rows, "contrasts": contrasts, "input_sha256": inputs,
        "bootstrap": {"method": "paired overlapping non-circular moving blocks, percentile 95% CI",
                      "block_length": block_length, "replications": n_bootstrap, "seed": bootstrap_seed},
        "interpretation": "Capacity effects are Small minus Large; context effects are K60 minus K1. Interaction is (K60-Small minus K1-Small) minus (K60-Large minus K1-Large); negative narrows the signed K60-versus-K1 loss gap. Pilot results describe execution only. Full results are single-training-seed, validation-selected development evidence, not independent final significance. Block intervals condition on saved finite ensembles and omit training/selection uncertainty. Loss gaps are descriptive: training corruption/examples and fixed validation corruption/examples differ. CRPS, Energy Score and predictive log-mean MSE are primary; diffusion loss, lag slope and calibration diagnose different properties.",
    }
    output.mkdir(parents=True, exist_ok=True)
    history = pd.concat([value.assign(model=label) for label, value in histories.items()], ignore_index=True)
    history.to_csv(output / "training_history.csv", index=False)
    plot_capacity_losses(history, {label: info["checkpoint_epoch"] for label, info in training.items()}, output / "loss_curves.png")
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(per_date).to_csv(output / "per_date_scores.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "contrasts.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    (output / "capacity_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(pd.DataFrame(rows).drop(columns=["note"], errors="ignore").to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))
    return report
