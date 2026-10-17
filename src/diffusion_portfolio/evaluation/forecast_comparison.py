"""Strict, paired validation comparison of saved synthetic forecast ensembles."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data.identity import ARITHMETIC_UNITS, file_sha256, synthetic_data_identity, require_training_contract
from diffusion_portfolio.data.preparation import load_standardizer, prepare_return_data
from diffusion_portfolio.evaluation.probabilistic import (
    crps_by_date, energy_score_by_date, evaluate_probabilistic_forecast,
    require_finite_arithmetic_moments,
)
from diffusion_portfolio.evaluation.scenarios import probabilistic_summary


def moving_block_bootstrap(differences, *, block_length=20, n_bootstrap=1000, seed=42):
    """Percentile 95% intervals for paired mean differences.

    Sample overlapping, non-circular contiguous blocks with replacement;
    concatenate and truncate to T observations. All columns share indices.
    Assumes dependence is adequately represented by the chosen block length
    and an approximately stationary score-difference process. This does not
    adjust for checkpoint selection or repeated ensemble Monte Carlo sampling.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim not in (1, 2) or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap differences must be finite with at least two dates")
    if values.ndim == 2 and values.shape[1] == 0:
        raise ValueError("Bootstrap requires at least one paired series")
    for name, value, minimum in (("block_length", block_length, 1), ("n_bootstrap", n_bootstrap, 2), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if block_length > len(values):
        raise ValueError("block_length cannot exceed the number of evaluated dates; choose a shorter block for small pilots")
    rng = np.random.default_rng(seed)
    means = np.empty((n_bootstrap,) + values.shape[1:])
    for i in range(n_bootstrap):
        starts = rng.integers(0, len(values) - block_length + 1, size=(len(values) + block_length - 1) // block_length)
        indices = (starts[:, None] + np.arange(block_length)).ravel()[:len(values)]
        means[i] = values[indices].mean(axis=0)
    return np.quantile(means, [.025, .975], axis=0)


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing required provenance: {path}")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Provenance must be a JSON object: {path}")
    return result


def _require(mapping, keys, label):
    if not isinstance(mapping, dict):
        raise ValueError(f"Invalid {label} metadata: expected an object")
    missing = set(keys) - mapping.keys()
    if missing:
        raise ValueError(f"Missing {label} metadata: {sorted(missing)}")


def _archive(path, report, *, columns, canonical, n_scenarios):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"scenarios", "observed", "dates", "columns"}:
            raise ValueError(f"Invalid scenario archive keys: {path}")
        samples, observed = archive["scenarios"].copy(), archive["observed"].copy()
        dates, assets = archive["dates"].copy(), archive["columns"].copy()
    if (dates.ndim != 1 or not np.issubdtype(dates.dtype, np.datetime64)
            or not len(dates) or np.isnat(dates).any()):
        raise ValueError(f"Invalid target dates: {path}")
    # Archives intentionally store days; Pandas may materialize seconds while
    # canonical windows use nanoseconds. Compare equal instants in one unit.
    dates = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"Target dates are not strictly chronological: {path}")
    if assets.tolist() != columns:
        raise ValueError(f"Asset order differs: {path}")
    if samples.shape != (len(dates), n_scenarios, len(columns)) or observed.shape != (len(dates), len(columns)):
        raise ValueError(f"Scenario count or forecast/observation shape differs: {path}")
    if (not np.isfinite(samples).all() or not np.isfinite(observed).all()
            or np.any(samples <= -1) or np.any(observed <= -1)):
        raise ValueError(f"Invalid arithmetic decimal returns: {path}; samples are never clipped")
    if report["n_dates"] != len(dates) or report["target_dates"] != [str(date.date()) for date in dates]:
        raise ValueError(f"Target dates differ from recorded metadata: {path}")
    if not dates.equals(canonical.target_dates[:len(dates)]):
        raise ValueError(f"Target dates differ from the canonical first validation dates: {path}")
    if not np.array_equal(observed, canonical.target[:len(dates), 0]):
        raise ValueError(f"Target observations differ from the canonical source: {path}")
    return samples, observed, dates


def compare_synthetic_forecasts(baseline_dir, diffusion_dir, *, output_dir, limit=None,
                                block_length=20, n_bootstrap=1000, bootstrap_seed=42):
    """Compare saved ensembles only; no training, sampling or baseline refitting."""
    baseline_dir, diffusion_dir, output = map(Path, (baseline_dir, diffusion_dir, output_dir))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Comparison output must be empty or absent: {output}")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 2):
        raise ValueError("Comparison limit must be an integer >= 2")
    baseline = _read_json(baseline_dir / "report.json")
    diffusion = _read_json(diffusion_dir / "forecast_metadata.json")
    _require(baseline, ("dataset", "split", "columns", "source_identity", "resolved_config", "return_units",
                        "n_dates", "n_scenarios", "target_dates", "models", "training", "available_validation_dates"), "baseline")
    _require(diffusion, ("data_config", "source_identity", "columns", "split", "return_units", "model_kind",
                         "n_dates", "n_scenarios", "target_dates", "training", "training_run",
                         "checkpoint_files_sha256", "archive_sha256", "subset", "training_contract"), "diffusion")
    if baseline["split"] != "validation" or diffusion["split"] != "validation":
        raise ValueError("Only validation archives may be compared; the test partition remains held out")
    if baseline["return_units"] != ARITHMETIC_UNITS or diffusion["return_units"] != ARITHMETIC_UNITS:
        raise ValueError("Return units must be raw arithmetic decimal asset returns")
    cfg = load_config(diffusion_dir / "config.yaml")
    baseline_cfg = load_config(baseline_dir / "config.yaml")
    require_finite_arithmetic_moments(cfg.data.dataset)
    if (asdict(cfg.data) != asdict(baseline_cfg.data) or diffusion["data_config"] != asdict(cfg.data)
            or baseline["dataset"] != cfg.data.dataset
            or baseline["resolved_config"] != json.loads(json.dumps(asdict(baseline_cfg)))):
        raise ValueError("Dataset/source configuration mismatch (including seeds, dates, lookback, horizon and RF treatment)")
    if diffusion["model_kind"] != "vanilla_diffusion" or cfg.model.history_encoder != "mlp":
        raise ValueError("Only Vanilla diffusion is supported in this comparison")
    scaler = load_standardizer(diffusion_dir / "standardizer.npz")
    identity = synthetic_data_identity(cfg, scaler)
    recorded_identity = _read_json(diffusion_dir / "data_identity.json")
    contract = require_training_contract(recorded_identity, cfg)
    if diffusion["training_contract"] != contract:
        raise ValueError("Diffusion metadata differs from the immutable training contract")
    recorded_identity.pop("training_contract")
    recorded_checkpoints = recorded_identity.pop("checkpoint_sha256", {})
    if recorded_identity != identity:
        raise ValueError("Saved training source provenance differs from reconstructed data identity")
    for record in (baseline, diffusion):
        if record["source_identity"] != identity["source_identity"]:
            raise ValueError("Source identity/fingerprint mismatch; numerical-library differences require investigation, not silent alignment")
        if record["columns"] != identity["columns"]:
            raise ValueError("Asset order differs from the verified source")
        if any(record["training"].get(key) != value for key, value in identity["training"].items()):
            raise ValueError("Training observation dates/count differ from the verified source")
    synthetic_data_identity(baseline_cfg, load_standardizer(baseline_dir / "standardizer.npz"))
    hashes = diffusion["checkpoint_files_sha256"]
    _require(hashes, ("config.yaml", "standardizer.npz", "best.pt", "data_identity.json"), "checkpoint identity")
    if ("best.pt" not in recorded_checkpoints
            or any(hashes.get(name) != digest for name, digest in recorded_checkpoints.items())):
        raise ValueError("Diffusion checkpoint identity is not bound to its recorded training source")
    for name in ("config.yaml", "standardizer.npz", "data_identity.json"):
        if file_sha256(diffusion_dir / name) != hashes[name]:
            raise ValueError(f"Diffusion saved file differs from the evaluated checkpoint provenance: {name}")
    training = diffusion["training_run"]
    _require(training, ("status", "completed_epochs", "configured_epochs", "checkpoint_epoch", "checkpoint_val_loss"), "training readiness")
    if (training["status"] not in ("preliminary", "completed_declared_budget")
            or not 1 <= training["checkpoint_epoch"] <= training["completed_epochs"] <= cfg.training.epochs
            or training["configured_epochs"] != cfg.training.epochs or not np.isfinite(training["checkpoint_val_loss"])
            or (training["status"] == "completed_declared_budget" and training["completed_epochs"] != cfg.training.epochs)):
        raise ValueError("Inconsistent diffusion training readiness metadata")
    if diffusion["n_scenarios"] != baseline["n_scenarios"] or diffusion["n_scenarios"] != cfg.evaluation.n_scenarios:
        raise ValueError("Predictive scenario counts differ")
    if file_sha256(diffusion_dir / "validation_scenarios.npz") != diffusion["archive_sha256"]:
        raise ValueError("Diffusion forecast archive differs from its saved provenance")
    datasets, _ = prepare_return_data(cfg, standardizer_path=diffusion_dir / "standardizer.npz")
    canonical = datasets.val.raw_windows
    if (baseline["available_validation_dates"] != len(canonical.target_dates)
            or diffusion["subset"].get("available") != len(canonical.target_dates)):
        raise ValueError("Validation availability differs from the configured chronology")
    n_scenarios = diffusion["n_scenarios"]
    samples, observed, dates = _archive(diffusion_dir / "validation_scenarios.npz", diffusion,
                                       columns=identity["columns"], canonical=canonical, n_scenarios=n_scenarios)
    count = len(dates) if limit is None else limit
    if count < 2 or count > len(dates):
        raise ValueError("Comparison limit exceeds available diffusion dates or fewer than two dates remain")
    forecasts = {"diffusion": samples[:count]}
    observed, dates = observed[:count], dates[:count]
    if not baseline["models"] or set(baseline["models"]) - {"gaussian", "ar1", "ewma"}:
        raise ValueError("Baseline report must contain supported statistical models")
    inputs = {"diffusion_metadata": file_sha256(diffusion_dir / "forecast_metadata.json"),
              "baseline_report": file_sha256(baseline_dir / "report.json"),
              "diffusion_archive": diffusion["archive_sha256"]}
    for name in sorted(baseline["models"]):
        path = baseline_dir / name / "validation_scenarios.npz"
        x, y, baseline_dates = _archive(path, baseline, columns=identity["columns"], canonical=canonical, n_scenarios=n_scenarios)
        if len(x) < count or not baseline_dates[:count].equals(dates) or not np.array_equal(y[:count], observed):
            raise ValueError(f"Paired dates/observations do not match for {name}")
        forecasts[name] = x[:count]
        inputs[name] = file_sha256(path)
    scores, metrics = {}, {}
    for name, forecast in forecasts.items():
        scores[name] = {"crps": crps_by_date(forecast, observed), "energy_score": energy_score_by_date(forecast, observed)}
        metric = evaluate_probabilistic_forecast(forecast, observed)
        metrics[name] = {**probabilistic_summary(metric, identity["columns"]),
                         "mean_abs_ace": float(np.abs(metric.calibration.ace).mean())}
    pairs = [(name, metric) for name in forecasts if name != "diffusion" for metric in ("crps", "energy_score")]
    differences = np.column_stack([scores["diffusion"][metric] - scores[name][metric] for name, metric in pairs])
    intervals = moving_block_bootstrap(differences, block_length=block_length, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    rows = []
    per_date = {"date": [str(date.date()) for date in dates]}
    for name in scores:
        per_date.update({f"{name}_{metric}": value for metric, value in scores[name].items()})
    for i, (name, metric) in enumerate(pairs):
        baseline_mean, diffusion_mean = float(scores[name][metric].mean()), float(scores["diffusion"][metric].mean())
        rows.append({"baseline": name, "metric": metric, "n_dates": count,
                     "diffusion_mean": diffusion_mean, "baseline_mean": baseline_mean,
                     "mean_difference": float(differences[:, i].mean()),
                     "relative_improvement": (baseline_mean - diffusion_mean) / baseline_mean if baseline_mean != 0 else None,
                     "paired_sample_std": float(differences[:, i].std(ddof=1)),
                     "ci95_lower": float(intervals[0, i]), "ci95_upper": float(intervals[1, i])})
        per_date[f"diffusion_minus_{name}_{metric}"] = differences[:, i]
    report = {
        "dataset": cfg.data.dataset, "split": "validation", "n_dates": count, "n_scenarios": n_scenarios,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "start": str(dates[0].date()), "end": str(dates[-1].date()), "return_units": ARITHMETIC_UNITS,
        "data_config": identity["data_config"], "source_identity": identity["source_identity"],
        "training_contract": contract, "training_run": training, "checkpoint_files_sha256": hashes, "input_sha256": inputs,
        "models": metrics, "paired_comparisons": rows,
        "bootstrap": {"method": "overlapping non-circular moving blocks, percentile 95% CI", "block_length": block_length,
                      "replications": n_bootstrap, "seed": bootstrap_seed},
        "interpretation": "Negative diffusion-minus-baseline differences favor diffusion. Validation selected the checkpoint: development evidence, not independent final generalization or selection-adjusted significance. Intervals assume approximately stationary score differences and adequate block length, condition on these finite ensembles, and do not quantify model-fitting or Monte Carlo replication uncertainty. Short pilots are exploratory.",
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_date).to_csv(output / "per_date_scores.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "paired_comparison.csv", index=False)
    pd.DataFrame([{"model": name, "level": row["level"], "coverage": row["picp"], "absolute_error": abs(row["ace"])}
                  for name, values in metrics.items() for row in values["calibration"]]).to_csv(output / "calibration.csv", index=False)
    (output / "comparison_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(pd.DataFrame(rows).to_string(index=False))
    return report
