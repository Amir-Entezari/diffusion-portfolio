"""Archive-only diagnostics for the matched historical-context experiment."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diffusion_portfolio.evaluation.forecast_comparison import (
    load_matched_synthetic_forecasts, moving_block_bootstrap,
)
from diffusion_portfolio.evaluation.probabilistic import (
    crps_by_date, energy_score_by_date, evaluate_probabilistic_forecast,
)
from diffusion_portfolio.evaluation.scenarios import probabilistic_summary


def conditional_mean_diagnostics(scenarios, observed, last_returns):
    """Demean each asset over dates, then pool its centered cross-products.

    Estimate E[log(1+r)] by averaging log scenarios, not by logging their
    arithmetic mean. No true generator parameter enters these calculations.
    """
    scenarios, observed, last_returns = (
        np.asarray(values, dtype=np.float64) for values in (scenarios, observed, last_returns)
    )
    if (scenarios.ndim != 3 or observed.ndim != 2 or last_returns.shape != observed.shape
            or scenarios.shape[0] != len(observed) or scenarios.shape[2] != observed.shape[1]
            or len(observed) < 2 or scenarios.shape[1] < 2):
        raise ValueError("Expected aligned scenarios [dates, samples, assets] and observed/last returns [dates, assets]")
    if any(not np.isfinite(v).all() or np.any(v <= -1) for v in (scenarios, observed, last_returns)):
        raise ValueError("Log diagnostics require finite arithmetic decimal returns greater than -1")
    last_log = np.log1p(last_returns)
    predictive_mean = np.log1p(scenarios).mean(axis=1)
    x = last_log - last_log.mean(axis=0)
    y = predictive_mean - predictive_mean.mean(axis=0)
    denominator = np.sum(x * x)
    if denominator <= 0:
        raise ValueError("Lag-dependence slope is undefined for constant historical log returns")
    mse_by_date = ((predictive_mean - np.log1p(observed)) ** 2).mean(axis=1)
    return {"pooled_lag_slope": float(np.sum(x * y) / denominator),
            "log_mean_mse": float(mse_by_date.mean()), "mse_by_date": mse_by_date}


def diagnose_history_ablation(baseline_dir, diffusion_dirs, *, output_dir,
                              block_length=20, n_bootstrap=1000, bootstrap_seed=42):
    """Compare matched mask arms and observed-only baselines, without sampling.

    diffusion_dirs maps K0/K1/K5/K60 labels to saved inference directories.
    Missing arms are allowed for the execution pilot. All supplied arms must
    have identical budgets, completed epochs, contracts except mask, targets
    and ensemble sizes. No intersection/subset selection is performed here.
    """
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Diagnostic output must be empty or absent: {output}")
    if not diffusion_dirs:
        raise ValueError("At least one ablation archive is required")
    first, training_contract, forecasts, training, inputs, policies = None, None, {}, {}, {}, {}
    for label, directory in diffusion_dirs.items():
        matched = load_matched_synthetic_forecasts(baseline_dir, directory)
        policy = matched.contract.get("history_mask")
        if (matched.cfg.data.dataset != "synthetic_mean_reverting" or policy is None
                or policy["keep_last"] not in (0, 1, 5, 60) or label != f'K{policy["keep_last"]}'):
            raise ValueError("Diagnostics require correctly labeled K0/K1/K5/K60 mean-reverting ablation archives")
        contract = {key: value for key, value in matched.contract.items() if key != "history_mask"}
        if matched.cfg.training.epochs != 15:
            raise ValueError("Primary ablation requires the predeclared 15-epoch budget; historical 50-epoch runs are separate")
        if first is None:
            first, training_contract = matched, contract
        elif (contract != training_contract or not matched.dates.equals(first.dates)
              or not np.array_equal(matched.observed, first.observed)
              or matched.metadata["n_scenarios"] != first.metadata["n_scenarios"]
              or matched.training["completed_epochs"] != first.training["completed_epochs"]
              or matched.training["status"] != first.training["status"]):
            raise ValueError("Ablation arms have different contracts, dates, observations, ensemble sizes or completed budgets")
        for key in ("final_train_loss", "final_val_loss", "parameter_count"):
            if key not in matched.training or not np.isfinite(matched.training[key]):
                raise ValueError(f"Missing/invalid training diagnostic: {key}; rerun verified inference")
        if matched.training["parameter_count"] != first.training["parameter_count"]:
            raise ValueError("Ablation parameter counts differ")
        training[label], inputs[label] = matched.training, matched.inputs
        policies[label] = policy
        forecasts[label] = matched.forecasts["diffusion"]
    forecasts.update({name: x for name, x in first.forecasts.items() if name != "diffusion"})
    observed, dates = first.observed, first.dates
    # These are the same unmasked raw histories used by existing baselines.
    last_returns = first.canonical.history[:len(dates), -1, :]
    rows, scores, calibration = [], {}, []
    per_date = {"date": [str(date.date()) for date in dates]}
    for name, samples in forecasts.items():
        diagnostics = conditional_mean_diagnostics(samples, observed, last_returns)
        metrics = evaluate_probabilistic_forecast(samples, observed)
        scores[name] = {"crps": crps_by_date(samples, observed),
                        "energy_score": energy_score_by_date(samples, observed),
                        "log_mean_mse": diagnostics.pop("mse_by_date")}
        rows.append({"model": name, **diagnostics, "crps": metrics.crps_mean,
                     "energy_score": metrics.energy_score,
                     "mean_abs_ace": float(np.abs(metrics.calibration.ace).mean()),
                     **training.get(name, {})})
        calibration.extend({"model": name, "level": item["level"], "coverage": item["picp"],
                            "absolute_error": abs(item["ace"])}
                           for item in probabilistic_summary(metrics)["calibration"])
        per_date.update({f"{name}_{metric}": values for metric, values in scores[name].items()})
    pairs = [(left, right, metric) for left, right in combinations(diffusion_dirs, 2)
             for metric in ("crps", "energy_score", "log_mean_mse")]
    paired = []
    if pairs:
        differences = np.column_stack([scores[a][m] - scores[b][m] for a, b, m in pairs])
        intervals = moving_block_bootstrap(differences, block_length=block_length,
                                           n_bootstrap=n_bootstrap, seed=bootstrap_seed)
        for i, (left, right, metric) in enumerate(pairs):
            paired.append({"left": left, "right": right, "metric": metric, "n_dates": len(dates),
                           "left_mean": float(scores[left][metric].mean()),
                           "right_mean": float(scores[right][metric].mean()),
                           "mean_difference": float(differences[:, i].mean()),
                           "paired_sample_std": float(differences[:, i].std(ddof=1)),
                           "ci95_lower": float(intervals[0, i]), "ci95_upper": float(intervals[1, i])})
    report = {
        "dataset": first.cfg.data.dataset, "split": "validation", "n_dates": len(dates),
        "target_dates": per_date["date"], "columns": first.identity["columns"],
        "return_units": first.identity["return_units"], "n_scenarios": first.metadata["n_scenarios"],
        "source_identity": first.identity["source_identity"], "common_training_contract": training_contract,
        "history_masks": policies,
        "generator_ar_coefficient_reference_only": first.cfg.data.source_options.get("ar_coefficients", 0.4),
        "models": rows, "paired_comparisons": paired, "input_sha256": inputs,
        "bootstrap": {"block_length": block_length, "replications": n_bootstrap, "seed": bootstrap_seed},
        "interpretation": "CRPS, Energy Score and predictive log-mean MSE are primary. Lag slope is explanatory, not a ranking criterion. Predictive log means are finite-ensemble estimates. Negative left-minus-right differences favor the left arm. Validation selected checkpoints: exploratory development evidence, not independent held-out or selection-adjusted significance. Moving-block intervals assume approximately stationary paired differences; short pilots and ensemble noise limit interpretation.",
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(per_date).to_csv(output / "per_date_scores.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_context.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(pd.DataFrame(rows).to_string(index=False))
    return report
