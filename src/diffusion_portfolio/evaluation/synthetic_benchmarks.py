"""Validation-only statistical forecasting on controlled synthetic returns."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from diffusion_portfolio.baselines.statistical import (
    EWMAGaussian, fit_ar1, fit_gaussian, observed_log_returns, sample_gaussian_returns,
)
from diffusion_portfolio.config import load_config
from diffusion_portfolio.data.core import slice_return_table
from diffusion_portfolio.data.preparation import prepare_return_data, save_standardizer
from diffusion_portfolio.data.registry import load_return_table
from diffusion_portfolio.evaluation.probabilistic import (
    evaluate_probabilistic_forecast, require_finite_arithmetic_moments,
)
from diffusion_portfolio.evaluation.scenarios import probabilistic_summary, save_scenarios


MODEL_STREAMS = {"gaussian": 0, "ar1": 1, "ewma": 2}
SUPPORTED_DATASETS = {
    "synthetic_gaussian", "synthetic_mean_reverting", "synthetic_regime",
    "synthetic_regime_jump", "synthetic_whipsaw",
}


def forecast_baseline(name, fitted, table, windows, *, train_end, n_scenarios, seed):
    """Forecast chronologically, using only observations strictly before each target.

    EWMA consumes every post-training observation before a target, including
    dates omitted by window segmentation. The current target is incorporated
    only when advancing to a later target. Fixed fits never update. Independent
    PCG64 streams keyed by (seed, model ID, date ordinal) preserve exact forecasts
    when model order or the evaluated date subset changes in one environment.
    """
    if name not in MODEL_STREAMS:
        raise ValueError(f"Unknown baseline: {name}")
    dates = windows.target_dates
    positions = table.dates.get_indexer(dates)
    if (not len(dates) or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or np.any(positions < windows.history.shape[1]) or np.any(dates <= pd.Timestamp(train_end))):
        raise ValueError("Forecast target dates must be ordered, post-training and present in the source panel")
    if windows.target.shape[1] != 1:
        raise ValueError("Statistical forecasts support horizon=1 only")
    if not np.array_equal(windows.target[:, 0], table.returns[positions].astype(windows.target.dtype)):
        raise ValueError("Forecast observations differ from the source panel")
    cursor = int(table.dates.searchsorted(pd.Timestamp(train_end), side="right"))
    forecasts = np.empty((len(dates), n_scenarios, len(table.columns)), dtype=np.float64)
    for i, (date, position) in enumerate(zip(dates, positions)):
        history = table.returns[position - windows.history.shape[1]:position]
        if not np.array_equal(windows.history[i], history.astype(windows.history.dtype)):
            raise ValueError("Forecast history differs from the causal source window")
        if name == "ewma":
            while cursor < position:
                fitted.update(table.returns[cursor])
                cursor += 1
            mean = fitted.mean
        else:
            mean = fitted.conditional_mean(windows.history[i, -1])
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, MODEL_STREAMS[name], date.toordinal()])))
        forecasts[i] = sample_gaussian_returns(mean, fitted.covariance, n_scenarios=n_scenarios, rng=rng)
    return forecasts


def benchmark_synthetic_forecasts(config_path, *, output_dir, models=("gaussian", "ar1", "ewma"),
                                  split="validation", n_scenarios=128, limit=None, seed=42, ewma_decay=.94):
    """Fit on unique training observations and score matched validation targets.

    Uses the existing paired raw/model window preparation, metric definitions
    and scenario archive contract. The standardizer is fitted/saved for source
    compatibility; these statistical models themselves fit unstandardized logs.
    No access to datasets.test or simulator ground-truth metadata is needed.
    """
    cfg = load_config(config_path)
    require_finite_arithmetic_moments(cfg.data.dataset)
    if cfg.data.dataset not in SUPPORTED_DATASETS:
        raise ValueError("This benchmark supports controlled finite-moment synthetic datasets only")
    if split != "validation":
        raise ValueError("Synthetic benchmarking supports validation only; the test partition remains held out")
    if cfg.data.horizon != 1 or cfg.data.subtract_risk_free:
        raise ValueError("Synthetic benchmarks require horizon=1 and subtract_risk_free=false")
    models = tuple(models)
    if not models or len(set(models)) != len(models) or any(name not in MODEL_STREAMS for name in models):
        raise ValueError("models must be a nonempty, unique selection of gaussian, ar1, ewma")
    for name, value, minimum in (("seed", seed, 0), ("n_scenarios", n_scenarios, 2), ("limit", limit, 1)):
        if value is None and name == "limit":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if "ewma" in models and (isinstance(ewma_decay, bool) or not np.isfinite(ewma_decay) or not 0 < ewma_decay < 1):
        raise ValueError("EWMA decay must lie strictly between 0 and 1")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}; choose an isolated new directory")
    # Source observations supply unique fitting rows, including the first
    # lookback. Never flatten overlapping windows or read internal log_returns.
    table = slice_return_table(load_return_table(cfg.data.dataset, **cfg.data.source_options),
                               cfg.data.sample_start, cfg.data.val_end)
    training_mask = table.dates <= pd.Timestamp(cfg.data.train_end)
    training = table.returns[training_mask]
    training_dates = table.dates[training_mask]
    datasets, scaler = prepare_return_data(cfg)
    if table.columns != scaler.columns:
        raise ValueError("Source asset order differs from the prepared dataset")
    windows = datasets.val.raw_windows
    available = len(windows.target_dates)
    count = int(available if limit is None else min(limit, available))
    windows = type(windows)(windows.history[:count], windows.target[:count], windows.target_dates[:count])
    dates, observed = windows.target_dates, windows.target[:, 0, :].copy()
    observed_log_returns(observed)  # Domain check only; targets never enter fitting.
    if not count or dates[-1] > pd.Timestamp(cfg.data.val_end):
        raise ValueError("No valid validation targets or targets beyond val_end")
    # Fingerprints identify observations/configuration, never serve as features.
    panel_hash = hashlib.sha256(table.dates.values.astype("datetime64[ns]").tobytes()
                                + np.ascontiguousarray(table.returns).tobytes()).hexdigest()
    source_file = Path(__file__).parents[1] / "data/sources/synthetic_market.py"
    report = {
        "dataset": cfg.data.dataset, "split": split, "seed": int(seed), "n_scenarios": int(n_scenarios),
        "n_dates": count, "available_validation_dates": available, "columns": list(table.columns),
        "target_dates": [str(date.date()) for date in dates], "return_units": "arithmetic decimal asset returns",
        "fit_domain": "log1p(observed arithmetic returns)", "resolved_config": json.loads(json.dumps(asdict(cfg))),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "source_identity": {"dataset": cfg.data.dataset, "source_options": cfg.data.source_options,
                            "train_validation_panel_sha256": panel_hash,
                            "simulator_source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest()},
        "training": {"n_observations": len(training), "start": str(training_dates[0].date()),
                     "end": str(training_dates[-1].date()), "covariance_divisor": "number of observations/pairs (MLE)"},
        "protocol": "First N validation targets; fixed fits freeze at train_end; EWMA updates only before each target.",
        "sampling": "PCG64 with SeedSequence([seed, model ID (gaussian=0, ar1=1, ewma=2), target date ordinal])",
        "numerics": "PSD eigensquare root; only roundoff-negative eigenvalues within 100*eps*spectral_scale become zero; no ridge or return clipping.",
        "interpretation": "Finite-ensemble CRPS/Energy Score contain Monte Carlo error; a small pilot establishes no significance.",
        "models": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    save_standardizer(output / "standardizer.npz", scaler)
    rows = []
    for name in models:
        started = perf_counter()
        if name == "ar1":
            fitted = fit_ar1(training, dates=training_dates, max_gap_days=cfg.data.max_gap_days)
        elif name == "ewma":
            fitted = EWMAGaussian.fit(training, decay=ewma_decay)
        else:
            fitted = fit_gaussian(training)
        parameters = {"mean_log": fitted.mean.tolist(), "covariance_log": fitted.covariance.tolist()}
        if name == "ewma":
            parameters.update(decay=fitted.decay, initialization="training MLE at train_end; no replay of training rows",
                              covariance_update="C_new=lambda*C_old + lambda*(1-lambda)*(x-m_old)*(x-m_old).T")
        else:
            parameters.update(phi=fitted.phi, intercept_log=fitted.intercept.tolist(),
                              covariance_kind="one-step residual innovation" if name == "ar1" else "unconditional marginal")
        samples = forecast_baseline(name, fitted, table, windows, train_end=cfg.data.train_end,
                                    n_scenarios=n_scenarios, seed=seed)
        forecast_seconds = perf_counter() - started
        metrics = evaluate_probabilistic_forecast(samples, observed)
        result = {
            **probabilistic_summary(metrics, table.columns),
            "mean_abs_ace": float(np.abs(metrics.calibration.ace).mean()),
            "mean_forecast_error": float((samples.mean(axis=1) - observed).mean()),
            "mean_predictive_std": float(samples.std(axis=1).mean()),
            "fit_and_forecast_seconds": forecast_seconds, "total_seconds": perf_counter() - started,
            "parameters": parameters,
        }
        directory = output / name
        directory.mkdir()
        save_scenarios(directory / "validation_scenarios.npz", samples, observed, dates, table.columns, compressed=True)
        (directory / "validation_probabilistic_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        report["models"][name] = result
        rows.append({"model": name, "n_dates": count, "n_scenarios": n_scenarios,
                     **{key: result[key] for key in ("crps_mean", "energy_score", "mean_abs_ace", "mean_forecast_error",
                                                     "mean_predictive_std", "total_seconds")}})
        print(f"{name}: CRPS={result['crps_mean']:.8f} | Energy Score={result['energy_score']:.8f} | mean |ACE|={result['mean_abs_ace']:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
