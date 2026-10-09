"""Current Gaussian, Lévy and cached-feature forecast evaluations."""
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import collate_return_batch
from diffusion_portfolio.data.features import FeatureDataset, cached_features
from diffusion_portfolio.data.preparation import prepare_return_data, load_standardizer
from diffusion_portfolio.evaluation.probabilistic import evaluate_probabilistic_forecast
from diffusion_portfolio.evaluation.scenarios import (
    generate_scenarios, probabilistic_summary, save_scenarios, seed_sampling,
)
from diffusion_portfolio.models.diffusion import PrecomputedConditionDiffusion
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.utils.device import resolve_device


def evaluate_diffusion(run_dir, *, batch_size=32, device="auto", split="validation",
                       variant=None, cde_run_dir=None, feature_dir=None):
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if split not in ("validation", "test"):
        raise ValueError("Unknown evaluation split")
    run_dir = Path(run_dir)
    source_dir = Path(cde_run_dir) if variant is not None else run_dir
    cfg = load_config(source_dir / "config.yaml")
    device = resolve_device(device)
    experiment_path = run_dir / "experiment.json"
    experiment = json.loads(experiment_path.read_text()) if experiment_path.exists() else {}
    alpha = experiment.get("alpha")
    if alpha is not None:
        alpha = float(alpha)
        if not 1 < alpha <= 2:
            raise ValueError("Saved experiment has invalid alpha")
    if split == "test" and (alpha is not None or variant is not None):
        raise ValueError("Lévy and feature experiments support validation evaluation only")
    # The legacy Gaussian test CLI seeded *before* model construction. Keep
    # that distinct stream; validation always seeds after checkpoint loading.
    if split == "test":
        seed_sampling(cfg.seed, device)
    datasets, scaler = prepare_return_data(
        cfg, standardizer_path=source_dir / "standardizer.npz"
    )
    dataset = datasets.val if split == "validation" else datasets.test
    dates = dataset.raw_windows.target_dates
    observed = dataset.raw_windows.target[:, 0, :].astype(np.float32, copy=True)
    if split == "validation":
        if dates[0] <= np.datetime64(cfg.data.train_end):
            raise RuntimeError("Validation overlaps training")
        if dates[-1] > np.datetime64(cfg.data.val_end):
            raise RuntimeError("Validation extends past val_end")
    if variant is None:
        model = build_diffusion(cfg, n_assets=len(scaler.columns), alpha=alpha).to(device)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=collate_return_batch, drop_last=False,
        )
    else:
        with np.load(Path(feature_dir) / "validation.npz") as cache:
            features, extra_dim = cached_features(cache, variant, dates)
        loader = DataLoader(
            FeatureDataset(features, dataset.model_windows.target),
            batch_size=batch_size, shuffle=False,
        )
        model = PrecomputedConditionDiffusion(
            build_diffusion(cfg, n_assets=len(scaler.columns), precomputed=True),
            extra_dim=extra_dim,
        ).to(device)
        scaler = load_standardizer(run_dir / "standardizer.npz")
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sampling_seed = int(cfg.training.validation_seed) if split == "validation" else cfg.seed
    if split == "validation":
        seed_sampling(sampling_seed, device)
    forecast = generate_scenarios(
        model, loader, scaler, n_samples=len(dataset),
        n_scenarios=cfg.evaluation.n_scenarios, device=device,
    )
    save_scenarios(
        run_dir / f"{split}_scenarios.npz", forecast, observed, dates, scaler.columns,
        compressed=variant is not None,
    )
    metrics = evaluate_probabilistic_forecast(forecast, observed)
    result = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "n_validation" if split == "validation" else "n_test": len(dataset),
        "n_scenarios": int(cfg.evaluation.n_scenarios),
        **probabilistic_summary(metrics, scaler.columns if variant is None else None),
    }
    if split == "validation":
        result.update(split=split, sampling_seed=sampling_seed)
    if variant is None:
        result.update(
            sampling_batch_size=batch_size,
            scenario_mean=float(forecast.mean()), scenario_std=float(forecast.std()),
            scenario_max_abs=float(np.abs(forecast).max()),
        )
        result[f"{split}_start"] = str(dates[0].date())
        result[f"{split}_end"] = str(dates[-1].date())
    else:
        result["variant"] = variant
    if alpha is not None:
        result.update(
            phase="3A", alpha=alpha,
            mean_abs_ace=float(np.mean(np.abs(metrics.calibration.ace))),
        )
    (run_dir / f"{split}_probabilistic_metrics.json").write_text(json.dumps(result, indent=2))
    print("CRPS:", result["crps_mean"], "Energy Score:", result["energy_score"])
    return result
