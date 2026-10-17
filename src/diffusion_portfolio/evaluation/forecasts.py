"""Current Gaussian, Lévy and cached-feature forecast evaluations."""
from pathlib import Path
import json
import shutil
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import collate_return_batch, ReturnWindowDataset, WindowedReturns
from diffusion_portfolio.data.identity import COMPARISON_DATASETS, file_sha256, require_training_contract, saved_history_mask
from diffusion_portfolio.data.features import FeatureDataset, cached_features
from diffusion_portfolio.data.preparation import prepare_return_data, load_standardizer
from diffusion_portfolio.evaluation.probabilistic import evaluate_probabilistic_forecast, require_finite_arithmetic_moments
from diffusion_portfolio.evaluation.scenarios import (
    generate_scenarios, probabilistic_summary, save_scenarios, seed_sampling,
)
from diffusion_portfolio.models.diffusion import PrecomputedConditionDiffusion
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.utils.device import resolve_device


def evaluate_diffusion(run_dir, *, batch_size=32, device="auto", split="validation",
                       variant=None, cde_run_dir=None, feature_dir=None, limit=None, output_dir=None):
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if split not in ("validation", "test"):
        raise ValueError("Unknown evaluation split")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if split != "validation" or variant is not None:
            raise ValueError("Limited inference supports ordinary validation diffusion only")
        if output_dir is None:
            raise ValueError("Limited inference requires an isolated output-dir")
    run_dir = Path(run_dir)
    destination = Path(output_dir) if output_dir is not None else run_dir
    if output_dir is not None and destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Inference output directory must be empty or absent: {destination}")
    source_dir = Path(cde_run_dir) if variant is not None else run_dir
    cfg = load_config(source_dir / "config.yaml")
    history_mask = saved_history_mask(source_dir, cfg)
    if history_mask is not None and (split != "validation" or variant is not None):
        raise ValueError("History-mask ablations support ordinary validation evaluation only")
    provenance_path = source_dir / "data_identity.json"
    if provenance_path.is_file():
        recorded = json.loads(provenance_path.read_text())
        if "training_contract" in recorded:
            require_training_contract(recorded, cfg)
    require_finite_arithmetic_moments(cfg.data.dataset)
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
    available = len(dataset)
    if limit is not None:
        if limit > available:
            raise ValueError(f"limit={limit} exceeds the {available} available validation dates")
        def prefix(windows):
            return WindowedReturns(windows.history[:limit], windows.target[:limit], windows.target_dates[:limit])
        dataset = ReturnWindowDataset(prefix(dataset.model_windows), prefix(dataset.raw_windows))
    dates = dataset.raw_windows.target_dates
    observed = dataset.raw_windows.target[:, 0, :].astype(np.float32, copy=True)
    if split == "validation":
        if dates[0] <= np.datetime64(cfg.data.train_end):
            raise RuntimeError("Validation overlaps training")
        if dates[-1] > np.datetime64(cfg.data.val_end):
            raise RuntimeError("Validation extends past val_end")
    if variant is None:
        model = build_diffusion(cfg, n_assets=len(scaler.columns), alpha=alpha,
                                **({"history_mask": history_mask} if history_mask is not None else {})).to(device)
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
    provenance = None
    if ((output_dir is not None or history_mask is not None) and split == "validation" and cfg.data.dataset in COMPARISON_DATASETS
            and variant is None):
        if alpha is not None:
            raise ValueError("Synthetic comparison supports Vanilla Gaussian diffusion only")
        from diffusion_portfolio.evaluation.forecast_identity import verify_synthetic_checkpoint
        provenance = verify_synthetic_checkpoint(run_dir, cfg, scaler, checkpoint)
    sampling_seed = int(cfg.training.validation_seed) if split == "validation" else cfg.seed
    if split == "validation":
        seed_sampling(sampling_seed, device)
    started = perf_counter()
    forecast = generate_scenarios(
        model, loader, scaler, n_samples=len(dataset),
        n_scenarios=cfg.evaluation.n_scenarios, device=device,
    )
    elapsed = perf_counter() - started
    destination.mkdir(parents=True, exist_ok=True)
    save_scenarios(
        destination / f"{split}_scenarios.npz", forecast, observed, dates, scaler.columns,
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
    if output_dir is not None:
        result.update(evaluation_limit=limit, available_samples=available, sampling_device=str(device), sampling_seconds=elapsed)
    (destination / f"{split}_probabilistic_metrics.json").write_text(json.dumps(result, indent=2))
    if provenance is not None:
        for name in ("config.yaml", "standardizer.npz", "data_identity.json") + (("experiment.json",) if history_mask is not None else ()):
            if destination.resolve() != run_dir.resolve():
                shutil.copy2(run_dir / name, destination / name)
        metadata = {
            **provenance, "model_kind": "vanilla_diffusion", "split": "validation", "n_dates": len(dataset),
            "target_dates": [str(date.date()) for date in dates], "n_scenarios": cfg.evaluation.n_scenarios,
            "sampling_seed": sampling_seed, "batch_size": batch_size, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "subset": {"selection": "first chronological validation targets", "limit": limit, "available": available},
            "archive_sha256": file_sha256(destination / "validation_scenarios.npz"),
            "sampling_note": "Reproducible with identical settings/environment; batch size, limit (last batch), or device can change individual draws.",
        }
        if history_mask is not None:
            metadata["history_mask"] = history_mask
        (destination / "forecast_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print("CRPS:", result["crps_mean"], "Energy Score:", result["energy_score"])
    return result
