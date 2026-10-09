"""Explicit Gaussian/Lévy training and frozen-feature ablation runs."""
from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import make_dataloaders
from diffusion_portfolio.data.features import FeatureDataset, cached_features
from diffusion_portfolio.data.preparation import prepare_return_data, save_standardizer
from diffusion_portfolio.models.diffusion import PrecomputedConditionDiffusion
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.training.diffusion import fit_diffusion
from diffusion_portfolio.utils.device import resolve_device
from diffusion_portfolio.utils.seed import set_global_seed


def fit_and_record(model, train_loader, val_loader, cfg, output_dir, *, device,
                   resume=False, stop_after_epoch=None):
    """Use the existing optimizer, validation corruption and resume protocol."""
    training = cfg.training
    result = fit_diffusion(
        model, train_loader, val_loader,
        epochs=training.epochs, learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        gradient_clip_norm=training.gradient_clip_norm,
        validation_seed=training.validation_seed, device=device,
        checkpoint_path=output_dir / "best.pt",
        resume_path=output_dir / "last.pt" if resume else None,
        stop_after_epoch=stop_after_epoch, verbose=True,
    )
    pd.DataFrame([
        {"epoch": r.epoch, "train_loss": r.train_loss, "val_loss": r.val_loss}
        for r in result.history
    ]).to_csv(output_dir / "history.csv", index=False)
    return result


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def train_diffusion(config_path, output_dir, *, device="auto", resume=False,
                    stop_after_epoch=None, alpha=None, reference_run_dir=None):
    config_path, output_dir = Path(config_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if alpha is None and resume and (output_dir / "summary.json").exists():
        print("Training already complete:", output_dir)
        return
    cfg = load_config(config_path)
    metadata = {}
    standardizer_path = None
    if alpha is not None:
        alpha = float(alpha)
        if not 1 < alpha <= 2:
            raise ValueError("alpha must satisfy 1 < alpha <= 2")
        if cfg.model.prediction_type != "epsilon":
            raise ValueError("Lévy config must use epsilon prediction")
        standardizer_path = Path(reference_run_dir) / "standardizer.npz"
        if not standardizer_path.exists():
            raise FileNotFoundError(standardizer_path)
        # Historical metadata schema is retained for existing consumers.
        metadata = {
            "phase": "3A", "alpha": alpha, "prediction_type": "epsilon",
            "loss": "mean_per_sample_rmse",
            "history_encoder": cfg.model.history_encoder, "seed": int(cfg.seed),
        }
        metadata_path = output_dir / "experiment.json"
        if metadata_path.exists():
            if float(json.loads(metadata_path.read_text())["alpha"]) != alpha:
                raise RuntimeError("Existing directory belongs to another alpha")
        else:
            metadata_path.write_text(json.dumps(metadata, indent=2))
        if resume and (output_dir / "summary.json").exists():
            print("Training already complete:", output_dir)
            return
    else:
        set_global_seed(cfg.seed, deterministic=False)
    device = resolve_device(device)
    datasets, scaler = prepare_return_data(cfg, standardizer_path=standardizer_path)
    loaders = make_dataloaders(
        datasets, batch_size=cfg.training.batch_size, seed=cfg.seed, num_workers=0
    )
    if alpha is not None:
        set_global_seed(cfg.seed, deterministic=False)
    model = build_diffusion(cfg, n_assets=len(scaler.columns), alpha=alpha)
    parameter_count = sum(p.numel() for p in model.parameters())
    if alpha is not None:
        set_global_seed(cfg.seed, deterministic=False)
    shutil.copy2(config_path, output_dir / "config.yaml")
    if standardizer_path is None:
        save_standardizer(output_dir / "standardizer.npz", scaler)
    else:
        shutil.copy2(standardizer_path, output_dir / "standardizer.npz")
    print("Device:", device, "Encoder:", cfg.model.history_encoder)
    print("Prediction:", model.prediction_type, "Parameters:", parameter_count)
    result = fit_and_record(
        model, loaders.train, loaders.val, cfg, output_dir, device=device,
        resume=resume, stop_after_epoch=stop_after_epoch,
    )
    if (stop_after_epoch is not None and result.history
            and result.history[-1].epoch < cfg.training.epochs):
        print("Training chunk complete. Resume checkpoint:", output_dir / "last.pt")
        return result
    summary = {
        **metadata, "parameter_count": parameter_count,
        "train_samples": len(datasets.train), "val_samples": len(datasets.val),
        "best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss,
    }
    if alpha is None:
        history = datasets.val[0]["history"].unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            generated = model.sample(history, n_scenarios=64).detach().cpu().numpy()
        raw = scaler.inverse_transform(generated)
        sampling_summary = {
            "standardized_mean": float(generated.mean()),
            "standardized_std": float(generated.std()),
            "standardized_max_abs": float(np.abs(generated).max()),
            "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
            "raw_max_abs": float(np.abs(raw).max()),
        }
        summary.update(
            git_commit=git_commit(), device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            prediction_type=cfg.model.prediction_type,
            test_samples=len(datasets.test), sampling_sanity=sampling_summary,
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Best epoch:", result.best_epoch, "Validation loss:", result.best_val_loss)
    return result


def train_feature_ablation(variant, cde_run_dir, feature_dir, output_dir, *,
                           device="auto", resume=False):
    cde_run_dir, feature_dir, output_dir = map(
        Path, (cde_run_dir, feature_dir, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and (output_dir / "summary.json").exists():
        print("Training already complete:", output_dir)
        return
    cfg = load_config(cde_run_dir / "config.yaml")
    device = resolve_device(device)
    set_global_seed(cfg.seed, deterministic=False)
    datasets, scaler = prepare_return_data(
        cfg, standardizer_path=cde_run_dir / "standardizer.npz"
    )
    with np.load(feature_dir / "train.npz") as cache:
        train_features, extra_dim = cached_features(
            cache, variant, datasets.train.model_windows.target_dates
        )
    with np.load(feature_dir / "validation.npz") as cache:
        val_features, val_extra_dim = cached_features(
            cache, variant, datasets.val.model_windows.target_dates
        )
    if extra_dim != val_extra_dim:
        raise ValueError("Train/validation feature dimensions differ")
    train_dataset = FeatureDataset(train_features, datasets.train.model_windows.target)
    val_dataset = FeatureDataset(val_features, datasets.val.model_windows.target)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size,
        shuffle=True, generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    model = PrecomputedConditionDiffusion(
        build_diffusion(cfg, n_assets=len(scaler.columns), precomputed=True),
        extra_dim=extra_dim,
    )
    set_global_seed(cfg.seed, deterministic=False)
    for filename in ("config.yaml", "standardizer.npz"):
        shutil.copy2(cde_run_dir / filename, output_dir / filename)
    result = fit_and_record(
        model, train_loader, val_loader, cfg, output_dir, device=device, resume=resume
    )
    summary = {
        "variant": variant, "seed": int(cfg.seed), "extra_dim": extra_dim,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_samples": len(train_dataset), "val_samples": len(val_dataset),
        "best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Variant:", variant, "Best validation loss:", result.best_val_loss)
    return result
