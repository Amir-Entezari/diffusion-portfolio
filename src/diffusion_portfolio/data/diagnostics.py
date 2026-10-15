"""Data and optional one-batch model smoke checks through shared preparation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import collate_return_batch, make_dataloaders
from diffusion_portfolio.data.preparation import prepare_return_data


def run_data_check(args) -> dict:
    cfg = load_config(args.config)
    datasets, scaler = prepare_return_data(cfg)
    loaders = make_dataloaders(
        datasets, batch_size=cfg.training.batch_size, seed=cfg.seed,
    )
    batch = next(iter(loaders.val))
    n_assets = len(scaler.columns)
    batch_size = len(batch["target_date"])
    history_shape = (batch_size, cfg.data.lookback, n_assets)
    target_shape = (batch_size, cfg.data.horizon, n_assets)
    for key in ("history", "history_raw"):
        assert tuple(batch[key].shape) == history_shape
        assert np.isfinite(batch[key].numpy()).all()
    for key in ("target", "target_raw"):
        assert tuple(batch[key].shape) == target_shape
        assert np.isfinite(batch[key].numpy()).all()
    assert batch["target_date"].is_monotonic_increasing
    recovered = scaler.inverse_transform(batch["target"].numpy())
    inverse_error = float(np.max(np.abs(recovered - batch["target_raw"].numpy())))
    assert inverse_error < 1e-6

    train_dates = datasets.train.raw_windows.target_dates
    val_dates = datasets.val.raw_windows.target_dates
    test_dates = datasets.test.raw_windows.target_dates
    assert train_dates[-1] <= pd.Timestamp(cfg.data.train_end) < val_dates[0]
    assert val_dates[-1] <= pd.Timestamp(cfg.data.val_end) < test_dates[0]
    assert test_dates[-1] <= pd.Timestamp(cfg.data.sample_end)
    report = {
        "dataset": cfg.data.dataset, "seed": cfg.seed,
        "n_assets": n_assets, "columns": list(scaler.columns),
        "train_samples": len(datasets.train), "validation_samples": len(datasets.val),
        "test_samples": len(datasets.test), "max_gap_days": cfg.data.max_gap_days,
        "history_shape": list(history_shape), "target_shape": list(target_shape),
        "target_ranges": {
            name: [str(dates[0].date()), str(dates[-1].date())]
            for name, dates in (("train", train_dates), ("validation", val_dates), ("test", test_dates))
        },
        "standardizer": {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()},
        "inverse_raw_max_error": inverse_error, "model_smoke": None,
    }

    print("Data smoke test passed")
    print("dataset:", cfg.data.dataset)
    print("raw convention:", "arithmetic excess returns" if cfg.data.subtract_risk_free
          else "arithmetic asset returns")
    print("source options:", cfg.data.source_options)
    print("assets:", n_assets)
    print("sample requested:", cfg.data.sample_start, "->", cfg.data.sample_end)
    print("usable windows:", sum(map(len, (datasets.train, datasets.val, datasets.test))))
    print("continuity: segment at gaps exceeding", cfg.data.max_gap_days, "calendar days")
    # Match actual diffusion preparation, including its legacy scaling convention.
    print("model scaling: train_zscore (configured:", cfg.data.return_scaling + ")")
    for name, dataset in (("train", datasets.train), ("validation", datasets.val),
                          ("test", datasets.test)):
        dates = dataset.raw_windows.target_dates
        print(f"{name}: {len(dataset)} windows; targets {dates[0].date()} -> {dates[-1].date()}")
    dates = train_dates.append([val_dates, test_dates])
    gaps = dates[1:] - dates[:-1]
    largest_gap = int(np.argmax(gaps))
    print(f"largest retained target-date gap: {gaps[largest_gap].days} calendar days "
          f"({dates[largest_gap].date()} -> {dates[largest_gap + 1].date()})")
    print("validation history:", tuple(batch["history"].shape))
    print("validation target:", tuple(batch["target"].shape))
    print("inverse/raw max error:", inverse_error)

    if getattr(args, "model_smoke", False):
        import torch
        from diffusion_portfolio.models.diffusion.build import build_diffusion
        from diffusion_portfolio.utils.device import resolve_device
        from diffusion_portfolio.utils.seed import set_global_seed

        count = getattr(args, "smoke_batch_size", 2)
        if count <= 0:
            raise ValueError("smoke-batch-size must be positive")
        device = resolve_device(getattr(args, "device", "cpu"))
        set_global_seed(cfg.seed, deterministic=False)
        model = build_diffusion(cfg, n_assets=n_assets).to(device)
        # Training examples only. No optimizer update, sampling or checkpoints.
        train_batch = collate_return_batch([
            datasets.train[i] for i in range(min(count, len(datasets.train)))
        ])
        assert all(np.isfinite(train_batch[key].numpy()).all() for key in ("history", "target"))
        output = model.training_loss(
            train_batch["history"].to(device), train_batch["target"].to(device),
        )
        assert output.prediction.shape == (len(train_batch["target"]), n_assets)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        report["model_smoke"] = {
            "loss": float(output.loss.detach()), "gradients_finite": True,
            "prediction_shape": list(output.prediction.shape), "device": str(device),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        print("Model forward/loss/backward smoke passed; training loss:", float(output.loss.detach()))
    return report
