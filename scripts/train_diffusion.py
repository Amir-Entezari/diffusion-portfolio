"""Train the vanilla conditional diffusion baseline on KF12."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import (
    TrainStandardizer,
    build_window_datasets,
    load_daily_risk_free,
    load_kf12_daily,
    make_dataloaders,
    slice_return_table,
    to_excess_returns,
)
from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)
from diffusion_portfolio.training import (
    fit_diffusion,
)
from diffusion_portfolio.utils.seed import (
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/mvp.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/vanilla_diffusion",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return torch.device(
        requested
    )


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


def main() -> None:
    args = parse_args()

    config_path = Path(
        args.config
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.resume
        and (
            output_dir
            / "summary.json"
        ).exists()
    ):
        print(
            "Training already complete:",
            output_dir,
        )
        return

    cfg = load_config(
        config_path
    )

    set_global_seed(
        cfg.seed,
        deterministic=False,
    )

    device = resolve_device(
        args.device
    )

    print("=" * 72)
    print(
        "VANILLA CONDITIONAL DIFFUSION TRAINING"
    )
    print("=" * 72)

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    print(
        "Prediction type:",
        cfg.model.prediction_type,
    )
    print(
        "History encoder:",
        cfg.model.history_encoder,
    )
    print(
        "Diffusion steps:",
        cfg.model.diffusion_steps,
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    print()
    print("Loading KF12 data...")

    assets = slice_return_table(
        load_kf12_daily(),
        start=cfg.data.sample_start,
        end=cfg.data.sample_end,
    )

    risk_free = (
        load_daily_risk_free()
    )

    excess = to_excess_returns(
        assets,
        risk_free,
    )

    scaler = TrainStandardizer.fit(
        excess,
        train_end=cfg.data.train_end,
    )

    scaled = scaler.transform(
        excess
    )

    datasets = build_window_datasets(
        scaled,
        excess,
        lookback=cfg.data.lookback,
        horizon=cfg.data.horizon,
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
        test_end=cfg.data.sample_end,
    )

    loaders = make_dataloaders(
        datasets,
        batch_size=(
            cfg.training.batch_size
        ),
        seed=cfg.seed,
        num_workers=0,
    )

    print(
        "Train samples:",
        len(datasets.train),
    )

    print(
        "Validation samples:",
        len(datasets.val),
    )

    print(
        "Test samples:",
        len(datasets.test),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = ConditionalDiffusionModel(
        lookback=cfg.data.lookback,
        n_assets=12,
        condition_dim=(
            cfg.model.condition_dim
        ),
        history_hidden_dim=(
            cfg.model.history_hidden_dim
        ),
        diffusion_steps=(
            cfg.model.diffusion_steps
        ),
        schedule_type=(
            cfg.model.schedule
        ),
        prediction_type=(
            cfg.model.prediction_type
        ),
        channels=list(
            cfg.model.channels
        ),
        time_embed_dim=(
            cfg.model.time_embed_dim
        ),
        n_res_blocks=(
            cfg.model.n_res_blocks
        ),
        history_encoder_type=(
            cfg.model.history_encoder
        ),
        cde_hidden_dim=(
            cfg.model.cde_hidden_dim
        ),
        cde_drift_hidden_dim=(
            cfg.model.cde_drift_hidden_dim
        ),
        cde_sensitivity_hidden_dim=(
            cfg.model
            .cde_sensitivity_hidden_dim
        ),
        cde_solver=(
            cfg.model.cde_solver
        ),
        cde_rtol=(
            cfg.model.cde_rtol
        ),
        cde_atol=(
            cfg.model.cde_atol
        ),
                cde_use_adjoint=(
            cfg.model.cde_use_adjoint
        ),
        cde_fixed_steps_per_interval=(
            cfg.model
            .cde_fixed_steps_per_interval
        ),
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    # ---------------------------------------------------------
    # Persist exact experiment inputs
    # ---------------------------------------------------------
    shutil.copy2(
        config_path,
        output_dir
        / "config.yaml",
    )

    np.savez(
        output_dir
        / "standardizer.npz",
        mean=scaler.mean,
        std=scaler.std,
        columns=np.asarray(
            scaler.columns
        ),
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    print()
    print("Training...")

    checkpoint_path = (
        output_dir
        / "best.pt"
    )

    result = fit_diffusion(
        model,
        loaders.train,
        loaders.val,
        epochs=cfg.training.epochs,
        learning_rate=(
            cfg.training.learning_rate
        ),
        weight_decay=(
            cfg.training.weight_decay
        ),
        gradient_clip_norm=(
            cfg.training.gradient_clip_norm
        ),
        validation_seed=(
            cfg.training.validation_seed
        ),
        device=device,
        checkpoint_path=(
            checkpoint_path
        ),
        resume_path=(
            output_dir
            / "last.pt"
            if args.resume
            else None
        ),
        stop_after_epoch=(
            args.stop_after_epoch
        ),
        verbose=True,
    )

    # ---------------------------------------------------------
    # Save loss history
    # ---------------------------------------------------------
    history_frame = pd.DataFrame(
        [
            {
                "epoch": record.epoch,
                "train_loss": (
                    record.train_loss
                ),
                "val_loss": (
                    record.val_loss
                ),
            }
            for record
            in result.history
        ]
    )

    history_frame.to_csv(
        output_dir
        / "history.csv",
        index=False,
    )
    if (
        args.stop_after_epoch is not None
        and result.history
        and result.history[-1].epoch
        < cfg.training.epochs
    ):
        print()
        print("=" * 72)
        print("TRAINING CHUNK COMPLETE")
        print("=" * 72)

        print(
            "Completed through epoch:",
            result.history[-1].epoch,
        )

        print(
            "Best epoch so far:",
            result.best_epoch,
        )

        print(
            "Best validation loss so far:",
            result.best_val_loss,
        )

        print(
            "Resume checkpoint:",
            output_dir / "last.pt",
        )

        return

    # ---------------------------------------------------------
    # Small VALIDATION sampling sanity check.
    # ---------------------------------------------------------
    # Small VALIDATION sampling sanity check.
    #
    # This is not the final probabilistic evaluation.
    # The test split remains untouched.
    # ---------------------------------------------------------
    validation_sample = (
        datasets.val[0]
    )

    history = (
        validation_sample[
            "history"
        ]
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    with torch.no_grad():
        generated = model.sample(
            history,
            n_scenarios=64,
        )

    generated_standardized = (
        generated
        .detach()
        .cpu()
        .numpy()
    )

    generated_raw = (
        scaler.inverse_transform(
            generated_standardized
        )
    )

    sampling_summary = {
        "standardized_mean": float(
            generated_standardized.mean()
        ),
        "standardized_std": float(
            generated_standardized.std()
        ),
        "standardized_max_abs": float(
            np.abs(
                generated_standardized
            ).max()
        ),
        "raw_mean": float(
            generated_raw.mean()
        ),
        "raw_std": float(
            generated_raw.std()
        ),
        "raw_max_abs": float(
            np.abs(
                generated_raw
            ).max()
        ),
    }

    summary = {
        "git_commit": git_commit(),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(
                device
            )
            if device.type == "cuda"
            else None
        ),
        "prediction_type": (
            cfg.model.prediction_type
        ),
        "parameter_count": (
            parameter_count
        ),
        "train_samples": (
            len(datasets.train)
        ),
        "val_samples": (
            len(datasets.val)
        ),
        "test_samples": (
            len(datasets.test)
        ),
        "best_epoch": (
            result.best_epoch
        ),
        "best_val_loss": (
            result.best_val_loss
        ),
        "sampling_sanity": (
            sampling_summary
        ),
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        "Best epoch:",
        result.best_epoch,
    )

    print(
        "Best validation loss:",
        result.best_val_loss,
    )

    print()
    print(
        "Validation sampling sanity:"
    )

    for key, value in (
        sampling_summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()
    print(
        "Artifacts:",
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()