"""Train Phase-3 end-to-end CDE + alpha-stable diffusion."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from diffusion_portfolio.config import (
    load_config,
)
from diffusion_portfolio.data import (
    make_dataloaders,
)
from diffusion_portfolio.models.levy import (
    ConditionalLevyDiffusionModel,
)
from diffusion_portfolio.training import (
    fit_diffusion,
)
from diffusion_portfolio.utils.seed import (
    set_global_seed,
)

from train_evidential_regime import (
    prepare_datasets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--config",
        default="configs/phase3_levy.yaml",
    )

    parser.add_argument(
        "--phase0-run-dir",
        default="/kaggle/working/phase0_cde",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
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
            "CUDA requested but unavailable"
        )

    return torch.device(
        requested
    )


def build_model(
    cfg,
    *,
    alpha: float,
) -> ConditionalLevyDiffusionModel:
    return ConditionalLevyDiffusionModel(
        alpha=alpha,
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


def main() -> None:
    args = parse_args()

    alpha = float(
        args.alpha
    )

    if not (
        1.0
        < alpha
        <= 2.0
    ):
        raise ValueError(
            "alpha must satisfy "
            "1 < alpha <= 2"
        )

    config_path = Path(
        args.config
    )

    phase0_run_dir = Path(
        args.phase0_run_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = load_config(
        config_path
    )

    if (
        cfg.model.prediction_type
        != "epsilon"
    ):
        raise RuntimeError(
            "Phase-3 config must use "
            "prediction_type=epsilon"
        )

    standardizer_path = (
        phase0_run_dir
        / "standardizer.npz"
    )

    if not standardizer_path.exists():
        raise FileNotFoundError(
            standardizer_path
        )

    metadata_path = (
        output_dir
        / "experiment.json"
    )

    metadata = {
        "phase": "3A",
        "alpha": alpha,
        "prediction_type": (
            "epsilon"
        ),
        "loss": (
            "mean_per_sample_rmse"
        ),
        "history_encoder": (
            cfg.model.history_encoder
        ),
        "seed": int(
            cfg.seed
        ),
    }

    if metadata_path.exists():
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            existing = json.load(
                handle
            )

        if float(
            existing["alpha"]
        ) != alpha:
            raise RuntimeError(
                "Existing output directory "
                "belongs to another alpha"
            )

    else:
        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metadata,
                handle,
                indent=2,
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

    device = resolve_device(
        args.device
    )

    print("=" * 72)
    print(
        "PHASE 3A — CDE + ALPHA-STABLE DIFFUSION"
    )
    print("=" * 72)

    print(
        "Alpha:",
        alpha,
    )

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
        "Objective:",
        "mean per-sample RMSE",
    )

    print(
        "History encoder:",
        cfg.model.history_encoder,
    )

    # ---------------------------------------------------------
    # Exact Phase-0 preprocessing / chronological splits.
    # ---------------------------------------------------------

    datasets = prepare_datasets(
        cfg,
        standardizer_path=(
            standardizer_path
        ),
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
        len(
            datasets.train
        ),
    )

    print(
        "Validation samples:",
        len(
            datasets.val
        ),
    )

    # ---------------------------------------------------------
    # Identical parameter initialization for every alpha.
    # ---------------------------------------------------------

    set_global_seed(
        cfg.seed,
        deterministic=False,
    )

    model = build_model(
        cfg,
        alpha=alpha,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    # Model construction consumes RNG.
    #
    # Reset so both alpha runs begin training from the same
    # subsequent global RNG state. Data-loader shuffling has
    # its own generator and is already paired by cfg.seed.
    set_global_seed(
        cfg.seed,
        deterministic=False,
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    # ---------------------------------------------------------
    # Persist exact experiment inputs.
    # ---------------------------------------------------------

    shutil.copy2(
        config_path,
        output_dir
        / "config.yaml",
    )

    shutil.copy2(
        standardizer_path,
        output_dir
        / "standardizer.npz",
    )

    # ---------------------------------------------------------
    # Train.
    # ---------------------------------------------------------

    result = fit_diffusion(
        model,
        loaders.train,
        loaders.val,
        epochs=(
            cfg.training.epochs
        ),
        learning_rate=(
            cfg.training.learning_rate
        ),
        weight_decay=(
            cfg.training.weight_decay
        ),
        gradient_clip_norm=(
            cfg.training
            .gradient_clip_norm
        ),
        validation_seed=(
            cfg.training
            .validation_seed
        ),
        device=device,
        checkpoint_path=(
            output_dir
            / "best.pt"
        ),
        resume_path=(
            output_dir
            / "last.pt"
            if args.resume
            else None
        ),
        verbose=True,
    )

    pd.DataFrame(
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
    ).to_csv(
        output_dir
        / "history.csv",
        index=False,
    )

    summary = {
        **metadata,
        "parameter_count": (
            parameter_count
        ),
        "train_samples": len(
            datasets.train
        ),
        "val_samples": len(
            datasets.val
        ),
        "best_epoch": int(
            result.best_epoch
        ),
        "best_val_loss": float(
            result.best_val_loss
        ),
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "PHASE 3A TRAINING COMPLETE"
    )
    print("=" * 72)

    print(
        "Alpha:",
        alpha,
    )

    print(
        "Best epoch:",
        result.best_epoch,
    )

    print(
        "Best validation loss:",
        result.best_val_loss,
    )

    print(
        "Artifacts:",
        output_dir,
    )

    print()
    print(
        "TEST SPLIT WAS NOT EVALUATED."
    )


if __name__ == "__main__":
    main()