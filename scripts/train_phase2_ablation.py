"""Train controlled Phase-2 covariance/SPD ablations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.config import load_config
from diffusion_portfolio.models.topology import (
    PrecomputedConditionDiffusion,
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

from train_phase1_ablation import (
    FeatureDataset,
    build_diffusion,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        choices=[
            "cde",
            "covariance",
            "log_spd",
        ],
        required=True,
    )

    parser.add_argument(
        "--cde-run-dir",
        default="/kaggle/working/phase0_cde",
    )

    parser.add_argument(
        "--feature-dir",
        default="/kaggle/working/phase2_features",
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


def main() -> None:
    args = parse_args()

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

    cde_run_dir = Path(
        args.cde_run_dir
    )

    feature_dir = Path(
        args.feature_dir
    )

    cfg = load_config(
        cde_run_dir
        / "config.yaml"
    )

    device = resolve_device(
        args.device
    )

    set_global_seed(
        cfg.seed,
        deterministic=False,
    )

    train_cache = np.load(
        feature_dir
        / "train.npz"
    )

    val_cache = np.load(
        feature_dir
        / "validation.npz"
    )

    datasets = prepare_datasets(
        cfg,
        standardizer_path=(
            cde_run_dir
            / "standardizer.npz"
        ),
    )

    train_dates = (
        datasets.train
        .model_windows
        .target_dates
        .values
    )

    val_dates = (
        datasets.val
        .model_windows
        .target_dates
        .values
    )

    if not np.array_equal(
        train_cache["dates"],
        train_dates,
    ):
        raise RuntimeError(
            "Cached train dates do not "
            "match canonical dataset"
        )

    if not np.array_equal(
        val_cache["dates"],
        val_dates,
    ):
        raise RuntimeError(
            "Cached validation dates do not "
            "match canonical dataset"
        )

    train_condition = train_cache[
        "cde_condition"
    ]

    val_condition = val_cache[
        "cde_condition"
    ]

    if args.variant == "cde":
        train_features = (
            train_condition
        )

        val_features = (
            val_condition
        )

        extra_dim = 0

    elif args.variant == "covariance":
        train_extra = train_cache[
            "covariance"
        ]

        val_extra = val_cache[
            "covariance"
        ]

        train_features = np.concatenate(
            [
                train_condition,
                train_extra,
            ],
            axis=1,
        )

        val_features = np.concatenate(
            [
                val_condition,
                val_extra,
            ],
            axis=1,
        )

        extra_dim = int(
            train_extra.shape[1]
        )

    else:
        train_extra = train_cache[
            "log_spd"
        ]

        val_extra = val_cache[
            "log_spd"
        ]

        train_features = np.concatenate(
            [
                train_condition,
                train_extra,
            ],
            axis=1,
        )

        val_features = np.concatenate(
            [
                val_condition,
                val_extra,
            ],
            axis=1,
        )

        extra_dim = int(
            train_extra.shape[1]
        )

    train_dataset = FeatureDataset(
        train_features,
        datasets.train
        .model_windows
        .target,
    )

    val_dataset = FeatureDataset(
        val_features,
        datasets.val
        .model_windows
        .target,
    )

    generator = torch.Generator()

    generator.manual_seed(
        cfg.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    diffusion = build_diffusion(
        cfg
    )

    model = PrecomputedConditionDiffusion(
        diffusion,
        extra_dim=extra_dim,
    )

    # Variant-specific residual layer construction
    # consumes RNG. Reset so all three runs receive
    # the same subsequent stochastic training stream.
    set_global_seed(
        cfg.seed,
        deterministic=False,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print("=" * 72)

    print(
        "PHASE 2 FROZEN-CDE SPD ABLATION"
    )

    print("=" * 72)

    print(
        "Variant:",
        args.variant,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Input dimension:",
        train_features.shape[1],
    )

    print(
        "Extra dimension:",
        extra_dim,
    )

    print(
        "Parameters:",
        parameter_count,
    )

    shutil.copy2(
        cde_run_dir
        / "config.yaml",
        output_dir
        / "config.yaml",
    )

    shutil.copy2(
        cde_run_dir
        / "standardizer.npz",
        output_dir
        / "standardizer.npz",
    )

    result = fit_diffusion(
        model,
        train_loader,
        val_loader,
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
            for record in result.history
        ]
    ).to_csv(
        output_dir
        / "history.csv",
        index=False,
    )

    summary = {
        "variant": args.variant,
        "seed": int(
            cfg.seed
        ),
        "extra_dim": extra_dim,
        "parameter_count": (
            parameter_count
        ),
        "train_samples": len(
            train_dataset
        ),
        "val_samples": len(
            val_dataset
        ),
        "best_epoch": (
            result.best_epoch
        ),
        "best_val_loss": (
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

    print(
        "TEST SPLIT WAS NOT EVALUATED."
    )


if __name__ == "__main__":
    main()