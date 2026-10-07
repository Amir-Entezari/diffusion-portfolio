"""Precompute covariance and Log-Euclidean SPD features for Phase 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.config import (
    load_config,
)
from diffusion_portfolio.data import (
    collate_return_batch,
)
from diffusion_portfolio.models.spd import (
    covariance_features,
    log_euclidean_spd_features,
    regularized_covariance,
)

from train_evidential_regime import (
    build_cde_model,
    prepare_datasets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cde-run-dir",
        default="/kaggle/working/phase0_cde",
    )

    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/phase2_features",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--eigenvalue-floor",
        type=float,
        default=1e-6,
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


def standardize_features(
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit feature-wise standardization on training only."""

    mean = train.mean(
        axis=0,
        dtype=np.float64,
    )

    std = train.std(
        axis=0,
        dtype=np.float64,
    )

    constant = (
        std <= 1e-12
    )

    safe_std = std.copy()

    safe_std[
        constant
    ] = 1.0

    train_scaled = (
        train - mean
    ) / safe_std

    validation_scaled = (
        validation - mean
    ) / safe_std

    return (
        train_scaled.astype(
            np.float32
        ),
        validation_scaled.astype(
            np.float32
        ),
        mean,
        std,
        constant,
    )


@torch.no_grad()
def extract_split(
    encoder,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    eigenvalue_floor: float,
) -> dict[str, np.ndarray]:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_return_batch,
    )

    conditions = []
    covariance_parts = []
    log_spd_parts = []

    min_eigenvalues = []
    max_eigenvalues = []
    condition_numbers = []

    dates = []

    encoder.eval()

    for batch in loader:
        history = batch[
            "history"
        ].to(
            device
        )

        history_raw = batch[
            "history_raw"
        ].to(
            device
        )

        condition = encoder(
            history
        )

        covariance = regularized_covariance(
            history_raw,
            relative_eigenvalue_floor=(
                eigenvalue_floor
            ),
        )

        eigenvalues = torch.linalg.eigvalsh(
            covariance
        )

        minimum = eigenvalues[
            :,
            0,
        ]

        maximum = eigenvalues[
            :,
            -1,
        ]

        condition_number = (
            maximum
            / minimum
        )

        covariance_vector = (
            covariance_features(
                history_raw,
                relative_eigenvalue_floor=(
                    eigenvalue_floor
                ),
            )
        )

        log_spd_vector = (
            log_euclidean_spd_features(
                history_raw,
                relative_eigenvalue_floor=(
                    eigenvalue_floor
                ),
            )
        )

        conditions.append(
            condition.cpu().numpy()
        )

        covariance_parts.append(
            covariance_vector
            .cpu()
            .numpy()
        )

        log_spd_parts.append(
            log_spd_vector
            .cpu()
            .numpy()
        )

        min_eigenvalues.append(
            minimum.cpu().numpy()
        )

        max_eigenvalues.append(
            maximum.cpu().numpy()
        )

        condition_numbers.append(
            condition_number
            .cpu()
            .numpy()
        )

        dates.append(
            batch[
                "target_date"
            ].values
        )

    return {
        "dates": np.concatenate(
            dates
        ),
        "cde_condition": np.concatenate(
            conditions
        ),
        "covariance_raw": np.concatenate(
            covariance_parts
        ),
        "log_spd_raw": np.concatenate(
            log_spd_parts
        ),
        "min_eigenvalue": np.concatenate(
            min_eigenvalues
        ),
        "max_eigenvalue": np.concatenate(
            max_eigenvalues
        ),
        "condition_number": np.concatenate(
            condition_numbers
        ),
    }


def spectral_summary(
    split: dict[str, np.ndarray],
) -> dict:
    minimum = split[
        "min_eigenvalue"
    ]

    maximum = split[
        "max_eigenvalue"
    ]

    condition = split[
        "condition_number"
    ]

    return {
        "min_eigenvalue_min": float(
            minimum.min()
        ),
        "min_eigenvalue_median": float(
            np.median(
                minimum
            )
        ),
        "max_eigenvalue_median": float(
            np.median(
                maximum
            )
        ),
        "condition_number_median": float(
            np.median(
                condition
            )
        ),
        "condition_number_p95": float(
            np.quantile(
                condition,
                0.95,
            )
        ),
        "condition_number_max": float(
            condition.max()
        ),
    }


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive"
        )

    if args.eigenvalue_floor <= 0:
        raise ValueError(
            "eigenvalue-floor must be positive"
        )

    cde_run_dir = Path(
        args.cde_run_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    cfg = load_config(
        cde_run_dir
        / "config.yaml"
    )

    checkpoint = torch.load(
        cde_run_dir
        / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = build_cde_model(
        cfg
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    encoder = (
        model.history_encoder
        .to(
            device
        )
        .eval()
    )

    for parameter in (
        encoder.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    datasets = prepare_datasets(
        cfg,
        standardizer_path=(
            cde_run_dir
            / "standardizer.npz"
        ),
    )

    print("=" * 72)
    print(
        "PHASE 2 SPD FEATURE PRECOMPUTATION"
    )
    print("=" * 72)

    print(
        "Device:",
        device,
    )

    print(
        "Phase 0 CDE epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "Eigenvalue floor:",
        args.eigenvalue_floor,
    )

    print()
    print(
        "Extracting training features..."
    )

    train = extract_split(
        encoder,
        datasets.train,
        batch_size=args.batch_size,
        device=device,
        eigenvalue_floor=(
            args.eigenvalue_floor
        ),
    )

    print(
        "Extracting validation features..."
    )

    validation = extract_split(
        encoder,
        datasets.val,
        batch_size=args.batch_size,
        device=device,
        eigenvalue_floor=(
            args.eigenvalue_floor
        ),
    )

    (
        train_covariance,
        val_covariance,
        covariance_mean,
        covariance_std,
        covariance_constant,
    ) = standardize_features(
        train[
            "covariance_raw"
        ],
        validation[
            "covariance_raw"
        ],
    )

    (
        train_log_spd,
        val_log_spd,
        log_spd_mean,
        log_spd_std,
        log_spd_constant,
    ) = standardize_features(
        train[
            "log_spd_raw"
        ],
        validation[
            "log_spd_raw"
        ],
    )

    if (
        train_covariance.shape[1]
        != train_log_spd.shape[1]
    ):
        raise RuntimeError(
            "Covariance and Log-SPD feature "
            "dimensions must match"
        )

    if not np.isfinite(
        train_covariance
    ).all():
        raise RuntimeError(
            "Training covariance features "
            "contain non-finite values"
        )

    if not np.isfinite(
        train_log_spd
    ).all():
        raise RuntimeError(
            "Training Log-SPD features "
            "contain non-finite values"
        )

    if not np.isfinite(
        val_covariance
    ).all():
        raise RuntimeError(
            "Validation covariance features "
            "contain non-finite values"
        )

    if not np.isfinite(
        val_log_spd
    ).all():
        raise RuntimeError(
            "Validation Log-SPD features "
            "contain non-finite values"
        )

    np.savez_compressed(
        output_dir
        / "train.npz",
        dates=train[
            "dates"
        ],
        cde_condition=train[
            "cde_condition"
        ],
        covariance=train_covariance,
        log_spd=train_log_spd,
        covariance_raw=train[
            "covariance_raw"
        ],
        log_spd_raw=train[
            "log_spd_raw"
        ],
    )

    np.savez_compressed(
        output_dir
        / "validation.npz",
        dates=validation[
            "dates"
        ],
        cde_condition=validation[
            "cde_condition"
        ],
        covariance=val_covariance,
        log_spd=val_log_spd,
        covariance_raw=validation[
            "covariance_raw"
        ],
        log_spd_raw=validation[
            "log_spd_raw"
        ],
    )

    metadata = {
        "phase0_checkpoint_epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "phase0_checkpoint_val_loss": float(
            checkpoint[
                "val_loss"
            ]
        ),
        "relative_eigenvalue_floor": float(
            args.eigenvalue_floor
        ),
        "cde_condition_dim": int(
            train[
                "cde_condition"
            ].shape[1]
        ),
        "covariance_dim": int(
            train_covariance.shape[1]
        ),
        "log_spd_dim": int(
            train_log_spd.shape[1]
        ),
        "covariance_constant_indices": (
            np.flatnonzero(
                covariance_constant
            ).tolist()
        ),
        "log_spd_constant_indices": (
            np.flatnonzero(
                log_spd_constant
            ).tolist()
        ),
        "covariance_mean": (
            covariance_mean.tolist()
        ),
        "covariance_std": (
            covariance_std.tolist()
        ),
        "log_spd_mean": (
            log_spd_mean.tolist()
        ),
        "log_spd_std": (
            log_spd_std.tolist()
        ),
        "train_spectrum": (
            spectral_summary(
                train
            )
        ),
        "validation_spectrum": (
            spectral_summary(
                validation
            )
        ),
    }

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print()
    print(
        "Train samples:",
        len(
            train[
                "dates"
            ]
        ),
    )

    print(
        "Validation samples:",
        len(
            validation[
                "dates"
            ]
        ),
    )

    print(
        "CDE condition:",
        train[
            "cde_condition"
        ].shape[1],
    )

    print(
        "Covariance features:",
        train_covariance.shape[1],
    )

    print(
        "Log-SPD features:",
        train_log_spd.shape[1],
    )

    print(
        "Covariance constant columns:",
        np.flatnonzero(
            covariance_constant
        ).tolist(),
    )

    print(
        "Log-SPD constant columns:",
        np.flatnonzero(
            log_spd_constant
        ).tolist(),
    )

    print()
    print(
        "Train spectrum:",
        spectral_summary(
            train
        ),
    )

    print(
        "Validation spectrum:",
        spectral_summary(
            validation
        ),
    )

    print()
    print(
        "Artifacts:",
        output_dir,
    )

    print(
        "TEST SPLIT WAS NOT TOUCHED."
    )


if __name__ == "__main__":
    main()