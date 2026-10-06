"""Validation-only probabilistic evaluation for Phase-1 ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import TrainStandardizer
from diffusion_portfolio.evaluation import (
    evaluate_probabilistic_forecast,
)
from diffusion_portfolio.models.topology import (
    PrecomputedConditionDiffusion,
)

from train_evidential_regime import (
    prepare_datasets,
)
from train_phase1_ablation import (
    build_diffusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        choices=[
            "cde",
            "geometry",
            "tda",
        ],
        required=True,
    )

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    parser.add_argument(
        "--cde-run-dir",
        default="/kaggle/working/phase0_cde",
    )

    parser.add_argument(
        "--feature-dir",
        default="/kaggle/working/phase1_features",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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


def main() -> None:
    args = parse_args()

    run_dir = Path(
        args.run_dir
    )

    cde_run_dir = Path(
        args.cde_run_dir
    )

    feature_dir = Path(
        args.feature_dir
    )

    device = resolve_device(
        args.device
    )

    cfg = load_config(
        cde_run_dir
        / "config.yaml"
    )

    validation_cache = np.load(
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

    dates = (
        datasets.val
        .raw_windows
        .target_dates
    )

    if not np.array_equal(
        validation_cache["dates"],
        dates.values,
    ):
        raise RuntimeError(
            "Cached validation dates do not "
            "match canonical dataset"
        )

    cde_condition = validation_cache[
        "cde_condition"
    ]

    if args.variant == "cde":
        features = cde_condition
        extra_dim = 0

    elif args.variant == "geometry":
        extra = validation_cache[
            "geometry"
        ]

        features = np.concatenate(
            [
                cde_condition,
                extra,
            ],
            axis=1,
        )

        extra_dim = int(
            extra.shape[1]
        )

    else:
        extra = validation_cache[
            "tda"
        ]

        features = np.concatenate(
            [
                cde_condition,
                extra,
            ],
            axis=1,
        )

        extra_dim = int(
            extra.shape[1]
        )

    feature_tensor = torch.from_numpy(
        features
    ).float()

    loader = DataLoader(
        TensorDataset(
            feature_tensor
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    diffusion = build_diffusion(
        cfg
    )

    model = PrecomputedConditionDiffusion(
        diffusion,
        extra_dim=extra_dim,
    ).to(
        device
    )

    checkpoint = torch.load(
        run_dir
        / "best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    scaler_file = np.load(
        run_dir
        / "standardizer.npz",
        allow_pickle=False,
    )

    scaler = TrainStandardizer(
        mean=scaler_file[
            "mean"
        ],
        std=scaler_file[
            "std"
        ],
        columns=tuple(
            str(column)
            for column
            in scaler_file[
                "columns"
            ]
        ),
    )

    observed_raw = (
        datasets.val
        .raw_windows
        .target[
            :,
            0,
            :,
        ]
        .astype(
            np.float32,
            copy=True,
        )
    )

    n_validation = len(
        feature_tensor
    )

    n_scenarios = int(
        cfg.evaluation.n_scenarios
    )

    n_assets = len(
        scaler.columns
    )

    sampling_seed = int(
        cfg.training.validation_seed
    )

    torch.manual_seed(
        sampling_seed
    )

    np.random.seed(
        sampling_seed
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            sampling_seed
        )

    forecast_raw = np.empty(
        (
            n_validation,
            n_scenarios,
            n_assets,
        ),
        dtype=np.float32,
    )

    offset = 0

    print("=" * 72)
    print(
        "PHASE 1 VALIDATION EVALUATION"
    )
    print("=" * 72)

    print(
        "Variant:",
        args.variant,
    )

    print(
        "Checkpoint epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "Checkpoint validation loss:",
        checkpoint[
            "val_loss"
        ],
    )

    print(
        "Validation samples:",
        n_validation,
    )

    print(
        "Scenarios per date:",
        n_scenarios,
    )

    print()

    with torch.inference_mode():
        for (batch_features,) in tqdm(
            loader,
            desc=args.variant,
        ):
            batch_features = (
                batch_features.to(
                    device
                )
            )

            batch_size = (
                batch_features.shape[0]
            )

            generated = model.sample(
                batch_features,
                n_scenarios=n_scenarios,
            )

            generated_raw = (
                scaler.inverse_transform(
                    generated
                    .cpu()
                    .numpy()
                )
            )

            end = (
                offset
                + batch_size
            )

            forecast_raw[
                offset:end
            ] = generated_raw

            offset = end

    if offset != n_validation:
        raise RuntimeError(
            "Did not generate every "
            "validation sample"
        )

    if not np.isfinite(
        forecast_raw
    ).all():
        raise RuntimeError(
            "Generated scenarios contain "
            "non-finite values"
        )

    metrics = (
        evaluate_probabilistic_forecast(
            forecast_raw,
            observed_raw,
        )
    )

    result = {
        "variant": args.variant,
        "split": "validation",
        "sampling_seed": (
            sampling_seed
        ),
        "checkpoint_epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "checkpoint_val_loss": float(
            checkpoint[
                "val_loss"
            ]
        ),
        "n_validation": (
            n_validation
        ),
        "n_scenarios": (
            n_scenarios
        ),
        "crps_mean": float(
            metrics.crps_mean
        ),
        "crps_std": float(
            metrics.crps_std
        ),
        "energy_score": float(
            metrics.energy_score
        ),
        "calibration": [
            {
                "level": float(
                    level
                ),
                "picp": float(
                    picp
                ),
                "ace": float(
                    ace
                ),
            }
            for level, picp, ace
            in zip(
                metrics.calibration.levels,
                metrics.calibration.picp,
                metrics.calibration.ace,
            )
        ],
    }

    np.savez_compressed(
        run_dir
        / "validation_scenarios.npz",
        scenarios=forecast_raw,
        observed=observed_raw,
        dates=dates.values.astype(
            "datetime64[D]"
        ),
        columns=np.asarray(
            scaler.columns
        ),
    )

    with (
        run_dir
        / "validation_probabilistic_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    print()
    print(
        "CRPS:",
        result[
            "crps_mean"
        ],
    )

    print(
        "Energy Score:",
        result[
            "energy_score"
        ],
    )

    print()
    print(
        "Calibration:"
    )

    for item in result[
        "calibration"
    ]:
        print(
            item
        )

    print()
    print(
        "TEST SPLIT WAS NOT EVALUATED."
    )


if __name__ == "__main__":
    main()