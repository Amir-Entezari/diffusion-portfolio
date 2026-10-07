"""Validation-only evaluation for Phase-3 alpha-stable diffusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_portfolio.config import (
    load_config,
)
from diffusion_portfolio.data import (
    TrainStandardizer,
    collate_return_batch,
)
from diffusion_portfolio.evaluation import (
    evaluate_probabilistic_forecast,
)

from train_evidential_regime import (
    prepare_datasets,
)
from train_phase3_levy import (
    build_model,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
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


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive"
        )

    run_dir = Path(
        args.run_dir
    )

    config_path = (
        run_dir
        / "config.yaml"
    )

    checkpoint_path = (
        run_dir
        / "best.pt"
    )

    standardizer_path = (
        run_dir
        / "standardizer.npz"
    )

    experiment_path = (
        run_dir
        / "experiment.json"
    )

    for path in (
        config_path,
        checkpoint_path,
        standardizer_path,
        experiment_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    cfg = load_config(
        config_path
    )

    with experiment_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        experiment = json.load(
            handle
        )

    alpha = float(
        experiment[
            "alpha"
        ]
    )

    if not (
        1.0
        < alpha
        <= 2.0
    ):
        raise RuntimeError(
            "Saved experiment has "
            "invalid alpha"
        )

    if (
        cfg.model.prediction_type
        != "epsilon"
    ):
        raise RuntimeError(
            "Phase-3 evaluation requires "
            "epsilon prediction"
        )

    device = resolve_device(
        args.device
    )

    print("=" * 72)
    print(
        "PHASE 3A VALIDATION EVALUATION"
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
        "Scenarios per date:",
        cfg.evaluation.n_scenarios,
    )

    print(
        "Inference batch size:",
        args.batch_size,
    )

    # ========================================================
    # Exact preprocessing.
    # ========================================================

    scaler_file = np.load(
        standardizer_path,
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

    datasets = prepare_datasets(
        cfg,
        standardizer_path=(
            standardizer_path
        ),
    )

    validation_loader = DataLoader(
        datasets.val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_return_batch,
        drop_last=False,
    )

    n_validation = len(
        datasets.val
    )

    n_scenarios = int(
        cfg.evaluation.n_scenarios
    )

    n_assets = len(
        scaler.columns
    )

    dates = (
        datasets.val
        .raw_windows
        .target_dates
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

    if n_validation <= 0:
        raise RuntimeError(
            "Validation split is empty"
        )

    if (
        dates[0]
        <= np.datetime64(
            cfg.data.train_end
        )
    ):
        raise RuntimeError(
            "Validation overlaps training"
        )

    if (
        dates[-1]
        > np.datetime64(
            cfg.data.val_end
        )
    ):
        raise RuntimeError(
            "Validation extends past val_end"
        )

    print()

    print(
        "Validation samples:",
        n_validation,
    )

    print(
        "Validation start:",
        dates[0],
    )

    print(
        "Validation end:",
        dates[-1],
    )

    # ========================================================
    # Restore exact Phase-3 model.
    # ========================================================

    model = build_model(
        cfg,
        alpha=alpha,
    ).to(
        device
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print()

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

    # Seed only after model reconstruction so construction
    # cannot affect the Monte-Carlo stream.
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

    # ========================================================
    # Validation scenario generation.
    # ========================================================

    forecast_raw = np.empty(
        (
            n_validation,
            n_scenarios,
            n_assets,
        ),
        dtype=np.float32,
    )

    offset = 0

    print()
    print(
        "Generating VALIDATION scenarios..."
    )

    with torch.inference_mode():
        for batch in tqdm(
            validation_loader,
            desc=(
                f"alpha={alpha:g}"
            ),
        ):
            history = batch[
                "history"
            ].to(
                device,
                non_blocking=True,
            )

            batch_size = (
                history.shape[0]
            )

            generated = model.sample(
                history,
                n_scenarios=n_scenarios,
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
            "Validation scenarios contain "
            "non-finite values"
        )

    # ========================================================
    # Save scenarios.
    # ========================================================

    scenario_path = (
        run_dir
        / "validation_scenarios.npz"
    )

    np.savez(
        scenario_path,
        scenarios=forecast_raw,
        observed=observed_raw,
        dates=dates.values.astype(
            "datetime64[D]"
        ),
        columns=np.asarray(
            scaler.columns
        ),
    )

    print()

    print(
        "Scenario cube:",
        forecast_raw.shape,
    )

    print(
        "Raw scenario mean:",
        float(
            forecast_raw.mean()
        ),
    )

    print(
        "Raw scenario std:",
        float(
            forecast_raw.std()
        ),
    )

    print(
        "Raw scenario max abs:",
        float(
            np.abs(
                forecast_raw
            ).max()
        ),
    )

    # ========================================================
    # Standard probabilistic evaluation.
    # ========================================================

    metrics = (
        evaluate_probabilistic_forecast(
            forecast_raw,
            observed_raw,
        )
    )

    mean_abs_ace = float(
        np.mean(
            np.abs(
                metrics
                .calibration
                .ace
            )
        )
    )

    result = {
        "split": "validation",
        "phase": "3A",
        "alpha": alpha,
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
        "sampling_batch_size": (
            args.batch_size
        ),
        "validation_start": str(
            dates[
                0
            ].date()
        ),
        "validation_end": str(
            dates[
                -1
            ].date()
        ),
        "scenario_mean": float(
            forecast_raw.mean()
        ),
        "scenario_std": float(
            forecast_raw.std()
        ),
        "scenario_max_abs": float(
            np.abs(
                forecast_raw
            ).max()
        ),
        "crps_by_asset": {
            column: float(
                value
            )
            for column, value
            in zip(
                scaler.columns,
                metrics.crps_by_asset,
            )
        },
        "crps_mean": float(
            metrics.crps_mean
        ),
        "crps_std": float(
            metrics.crps_std
        ),
        "energy_score": float(
            metrics.energy_score
        ),
        "mean_abs_ace": (
            mean_abs_ace
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
            for (
                level,
                picp,
                ace,
            ) in zip(
                metrics.calibration.levels,
                metrics.calibration.picp,
                metrics.calibration.ace,
            )
        ],
    }

    metrics_path = (
        run_dir
        / "validation_probabilistic_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "PHASE 3A VALIDATION RESULTS"
    )
    print("=" * 72)

    print(
        "Alpha:",
        alpha,
    )

    print(
        "CRPS:",
        result[
            "crps_mean"
        ],
    )

    print(
        "Energy:",
        result[
            "energy_score"
        ],
    )

    print(
        "Mean |ACE|:",
        result[
            "mean_abs_ace"
        ],
    )

    print()

    print(
        f"{'level':>8}"
        f"{'PICP':>14}"
        f"{'ACE':>14}"
    )

    for item in result[
        "calibration"
    ]:
        print(
            f"{item['level']:8.2f}"
            f"{item['picp']:14.6f}"
            f"{item['ace']:14.6f}"
        )

    print()

    print(
        "Scenarios saved:",
        scenario_path,
    )

    print(
        "Metrics saved:",
        metrics_path,
    )

    print()

    print(
        "TEST SPLIT WAS NOT EVALUATED."
    )


if __name__ == "__main__":
    main()