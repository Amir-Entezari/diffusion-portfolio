"""Evaluate a trained Diffolio checkpoint on validation or test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from diffusion_portfolio.data import (
    CovariateStandardizer,
    DiffolioDataset,
    TrainStandardizer,
    align_systematic_covariates_to_dates,
    build_asset_characteristics,
    build_monthly_systematic_covariates,
    collate_diffolio_batch,
    load_daily_factors,
    load_daily_risk_free,
    load_kf12_daily,
    make_diffolio_windows,
    slice_return_table,
    split_diffolio_windows_by_date,
    to_excess_returns,
)
from diffusion_portfolio.evaluation import (
    evaluate_probabilistic_forecast,
)
from diffusion_portfolio.models.diffolio import (
    DiffolioObjective,
    sample_diffolio_ddim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/diffolio/"
            "diffolio_100k.pt"
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "val",
            "test",
        ],
        default="val",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional limit for a sampling smoke test."
        ),
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


def load_artifacts(
    checkpoint_path: Path,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    config_path = (
        checkpoint_path.parent
        / (
            checkpoint_path.stem
            + "_config.yaml"
        )
    )

    preprocessing_path = (
        checkpoint_path.parent
        / (
            checkpoint_path.stem
            + "_preprocessing.npz"
        )
    )

    for path in (
        config_path,
        preprocessing_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    preprocessing = np.load(
        preprocessing_path,
        allow_pickle=False,
    )

    return_scaler = TrainStandardizer(
        mean=preprocessing[
            "return_mean"
        ],
        std=preprocessing[
            "return_std"
        ],
        columns=tuple(
            str(value)
            for value
            in preprocessing[
                "return_columns"
            ]
        ),
    )

    covariate_scaler = (
        CovariateStandardizer(
            asset_mean=preprocessing[
                "asset_covariate_mean"
            ],
            asset_std=preprocessing[
                "asset_covariate_std"
            ],
            systematic_mean=preprocessing[
                "systematic_mean"
            ],
            systematic_std=preprocessing[
                "systematic_std"
            ],
            asset_columns=tuple(
                str(value)
                for value
                in preprocessing[
                    "asset_covariate_columns"
                ]
            ),
            systematic_columns=tuple(
                str(value)
                for value
                in preprocessing[
                    "systematic_columns"
                ]
            ),
        )
    )

    return (
        config,
        return_scaler,
        covariate_scaler,
    )


def build_evaluation_dataset(
    config: dict,
    return_scaler: TrainStandardizer,
    covariate_scaler: CovariateStandardizer,
    split_name: str,
) -> DiffolioDataset:
    data_cfg = config[
        "data"
    ]

    assets = load_kf12_daily()

    factors = load_daily_factors()

    risk_free = load_daily_risk_free()

    excess_full = to_excess_returns(
        assets,
        risk_free,
    )

    raw_returns = slice_return_table(
        excess_full,
        data_cfg[
            "sample_start"
        ],
        data_cfg[
            "sample_end"
        ],
    )

    model_returns = (
        return_scaler.transform(
            raw_returns
        )
    )

    asset_covariates = (
        build_asset_characteristics(
            excess_full,
            factors,
            start=data_cfg[
                "sample_start"
            ],
            end=data_cfg[
                "sample_end"
            ],
        )
    )

    goyal_path = Path(
        data_cfg[
            "goyal_path"
        ]
    )

    raw_goyal = pd.read_excel(
        goyal_path,
        sheet_name="Monthly",
        engine="openpyxl",
    )

    monthly_systematic = (
        build_monthly_systematic_covariates(
            raw_goyal
        )
    )

    systematic_covariates = (
        align_systematic_covariates_to_dates(
            monthly_systematic,
            raw_returns.dates,
        )
    )

    asset_scaled = (
        covariate_scaler.transform_asset(
            asset_covariates
        )
    )

    systematic_scaled = (
        covariate_scaler
        .transform_systematic(
            systematic_covariates
        )
    )

    windows = make_diffolio_windows(
        model_returns,
        raw_returns,
        asset_scaled,
        systematic_scaled,
        lookback=int(
            data_cfg[
                "lookback"
            ]
        ),
    )

    splits = (
        split_diffolio_windows_by_date(
            windows,
            train_end=data_cfg[
                "train_end"
            ],
            val_end=data_cfg[
                "val_end"
            ],
            test_end=data_cfg[
                "sample_end"
            ],
        )
    )

    selected = (
        splits.val
        if split_name == "val"
        else splits.test
    )

    return DiffolioDataset(
        selected
    )


def build_model(
    config: dict,
    checkpoint: dict,
    device: torch.device,
) -> DiffolioObjective:
    model_cfg = config[
        "model"
    ]

    data_cfg = config[
        "data"
    ]

    state_dict = checkpoint[
        "model_state_dict"
    ]

    training_covariance = (
        state_dict[
            "training_covariance"
        ]
        .detach()
        .clone()
    )

    model = DiffolioObjective(
        training_covariance=(
            training_covariance
        ),
        n_assets=int(
            model_cfg[
                "n_assets"
            ]
        ),
        n_asset_characteristics=int(
            model_cfg[
                "n_asset_characteristics"
            ]
        ),
        n_systematic=int(
            model_cfg[
                "n_systematic"
            ]
        ),
        lookback=int(
            data_cfg[
                "lookback"
            ]
        ),
        hidden_dim=int(
            model_cfg[
                "hidden_dim"
            ]
        ),
        num_heads=int(
            model_cfg[
                "num_heads"
            ]
        ),
        mlp_dim=int(
            model_cfg[
                "mlp_dim"
            ]
        ),
        time_embedding_dim=int(
            model_cfg[
                "time_embedding_dim"
            ]
        ),
        diffusion_steps=int(
            model_cfg[
                "diffusion_steps"
            ]
        ),
        beta_start=float(
            model_cfg[
                "beta_start"
            ]
        ),
        beta_end=float(
            model_cfg[
                "beta_end"
            ]
        ),
        lambda_corr=float(
            model_cfg[
                "lambda_corr"
            ]
        ),
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    return model


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive"
        )

    device = resolve_device(
        args.device
    )

    (
        config,
        return_scaler,
        covariate_scaler,
    ) = load_artifacts(
        args.checkpoint
    )

    inference_cfg = config[
        "inference"
    ]

    if (
        inference_cfg[
            "sampler"
        ]
        != "ddim"
    ):
        raise ValueError(
            "Expected DDIM inference"
        )

    if float(
        inference_cfg[
            "eta"
        ]
    ) != 0.0:
        raise ValueError(
            "Diffolio reproduction "
            "requires eta=0"
        )

    dataset = build_evaluation_dataset(
        config,
        return_scaler,
        covariate_scaler,
        args.split,
    )

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(
                "max-samples must be positive"
            )

        n_selected = min(
            args.max_samples,
            len(
                dataset
            ),
        )

        evaluation_dataset = Subset(
            dataset,
            range(
                n_selected
            ),
        )

    else:
        evaluation_dataset = dataset

    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_diffolio_batch,
        drop_last=False,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model(
        config,
        checkpoint,
        device,
    )

    seed = int(
        config[
            "seed"
        ]
    )

    torch.manual_seed(
        seed
    )

    np.random.seed(
        seed
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            seed
        )

    n_scenarios = int(
        inference_cfg[
            "n_scenarios"
        ]
    )

    sampling_steps = int(
        inference_cfg[
            "sampling_steps"
        ]
    )

    n_samples = len(
        evaluation_dataset
    )

    n_assets = len(
        return_scaler.columns
    )

    print(
        "=" * 72
    )
    print(
        "DIFFOLIO PROBABILISTIC EVALUATION"
    )
    print(
        "=" * 72
    )

    print(
        "Split:",
        args.split,
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
        "Checkpoint step:",
        checkpoint[
            "step"
        ],
    )

    print(
        "Evaluation samples:",
        n_samples,
    )

    print(
        "Scenarios per date:",
        n_scenarios,
    )

    print(
        "DDIM steps:",
        sampling_steps,
    )

    print(
        "Inference batch size:",
        args.batch_size,
    )

    forecast_raw = np.empty(
        (
            n_samples,
            n_scenarios,
            n_assets,
        ),
        dtype=np.float32,
    )

    observed_raw = np.empty(
        (
            n_samples,
            n_assets,
        ),
        dtype=np.float32,
    )

    dates = []

    offset = 0

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="DDIM sampling",
        ):
            return_history = batch[
                "return_history"
            ].to(
                device,
                non_blocking=True,
            )

            asset_covariates = batch[
                "asset_covariates"
            ].to(
                device,
                non_blocking=True,
            )

            systematic_covariates = batch[
                "systematic_covariates"
            ].to(
                device,
                non_blocking=True,
            )

            generated = sample_diffolio_ddim(
                model,
                return_history,
                asset_covariates,
                systematic_covariates,
                n_scenarios=n_scenarios,
                sampling_steps=(
                    sampling_steps
                ),
            )

            generated_model = (
                generated
                .cpu()
                .numpy()
            )

            generated_raw = (
                return_scaler
                .inverse_transform(
                    generated_model
                )
            )

            batch_size = (
                generated_raw.shape[
                    0
                ]
            )

            end = (
                offset
                + batch_size
            )

            forecast_raw[
                offset:end
            ] = generated_raw

            observed_raw[
                offset:end
            ] = (
                batch[
                    "target_raw"
                ]
                .numpy()
            )

            dates.extend(
                batch[
                    "target_date"
                ].tolist()
            )

            offset = end

    if offset != n_samples:
        raise RuntimeError(
            "Evaluation sample count mismatch"
        )

    if not np.isfinite(
        forecast_raw
    ).all():
        raise RuntimeError(
            "Generated scenarios contain "
            "NaN or infinite values"
        )

    dates = pd.DatetimeIndex(
        dates
    )

    print()
    print(
        "Scenario cube:",
        forecast_raw.shape,
    )

    print(
        "Date range:",
        dates[
            0
        ],
        "to",
        dates[
            -1
        ],
    )

    print()
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

    print(
        "Observed raw std:",
        float(
            observed_raw.std()
        ),
    )

    metrics = (
        evaluate_probabilistic_forecast(
            forecast_raw,
            observed_raw,
        )
    )

    suffix = args.split

    if args.max_samples is not None:
        suffix += (
            f"_smoke_{n_samples}"
        )

    output_dir = (
        args.checkpoint.parent
    )

    scenario_path = (
        output_dir
        / f"{suffix}_scenarios.npz"
    )

    metrics_path = (
        output_dir
        / (
            f"{suffix}_"
            "probabilistic_metrics.json"
        )
    )

    np.savez(
        scenario_path,
        scenarios=forecast_raw,
        observed=observed_raw,
        dates=dates.values.astype(
            "datetime64[D]"
        ),
        columns=np.asarray(
            return_scaler.columns,
            dtype=str,
        ),
    )

    result = {
        "split": args.split,
        "checkpoint_step": int(
            checkpoint[
                "step"
            ]
        ),
        "n_dates": n_samples,
        "n_scenarios": n_scenarios,
        "sampling_steps": sampling_steps,
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
        "observed_std": float(
            observed_raw.std()
        ),
        "crps_by_asset": {
            column: float(
                value
            )
            for column, value
            in zip(
                return_scaler.columns,
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
    print(
        "=" * 72
    )
    print(
        "PROBABILISTIC METRICS"
    )
    print(
        "=" * 72
    )

    print(
        "CRPS mean:",
        metrics.crps_mean,
    )

    print(
        "CRPS std:",
        metrics.crps_std,
    )

    print(
        "Energy Score:",
        metrics.energy_score,
    )

    print()
    print(
        f"{'level':>8}"
        f"{'PICP':>12}"
        f"{'ACE':>12}"
    )

    for (
        level,
        picp,
        ace,
    ) in zip(
        metrics.calibration.levels,
        metrics.calibration.picp,
        metrics.calibration.ace,
    ):
        print(
            f"{level:8.2f}"
            f"{picp:12.6f}"
            f"{ace:12.6f}"
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


if __name__ == "__main__":
    main()