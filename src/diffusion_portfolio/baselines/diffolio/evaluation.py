"""Evaluate a trained Diffolio checkpoint on validation or test data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from diffusion_portfolio.data import CovariateStandardizer, TrainStandardizer
from diffusion_portfolio.baselines.diffolio.loaders import DiffolioDataset, collate_diffolio_batch
from diffusion_portfolio.evaluation import (
    evaluate_probabilistic_forecast,
)
from diffusion_portfolio.baselines.diffolio import sample_diffolio_ddim


from diffusion_portfolio.utils.device import resolve_device
from diffusion_portfolio.evaluation.scenarios import probabilistic_summary
from diffusion_portfolio.baselines.diffolio.build import restore_model
from diffusion_portfolio.baselines.diffolio.preparation import prepare_diffolio_data

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


def build_evaluation_dataset(config, return_scaler, covariate_scaler, split_name):
    splits, _, _, _ = prepare_diffolio_data(
        config["data"], return_scaler=return_scaler, covariate_scaler=covariate_scaler
    )
    return DiffolioDataset(splits.val if split_name == "val" else splits.test)


def run_evaluation(args) -> None:

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

    model = restore_model(
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
        'split': args.split,
        'checkpoint_step': int(checkpoint['step']),
        'n_dates': n_samples,
        'n_scenarios': n_scenarios,
        'sampling_steps': sampling_steps,
        'scenario_mean': float(forecast_raw.mean()),
        'scenario_std': float(forecast_raw.std()),
        'scenario_max_abs': float(np.abs(forecast_raw).max()),
        'observed_std': float(observed_raw.std()),
        **probabilistic_summary(metrics, return_scaler.columns),
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
