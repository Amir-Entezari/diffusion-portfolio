"""Generate and evaluate conditional diffusion forecasts on the test period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import (
    TrainStandardizer,
    build_window_datasets,
    collate_return_batch,
    load_daily_risk_free,
    load_kf12_daily,
    slice_return_table,
    to_excess_returns,
)
from diffusion_portfolio.evaluation import (
    evaluate_probabilistic_forecast,
)
from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing best.pt, config.yaml, standardizer.npz",
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

    for path in (
        config_path,
        checkpoint_path,
        standardizer_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    cfg = load_config(
        config_path
    )

    device = resolve_device(
        args.device
    )

    print("=" * 72)
    print("DIFFUSION TEST FORECAST EVALUATION")
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
        "Scenarios per date:",
        cfg.evaluation.n_scenarios,
    )

    print(
        "Inference batch size:",
        args.batch_size,
    )

    # ---------------------------------------------------------
    # Reproducible sampling
    # ---------------------------------------------------------
    torch.manual_seed(
        cfg.seed
    )

    np.random.seed(
        cfg.seed
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            cfg.seed
        )

    # ---------------------------------------------------------
    # Reconstruct exact standardizer from training artifacts.
    #
    # Do NOT refit it here.
    # ---------------------------------------------------------
    scaler_file = np.load(
        standardizer_path,
        allow_pickle=False,
    )

    scaler = TrainStandardizer(
        mean=scaler_file["mean"],
        std=scaler_file["std"],
        columns=tuple(
            str(column)
            for column
            in scaler_file["columns"]
        ),
    )

    # ---------------------------------------------------------
    # Reconstruct benchmark data/windows
    # ---------------------------------------------------------
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

    test_loader = DataLoader(
        datasets.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_return_batch,
        drop_last=False,
    )

    n_test = len(
        datasets.test
    )

    n_scenarios = (
        cfg.evaluation.n_scenarios
    )

    n_assets = len(
        scaler.columns
    )

    print()
    print(
        "Test samples:",
        n_test,
    )

    print(
        "Test start:",
        datasets.test.raw_windows.target_dates[0],
    )

    print(
        "Test end:",
        datasets.test.raw_windows.target_dates[-1],
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = ConditionalDiffusionModel(
        lookback=cfg.data.lookback,
        n_assets=n_assets,
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
    ).to(device)

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
        checkpoint["epoch"],
    )

    print(
        "Checkpoint validation loss:",
        checkpoint["val_loss"],
    )

    # ---------------------------------------------------------
    # Preallocate final raw-return scenario cube:
    #
    # [test dates, scenarios, assets]
    # ---------------------------------------------------------
    forecast_raw = np.empty(
        (
            n_test,
            n_scenarios,
            n_assets,
        ),
        dtype=np.float32,
    )

    observed_raw = (
        datasets.test
        .raw_windows
        .target[:, 0, :]
        .astype(
            np.float32,
            copy=True,
        )
    )

    dates = (
        datasets.test
        .raw_windows
        .target_dates
    )

    # ---------------------------------------------------------
    # Batched reverse diffusion
    # ---------------------------------------------------------
    offset = 0

    print()
    print(
        "Generating test scenarios..."
    )

    with torch.inference_mode():
        for batch in tqdm(
            test_loader,
            desc="Sampling",
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
                n_scenarios=(
                    n_scenarios
                ),
            )

            generated_standardized = (
                generated
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

    if offset != n_test:
        raise RuntimeError(
            "Did not generate all test samples"
        )

    if not np.isfinite(
        forecast_raw
    ).all():
        raise RuntimeError(
            "Generated scenarios contain non-finite values"
        )

    # ---------------------------------------------------------
    # Save scenarios BEFORE computing metrics.
    #
    # These become the frozen vanilla test forecasts reused by
    # the later portfolio analysis.
    # ---------------------------------------------------------
    scenario_path = (
        run_dir
        / "test_scenarios.npz"
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
        "Saved scenarios:",
        scenario_path,
    )

    print(
        "Scenario cube:",
        forecast_raw.shape,
    )

    size_mb = (
        forecast_raw.nbytes
        / 1024**2
    )

    print(
        "Scenario cube size (MiB):",
        size_mb,
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

    # ---------------------------------------------------------
    # Probabilistic forecast metrics
    # ---------------------------------------------------------
    print()
    print(
        "Computing probabilistic metrics..."
    )

    metrics = (
        evaluate_probabilistic_forecast(
            forecast_raw,
            observed_raw,
        )
    )

    result = {
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "checkpoint_val_loss": float(
            checkpoint["val_loss"]
        ),
        "n_test": n_test,
        "n_scenarios": n_scenarios,
        "sampling_batch_size": (
            args.batch_size
        ),
        "test_start": str(
            dates[0].date()
        ),
        "test_end": str(
            dates[-1].date()
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
            column: float(value)
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
        "calibration": [
            {
                "level": float(level),
                "picp": float(picp),
                "ace": float(ace),
            }
            for level, picp, ace
            in zip(
                metrics.calibration.levels,
                metrics.calibration.picp,
                metrics.calibration.ace,
            )
        ],
    }

    metrics_path = (
        run_dir
        / "test_probabilistic_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print("=" * 72)
    print("TEST PROBABILISTIC EVALUATION")
    print("=" * 72)

    print(
        "CRPS mean:",
        metrics.crps_mean,
    )

    print(
        "CRPS std across assets:",
        metrics.crps_std,
    )

    print(
        "Energy Score:",
        metrics.energy_score,
    )

    print()
    print(
        "Calibration:"
    )

    print(
        f"{'level':>8} "
        f"{'PICP':>12} "
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
            f"{level:8.2f} "
            f"{picp:12.6f} "
            f"{ace:12.6f}"
        )

    print()
    print(
        "Metrics saved:",
        metrics_path,
    )


if __name__ == "__main__":
    main()