"""End-to-end smoke test for the real KF12 MVP data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/mvp.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(
        Path(args.config)
    )

    # ---------------------------------------------------------
    # 1. Official KF12 industry returns
    # ---------------------------------------------------------
    assets = load_kf12_daily()

    assets = slice_return_table(
        assets,
        start=cfg.data.sample_start,
        end=cfg.data.sample_end,
    )

    # ---------------------------------------------------------
    # 2. Official daily risk-free return
    # ---------------------------------------------------------
    risk_free = load_daily_risk_free()

    # ---------------------------------------------------------
    # 3. Daily excess returns
    # ---------------------------------------------------------
    excess = to_excess_returns(
        assets,
        risk_free,
    )

    assert excess.returns.shape == assets.returns.shape
    assert excess.dates.equals(assets.dates)
    assert np.isfinite(excess.returns).all()

    # ---------------------------------------------------------
    # 4. Model-space representation
    # ---------------------------------------------------------
    scaler = None

    if cfg.data.return_scaling == "train_zscore":
        scaler = TrainStandardizer.fit(
            excess,
            train_end=cfg.data.train_end,
        )

        model_table = scaler.transform(
            excess
        )

    elif cfg.data.return_scaling == "none":
        model_table = excess

    else:
        raise RuntimeError(
            f"Unsupported return scaling: "
            f"{cfg.data.return_scaling}"
        )

    # ---------------------------------------------------------
    # 5. Windows + temporal splits
    # ---------------------------------------------------------
    datasets = build_window_datasets(
        model_table,
        excess,
        lookback=cfg.data.lookback,
        horizon=cfg.data.horizon,
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
        test_end=cfg.data.sample_end,
    )

    assert len(datasets.train) > 0
    assert len(datasets.val) > 0
    assert len(datasets.test) > 0

    # Temporal boundary checks.
    train_last = (
        datasets.train.model_windows.target_dates[-1]
    )
    val_first = (
        datasets.val.model_windows.target_dates[0]
    )
    val_last = (
        datasets.val.model_windows.target_dates[-1]
    )
    test_first = (
        datasets.test.model_windows.target_dates[0]
    )

    assert train_last <= pd.Timestamp(
        cfg.data.train_end
    )

    assert val_first > pd.Timestamp(
        cfg.data.train_end
    )

    assert val_last <= pd.Timestamp(
        cfg.data.val_end
    )

    assert test_first > pd.Timestamp(
        cfg.data.val_end
    )

    # ---------------------------------------------------------
    # 6. PyTorch loaders
    # ---------------------------------------------------------
    loaders = make_dataloaders(
        datasets,
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
    )

    batch = next(
        iter(loaders.val)
    )

    expected_history_shape = (
        batch["history"].shape[0],
        cfg.data.lookback,
        len(excess.columns),
    )

    expected_target_shape = (
        batch["target"].shape[0],
        cfg.data.horizon,
        len(excess.columns),
    )

    assert tuple(
        batch["history"].shape
    ) == expected_history_shape

    assert tuple(
        batch["target"].shape
    ) == expected_target_shape

    assert tuple(
        batch["history_raw"].shape
    ) == expected_history_shape

    assert tuple(
        batch["target_raw"].shape
    ) == expected_target_shape

    assert batch[
        "target_date"
    ].is_monotonic_increasing

    # ---------------------------------------------------------
    # 7. Model-space ↔ raw-space consistency
    # ---------------------------------------------------------
    if scaler is not None:
        recovered = scaler.inverse_transform(
            batch["target"].numpy()
        )

        inverse_error = float(
            np.max(
                np.abs(
                    recovered
                    - batch["target_raw"].numpy()
                )
            )
        )

        assert inverse_error < 1e-6

    else:
        inverse_error = 0.0

        np.testing.assert_allclose(
            batch["target"].numpy(),
            batch["target_raw"].numpy(),
        )

    # ---------------------------------------------------------
    # Reproducibility report
    # ---------------------------------------------------------
    print("Real-data smoke test passed")
    print()

    print("Config")
    print("------")
    print(f"sample: {cfg.data.sample_start} -> {cfg.data.sample_end}")
    print(f"train_end: {cfg.data.train_end}")
    print(f"val_end: {cfg.data.val_end}")
    print(f"lookback: {cfg.data.lookback}")
    print(f"horizon: {cfg.data.horizon}")
    print(f"scaling: {cfg.data.return_scaling}")

    print()
    print("Data")
    print("----")
    print(f"observations: {len(excess.dates)}")
    print(f"assets: {len(excess.columns)}")
    print(
        f"dates: {excess.dates[0].date()} "
        f"-> {excess.dates[-1].date()}"
    )

    print()
    print("Datasets")
    print("--------")
    print(f"train: {len(datasets.train)}")
    print(f"val:   {len(datasets.val)}")
    print(f"test:  {len(datasets.test)}")

    print()
    print("Validation batch")
    print("----------------")
    print(
        "history:",
        tuple(batch["history"].shape),
    )
    print(
        "target:",
        tuple(batch["target"].shape),
    )
    print(
        "history_raw:",
        tuple(batch["history_raw"].shape),
    )
    print(
        "target_raw:",
        tuple(batch["target_raw"].shape),
    )
    print(
        "first target date:",
        batch["target_date"][0],
    )
    print(
        "inverse/raw max error:",
        inverse_error,
    )


if __name__ == "__main__":
    main()