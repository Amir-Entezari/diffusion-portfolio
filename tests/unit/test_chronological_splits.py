import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data import (
    INDUSTRY_COLUMNS,
    dataframe_to_return_table,
    make_windows,
    split_windows_chronologically,
)


def make_sequential_table(n_days: int = 160):
    dates = pd.date_range(
        "2020-01-01",
        periods=n_days,
        freq="B",
    )

    # Each day has a unique identifiable value.
    daily_value = np.arange(
        n_days,
        dtype=np.float64,
    )[:, None]

    values = np.repeat(
        daily_value,
        len(INDUSTRY_COLUMNS),
        axis=1,
    )

    # dataframe_to_return_table expects percentage returns.
    frame = pd.DataFrame(
        values,
        index=dates,
        columns=INDUSTRY_COLUMNS,
    )

    return dataframe_to_return_table(frame)


def test_split_sizes_for_mvp_ratios():
    table = make_sequential_table(160)

    windows = make_windows(
        table,
        lookback=60,
        horizon=1,
    )

    # 160 - 60 - 1 + 1 = 100 samples.
    assert windows.history.shape[0] == 100

    splits = split_windows_chronologically(
        windows,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    assert splits.train.history.shape[0] == 70
    assert splits.val.history.shape[0] == 15
    assert splits.test.history.shape[0] == 15


def test_split_order_is_strictly_chronological():
    table = make_sequential_table(160)

    windows = make_windows(
        table,
        lookback=60,
        horizon=1,
    )

    splits = split_windows_chronologically(
        windows,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    assert (
        splits.train.target_dates[-1]
        < splits.val.target_dates[0]
    )

    assert (
        splits.val.target_dates[-1]
        < splits.test.target_dates[0]
    )


def test_samples_are_not_shuffled():
    table = make_sequential_table(160)

    windows = make_windows(
        table,
        lookback=60,
        horizon=1,
    )

    splits = split_windows_chronologically(
        windows,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    np.testing.assert_allclose(
        splits.train.history,
        windows.history[:70],
    )

    np.testing.assert_allclose(
        splits.val.history,
        windows.history[70:85],
    )

    np.testing.assert_allclose(
        splits.test.history,
        windows.history[85:],
    )


def test_validation_can_use_past_training_context():
    table = make_sequential_table(160)

    windows = make_windows(
        table,
        lookback=60,
        horizon=1,
    )

    splits = split_windows_chronologically(
        windows,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    # Validation target begins immediately after the final
    # training target for horizon=1.
    assert (
        splits.val.target_dates[0]
        > splits.train.target_dates[-1]
    )

    # But its HISTORY legitimately contains observations
    # from the preceding training-era dates.
    #
    # First validation sample corresponds exactly to
    # original chronological window 70.
    np.testing.assert_allclose(
        splits.val.history[0],
        windows.history[70],
    )


def test_multistep_horizon_is_purged_at_boundaries():
    table = make_sequential_table(170)

    windows = make_windows(
        table,
        lookback=60,
        horizon=5,
    )

    splits = split_windows_chronologically(
        windows,
        train_ratio=0.70,
        val_ratio=0.15,
        purge_target_overlap=True,
    )

    # horizon=5 -> four samples are purged at each boundary.
    #
    # Therefore the first validation target start must be
    # five sample positions after the final training target start.
    train_last_date = splits.train.target_dates[-1]
    val_first_date = splits.val.target_dates[0]

    train_last_idx = windows.target_dates.get_loc(
        train_last_date
    )

    val_first_idx = windows.target_dates.get_loc(
        val_first_date
    )

    assert val_first_idx - train_last_idx == 5


def test_invalid_ratios_are_rejected():
    table = make_sequential_table(160)

    windows = make_windows(
        table,
        lookback=60,
        horizon=1,
    )

    with pytest.raises(ValueError):
        split_windows_chronologically(
            windows,
            train_ratio=0.8,
            val_ratio=0.3,
        )