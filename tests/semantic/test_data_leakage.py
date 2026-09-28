"""Semantic tests for temporal leakage in the data pipeline.

These tests deliberately alter future observations and verify that training
statistics and training samples are invariant.

They are intended to test scientific semantics, not merely tensor shapes.
"""

import numpy as np
import pandas as pd

from diffusion_portfolio.data import (
    INDUSTRY_COLUMNS,
    ReturnTable,
    TrainStandardizer,
    build_window_datasets,
)


LOOKBACK = 20


def make_table(
    n_days: int = 240,
) -> ReturnTable:
    dates = pd.date_range(
        "2019-01-01",
        periods=n_days,
        freq="B",
    )

    time = np.arange(
        n_days,
        dtype=np.float32,
    )[:, None]

    assets = np.arange(
        len(INDUSTRY_COLUMNS),
        dtype=np.float32,
    )[None, :]

    # Deterministic values with non-zero variance for every asset.
    returns = (
        time * 0.0001
        + assets * 0.00001
    ).astype(np.float32)

    return ReturnTable(
        dates=dates,
        returns=returns,
        columns=INDUSTRY_COLUMNS,
    )


def build_pipeline(
    raw: ReturnTable,
):
    train_end = raw.dates[119]
    val_end = raw.dates[179]
    test_end = raw.dates[-1]

    scaler = TrainStandardizer.fit(
        raw,
        train_end=train_end,
    )

    model = scaler.transform(
        raw
    )

    datasets = build_window_datasets(
        model,
        raw,
        lookback=LOOKBACK,
        horizon=1,
        train_end=str(train_end.date()),
        val_end=str(val_end.date()),
        test_end=str(test_end.date()),
    )

    return (
        scaler,
        datasets,
        train_end,
        val_end,
        test_end,
    )


def test_future_corruption_cannot_change_training_pipeline():
    """Changing every post-training value must not affect training."""

    raw = make_table()

    (
        scaler_a,
        datasets_a,
        train_end,
        _,
        _,
    ) = build_pipeline(raw)

    altered_returns = raw.returns.copy()

    future_mask = (
        raw.dates > train_end
    )

    # Deliberately absurd future values.
    altered_returns[future_mask] += 1000.0

    altered = ReturnTable(
        dates=raw.dates,
        returns=altered_returns,
        columns=raw.columns,
    )

    (
        scaler_b,
        datasets_b,
        _,
        _,
        _,
    ) = build_pipeline(altered)

    # Future data must not affect fitted statistics.
    np.testing.assert_array_equal(
        scaler_a.mean,
        scaler_b.mean,
    )

    np.testing.assert_array_equal(
        scaler_a.std,
        scaler_b.std,
    )

    # Or any standardized training histories/targets.
    np.testing.assert_array_equal(
        datasets_a.train.model_windows.history,
        datasets_b.train.model_windows.history,
    )

    np.testing.assert_array_equal(
        datasets_a.train.model_windows.target,
        datasets_b.train.model_windows.target,
    )

    # Or raw training samples.
    np.testing.assert_array_equal(
        datasets_a.train.raw_windows.history,
        datasets_b.train.raw_windows.history,
    )

    np.testing.assert_array_equal(
        datasets_a.train.raw_windows.target,
        datasets_b.train.raw_windows.target,
    )

    # Sanity check: the corruption actually reached future data.
    assert not np.array_equal(
        datasets_a.val.raw_windows.target,
        datasets_b.val.raw_windows.target,
    )


def test_target_day_cannot_enter_its_own_history():
    """Changing y_t must not change X_t for that same sample."""

    raw = make_table()

    (
        scaler,
        datasets_a,
        train_end,
        val_end,
        test_end,
    ) = build_pipeline(raw)

    first_val_date = (
        datasets_a.val.model_windows.target_dates[0]
    )

    target_idx = raw.dates.get_loc(
        first_val_date
    )

    altered_returns = raw.returns.copy()

    altered_returns[target_idx] += 10.0

    altered = ReturnTable(
        dates=raw.dates,
        returns=altered_returns,
        columns=raw.columns,
    )

    # Reuse the already-fitted training scaler.
    altered_model = scaler.transform(
        altered
    )

    datasets_b = build_window_datasets(
        altered_model,
        altered,
        lookback=LOOKBACK,
        horizon=1,
        train_end=str(train_end.date()),
        val_end=str(val_end.date()),
        test_end=str(test_end.date()),
    )

    # X_t must be unchanged.
    np.testing.assert_array_equal(
        datasets_a.val.raw_windows.history[0],
        datasets_b.val.raw_windows.history[0],
    )

    np.testing.assert_array_equal(
        datasets_a.val.model_windows.history[0],
        datasets_b.val.model_windows.history[0],
    )

    # y_t must reflect the change.
    assert not np.array_equal(
        datasets_a.val.raw_windows.target[0],
        datasets_b.val.raw_windows.target[0],
    )

    assert not np.array_equal(
        datasets_a.val.model_windows.target[0],
        datasets_b.val.model_windows.target[0],
    )


def test_first_validation_history_is_exactly_the_prior_window():
    """Validate the causal off-by-one boundary explicitly."""

    raw = make_table()

    (
        _,
        datasets,
        _,
        _,
        _,
    ) = build_pipeline(raw)

    target_date = (
        datasets.val.raw_windows.target_dates[0]
    )

    target_idx = raw.dates.get_loc(
        target_date
    )

    expected_history = raw.returns[
        target_idx - LOOKBACK : target_idx
    ]

    expected_target = raw.returns[
        target_idx : target_idx + 1
    ]

    np.testing.assert_array_equal(
        datasets.val.raw_windows.history[0],
        expected_history,
    )

    np.testing.assert_array_equal(
        datasets.val.raw_windows.target[0],
        expected_target,
    )


def test_split_target_dates_are_disjoint_and_ordered():
    """No target date may belong to more than one partition."""

    raw = make_table()

    (
        _,
        datasets,
        train_end,
        val_end,
        _,
    ) = build_pipeline(raw)

    train_dates = (
        datasets.train.model_windows.target_dates
    )

    val_dates = (
        datasets.val.model_windows.target_dates
    )

    test_dates = (
        datasets.test.model_windows.target_dates
    )

    assert train_dates[-1] <= train_end
    assert val_dates[0] > train_end

    assert val_dates[-1] <= val_end
    assert test_dates[0] > val_end

    train_set = set(train_dates)
    val_set = set(val_dates)
    test_set = set(test_dates)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)