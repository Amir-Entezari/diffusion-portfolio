import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data import (
    INDUSTRY_COLUMNS,
    dataframe_to_return_table,
    make_windows,
)


def make_fake_frame(n_days: int = 100) -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-01",
        periods=n_days,
        freq="B",
    )

    values = np.arange(
        n_days * len(INDUSTRY_COLUMNS),
        dtype=np.float64,
    ).reshape(n_days, len(INDUSTRY_COLUMNS))

    # Keep values in a return-like percentage range.
    values = values / 1000.0

    return pd.DataFrame(
        values,
        index=dates,
        columns=INDUSTRY_COLUMNS,
    )


def test_percentage_returns_are_converted_to_decimals():
    frame = make_fake_frame(10)

    frame.iloc[0, 0] = 1.25

    table = dataframe_to_return_table(frame)

    assert table.returns[0, 0] == pytest.approx(0.0125)


def test_missing_value_sentinel_is_rejected():
    frame = make_fake_frame(10)

    frame.iloc[3, 2] = -99.99

    with pytest.raises(ValueError, match="missing-value sentinel"):
        dataframe_to_return_table(frame)


def test_industry_order_is_canonical():
    frame = make_fake_frame(10)

    frame = frame.loc[:, list(reversed(INDUSTRY_COLUMNS))]

    table = dataframe_to_return_table(frame)

    assert table.columns == INDUSTRY_COLUMNS


def test_window_shapes():
    frame = make_fake_frame(100)
    table = dataframe_to_return_table(frame)

    windows = make_windows(
        table,
        lookback=60,
        horizon=1,
    )

    assert windows.history.shape == (40, 60, 12)
    assert windows.target.shape == (40, 1, 12)
    assert len(windows.target_dates) == 40


def test_target_is_strictly_after_history():
    frame = make_fake_frame(70)
    table = dataframe_to_return_table(frame)

    windows = make_windows(
        table,
        lookback=60,
        horizon=1,
    )

    # First sample:
    # history = rows 0..59
    # target = row 60
    np.testing.assert_allclose(
        windows.history[0, -1],
        table.returns[59],
    )

    np.testing.assert_allclose(
        windows.target[0, 0],
        table.returns[60],
    )

    assert windows.target_dates[0] == table.dates[60]