import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data import (
    INDUSTRY_COLUMNS,
    ReturnTable,
    RiskFreeSeries,
    to_excess_returns,
)


def make_return_table():
    dates = pd.date_range(
        "2020-01-01",
        periods=3,
        freq="B",
    )

    returns = np.array(
        [
            [0.010] * 12,
            [0.020] * 12,
            [-0.010] * 12,
        ],
        dtype=np.float32,
    )

    return ReturnTable(
        dates=dates,
        returns=returns,
        columns=INDUSTRY_COLUMNS,
    )


def test_excess_returns_are_asset_minus_rf():
    table = make_return_table()

    rf = RiskFreeSeries(
        dates=table.dates,
        returns=np.array(
            [
                0.001,
                0.002,
                0.003,
            ],
            dtype=np.float32,
        ),
    )

    excess = to_excess_returns(
        table,
        rf,
    )

    assert excess.returns[0, 0] == pytest.approx(
        0.009
    )

    assert excess.returns[1, 0] == pytest.approx(
        0.018
    )

    assert excess.returns[2, 0] == pytest.approx(
        -0.013
    )


def test_excess_returns_preserve_dates_and_columns():
    table = make_return_table()

    rf = RiskFreeSeries(
        dates=table.dates,
        returns=np.zeros(
            3,
            dtype=np.float32,
        ),
    )

    excess = to_excess_returns(
        table,
        rf,
    )

    assert excess.dates.equals(
        table.dates
    )

    assert excess.columns == table.columns


    
def test_extra_rf_dates_do_not_change_alignment():
    table = make_return_table()

    rf_dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2019-12-31"),
            *table.dates,
        ]
    )

    rf = RiskFreeSeries(
        dates=rf_dates,
        returns=np.array(
            [
                0.5,
                0.001,
                0.002,
                0.003,
            ],
            dtype=np.float32,
        ),
    )

    excess = to_excess_returns(
        table,
        rf,
    )

    assert excess.returns[0, 0] == pytest.approx(
        0.009
    )


def test_missing_rf_date_is_rejected():
    table = make_return_table()

    rf = RiskFreeSeries(
        dates=table.dates[:2],
        returns=np.array(
            [
                0.001,
                0.002,
            ],
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing dates",
    ):
        to_excess_returns(
            table,
            rf,
        )