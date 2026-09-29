import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data.systematic_covariates import (
    SYSTEMATIC_COLUMNS,
    SystematicCovariateTable,
    align_systematic_covariates_to_dates,
    build_monthly_systematic_covariates,
)


def make_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "yyyymm": [
                199912,
                200001,
                200002,
            ],
            "Index": [
                100.0,
                110.0,
                120.0,
            ],
            "D12": [
                4.0,
                4.4,
                4.8,
            ],
            "E12": [
                5.0,
                5.5,
                6.0,
            ],
            "b/m": [
                0.30,
                0.31,
                0.32,
            ],
            "tbl": [
                0.05,
                0.051,
                0.052,
            ],
            "AAA": [
                0.07,
                0.071,
                0.072,
            ],
            "BAA": [
                0.08,
                0.083,
                0.086,
            ],
            "lty": [
                0.06,
                0.063,
                0.066,
            ],
            "ntis": [
                0.01,
                0.02,
                0.03,
            ],
            "svar": [
                0.002,
                0.003,
                0.004,
            ],
        }
    )


def test_systematic_definitions():
    raw = make_raw_data()

    result = (
        build_monthly_systematic_covariates(
            raw
        )
    )

    assert result.columns == (
        SYSTEMATIC_COLUMNS
    )

    assert result.values.shape == (
        3,
        8,
    )

    row = result.values[
        1
    ]

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "tbl"
        )
    ] == pytest.approx(
        0.051
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "dp"
        )
    ] == pytest.approx(
        np.log(
            4.4
        )
        - np.log(
            110.0
        )
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "ep"
        )
    ] == pytest.approx(
        np.log(
            5.5
        )
        - np.log(
            110.0
        )
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "bm"
        )
    ] == pytest.approx(
        0.31
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "tms"
        )
    ] == pytest.approx(
        0.063
        - 0.051
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "dfy"
        )
    ] == pytest.approx(
        0.083
        - 0.071
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "ntis"
        )
    ] == pytest.approx(
        0.02
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "svar"
        )
    ] == pytest.approx(
        0.003
    )


def test_monthly_dates_are_month_end():
    result = (
        build_monthly_systematic_covariates(
            make_raw_data()
        )
    )

    assert result.dates[
        0
    ] == pd.Timestamp(
        "1999-12-31"
    )

    assert result.dates[
        1
    ] == pd.Timestamp(
        "2000-01-31"
    )

    assert result.dates[
        2
    ] == pd.Timestamp(
        "2000-02-29"
    )


def test_daily_alignment_uses_latest_available_month():
    monthly = (
        build_monthly_systematic_covariates(
            make_raw_data()
        )
    )

    dates = pd.DatetimeIndex(
        [
            "2000-01-03",
            "2000-01-31",
            "2000-02-01",
            "2000-02-28",
            "2000-03-01",
        ]
    )

    daily = (
        align_systematic_covariates_to_dates(
            monthly,
            dates,
        )
    )

    # Jan 3 cannot see the January month-end value yet.
    np.testing.assert_allclose(
        daily.values[
            0
        ],
        monthly.values[
            0
        ],
    )

    # On Jan 31, January's observation becomes available.
    np.testing.assert_allclose(
        daily.values[
            1
        ],
        monthly.values[
            1
        ],
    )

    # Feb 1 still uses January.
    np.testing.assert_allclose(
        daily.values[
            2
        ],
        monthly.values[
            1
        ],
    )

    # Feb 28 is before Feb 29 month-end.
    np.testing.assert_allclose(
        daily.values[
            3
        ],
        monthly.values[
            1
        ],
    )

    # March 1 can use February's observation.
    np.testing.assert_allclose(
        daily.values[
            4
        ],
        monthly.values[
            2
        ],
    )


def test_future_month_cannot_change_past_daily_values():
    monthly = (
        build_monthly_systematic_covariates(
            make_raw_data()
        )
    )

    dates = pd.DatetimeIndex(
        [
            "2000-01-03",
            "2000-01-20",
        ]
    )

    original = (
        align_systematic_covariates_to_dates(
            monthly,
            dates,
        )
    )

    corrupted_values = (
        monthly.values.copy()
    )

    corrupted_values[
        1:
    ] += 1000.0

    corrupted = SystematicCovariateTable(
        dates=monthly.dates,
        values=corrupted_values,
        columns=monthly.columns,
    )

    changed = (
        align_systematic_covariates_to_dates(
            corrupted,
            dates,
        )
    )

    np.testing.assert_allclose(
        original.values,
        changed.values,
        rtol=0.0,
        atol=0.0,
    )
    
    
    
    
def test_current_goyal_schema_is_supported():
    raw = pd.DataFrame(
        {
            "yyyymm": [
                200001,
                200002,
            ],
            "price": [
                100.0,
                101.0,
            ],
            "d12": [
                4.0,
                4.1,
            ],
            "e12": [
                5.0,
                5.1,
            ],
            "b/m": [
                0.30,
                0.31,
            ],
            "tbl": [
                0.05,
                0.051,
            ],
            "d/p": [
                -3.21,
                -3.20,
            ],
            "e/p": [
                -2.99,
                -2.98,
            ],
            "tms": [
                0.012,
                0.013,
            ],
            "dfy": [
                0.009,
                0.010,
            ],
            "ntis": [
                0.01,
                0.02,
            ],
            "svar": [
                0.002,
                0.003,
            ],
        }
    )

    result = (
        build_monthly_systematic_covariates(
            raw
        )
    )

    assert result.values.shape == (
        2,
        8,
    )

    row = result.values[
        0
    ]

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "dp"
        )
    ] == pytest.approx(
        -3.21
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "ep"
        )
    ] == pytest.approx(
        -2.99
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "tms"
        )
    ] == pytest.approx(
        0.012
    )

    assert row[
        SYSTEMATIC_COLUMNS.index(
            "dfy"
        )
    ] == pytest.approx(
        0.009
    )