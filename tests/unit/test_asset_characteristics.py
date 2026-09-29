import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data import (
    FactorTable,
    ReturnTable,
)
from diffusion_portfolio.data.asset_characteristics import (
    CHARACTERISTIC_COLUMNS,
    build_asset_characteristics,
)


def make_dates(
    n: int = 900,
) -> pd.DatetimeIndex:
    return pd.bdate_range(
        "2000-01-03",
        periods=n,
    )


def make_factors(
    dates: pd.DatetimeIndex,
) -> FactorTable:
    n = len(
        dates
    )

    x = np.linspace(
        0.0,
        30.0,
        n,
    )

    market = (
        0.01
        * np.sin(x)
    )

    smb = (
        0.005
        * np.cos(
            0.71 * x
        )
    )

    hml = (
        0.004
        * np.sin(
            1.37 * x
            + 0.4
        )
    )

    rf = np.full(
        n,
        0.0001,
    )

    return FactorTable(
        dates=dates,
        returns=np.column_stack(
            (
                market,
                smb,
                hml,
                rf,
            )
        ).astype(
            np.float32
        ),
        columns=(
            "Mkt-RF",
            "SMB",
            "HML",
            "RF",
        ),
    )


def make_return_table(
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
) -> ReturnTable:
    returns = np.asarray(
        returns,
        dtype=np.float32,
    )

    if returns.ndim == 1:
        returns = returns[
            :,
            None,
        ]

    columns = tuple(
        f"Asset{index}"
        for index
        in range(
            returns.shape[1]
        )
    )

    return ReturnTable(
        dates=dates,
        returns=returns,
        columns=columns,
    )


def test_momentum_and_chmom_match_direct_formula():
    dates = make_dates()

    n = len(
        dates
    )

    # Slowly changing returns so the two 6-month periods
    # are not identical.
    returns = (
        0.0002
        + np.arange(
            n,
            dtype=np.float64,
        )
        * 1e-7
    )

    table = make_return_table(
        dates,
        returns,
    )

    factors = make_factors(
        dates
    )

    t = 800

    result = build_asset_characteristics(
        table,
        factors,
        start=dates[t],
        end=dates[t],
    )

    row = result.values[
        0,
        0,
    ]

    def compounded(
        start: int,
        stop: int,
    ) -> float:
        return float(
            np.prod(
                1.0
                + returns[
                    start:stop
                ]
            )
            - 1.0
        )

    expected_mom1m = compounded(
        t - 20,
        t + 1,
    )

    expected_mom6m = compounded(
        t - 125,
        t + 1,
    )

    expected_mom12m = compounded(
        t - 251,
        t + 1,
    )

    expected_mom36m = compounded(
        t - 755,
        t + 1,
    )

    previous_mom6m = compounded(
        t - 251,
        t - 125,
    )

    expected_chmom = (
        expected_mom6m
        - previous_mom6m
    )

    assert row[
        CHARACTERISTIC_COLUMNS.index(
            "mom1m"
        )
    ] == pytest.approx(
        expected_mom1m,
        rel=1e-5,
    )

    assert row[
        CHARACTERISTIC_COLUMNS.index(
            "mom6m"
        )
    ] == pytest.approx(
        expected_mom6m,
        rel=1e-5,
    )

    assert row[
        CHARACTERISTIC_COLUMNS.index(
            "mom12m"
        )
    ] == pytest.approx(
        expected_mom12m,
        rel=1e-5,
    )

    assert row[
        CHARACTERISTIC_COLUMNS.index(
            "mom36m"
        )
    ] == pytest.approx(
        expected_mom36m,
        rel=1e-5,
    )

    assert row[
        CHARACTERISTIC_COLUMNS.index(
            "chmom"
        )
    ] == pytest.approx(
        expected_chmom,
        rel=1e-5,
        abs=1e-7,
    )


def test_retvol_and_maxret_match_direct_formula():
    dates = make_dates()

    n = len(
        dates
    )

    returns = (
        0.002
        * np.sin(
            np.linspace(
                0.0,
                20.0,
                n,
            )
        )
    )

    table = make_return_table(
        dates,
        returns,
    )

    factors = make_factors(
        dates
    )

    t = 800

    result = build_asset_characteristics(
        table,
        factors,
        start=dates[t],
        end=dates[t],
    )

    row = result.values[
        0,
        0,
    ]

    window = returns[
        t - 20 :
        t + 1
    ]

    expected_vol = float(
        np.std(
            window,
            ddof=1,
        )
    )

    expected_max = float(
        np.max(
            window
        )
    )

    assert row[
        CHARACTERISTIC_COLUMNS.index(
            "retvol"
        )
    ] == pytest.approx(
        expected_vol,
        rel=1e-5,
    )

    assert row[
        CHARACTERISTIC_COLUMNS.index(
            "maxret"
        )
    ] == pytest.approx(
        expected_max,
        rel=1e-5,
    )


def test_known_capm_beta_and_ff3_residual():
    dates = make_dates()

    factors = make_factors(
        dates
    )

    market_index = (
        factors.columns.index(
            "Mkt-RF"
        )
    )

    market = factors.returns[
        :,
        market_index,
    ].astype(
        np.float64
    )

    # Exactly generated by:
    #
    # r = alpha + 2 * market
    #
    # so CAPM beta should be 2 and FF3 residual
    # volatility should be essentially zero.
    returns = (
        0.0003
        + 2.0
        * market
    )

    table = make_return_table(
        dates,
        returns,
    )

    t = 800

    result = build_asset_characteristics(
        table,
        factors,
        start=dates[t],
        end=dates[t],
    )

    row = result.values[
        0,
        0,
    ]

    beta = row[
        CHARACTERISTIC_COLUMNS.index(
            "beta"
        )
    ]

    betasq = row[
        CHARACTERISTIC_COLUMNS.index(
            "betasq"
        )
    ]

    idiovol = row[
        CHARACTERISTIC_COLUMNS.index(
            "idiovol"
        )
    ]

    assert beta == pytest.approx(
        2.0,
        rel=1e-5,
        abs=1e-6,
    )

    assert betasq == pytest.approx(
        4.0,
        rel=1e-5,
        abs=1e-6,
    )

    assert idiovol < 1e-7


def test_future_corruption_cannot_change_past_characteristics():
    dates = make_dates()

    rng = np.random.default_rng(
        123
    )

    returns = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=len(
            dates
        ),
    )

    factors = make_factors(
        dates
    )

    table_a = make_return_table(
        dates,
        returns,
    )

    evaluation_end = 805

    result_a = build_asset_characteristics(
        table_a,
        factors,
        start=dates[800],
        end=dates[
            evaluation_end
        ],
    )

    corrupted = returns.copy()

    # Modify observations strictly AFTER the final
    # characteristic date being compared.
    corrupted[
        evaluation_end + 1 :
    ] += 0.25

    table_b = make_return_table(
        dates,
        corrupted,
    )

    result_b = build_asset_characteristics(
        table_b,
        factors,
        start=dates[800],
        end=dates[
            evaluation_end
        ],
    )

    np.testing.assert_allclose(
        result_a.values,
        result_b.values,
        rtol=0.0,
        atol=0.0,
    )


def test_requires_756_observations_of_warmup():
    dates = make_dates()

    returns = np.full(
        len(
            dates
        ),
        0.001,
    )

    table = make_return_table(
        dates,
        returns,
    )

    factors = make_factors(
        dates
    )

    with pytest.raises(
        ValueError,
        match="756",
    ):
        build_asset_characteristics(
            table,
            factors,
            start=dates[700],
            end=dates[700],
        )