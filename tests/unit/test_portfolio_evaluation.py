import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.evaluation import (
    average_monthly_one_way_turnover,
    compute_one_way_turnover,
    cumulative_growth,
    evaluate_portfolio,
    maximum_drawdown,
)


def test_equal_weight_portfolio_is_cross_sectional_mean():
    dates = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="B",
    )

    returns = np.array(
        [
            [0.01, 0.03],
            [-0.02, 0.00],
            [0.04, 0.02],
            [0.01, -0.01],
        ],
        dtype=np.float64,
    )

    weights = np.full(
        (4, 2),
        0.5,
    )

    result = evaluate_portfolio(
        returns,
        weights,
        dates,
    )

    expected = returns.mean(
        axis=1
    )

    np.testing.assert_allclose(
        result.excess_returns,
        expected,
    )


def test_annualized_metrics_follow_standard_formulas():
    dates = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="B",
    )

    portfolio_returns = np.array(
        [
            0.01,
            -0.005,
            0.015,
            0.00,
            0.005,
        ],
        dtype=np.float64,
    )

    # One asset, therefore weight = 1.
    asset_returns = portfolio_returns[
        :, None
    ]

    weights = np.ones(
        (5, 1),
        dtype=np.float64,
    )

    result = evaluate_portfolio(
        asset_returns,
        weights,
        dates,
        annualization_factor=252,
    )

    expected_return = (
        portfolio_returns.mean()
        * 252
    )

    expected_vol = (
        portfolio_returns.std(
            ddof=1
        )
        * np.sqrt(252)
    )

    expected_sharpe = (
        expected_return
        / expected_vol
    )

    assert (
        result.metrics.annualized_return
        == pytest.approx(expected_return)
    )

    assert (
        result.metrics.annualized_volatility
        == pytest.approx(expected_vol)
    )

    assert (
        result.metrics.sharpe_ratio
        == pytest.approx(expected_sharpe)
    )


def test_one_way_turnover_has_known_values():
    weights = np.array(
        [
            [0.5, 0.5],
            [0.7, 0.3],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )

    turnover = compute_one_way_turnover(
        weights
    )

    np.testing.assert_allclose(
        turnover,
        [
            0.0,
            0.2,
            0.5,
        ],
    )


def test_constant_weights_have_zero_turnover():
    weights = np.full(
        (100, 4),
        0.25,
    )

    turnover = compute_one_way_turnover(
        weights
    )

    np.testing.assert_array_equal(
        turnover,
        np.zeros(100),
    )


def test_average_monthly_turnover_sums_daily_trading():
    dates = pd.DatetimeIndex(
        [
            "2020-01-02",
            "2020-01-03",
            "2020-02-03",
            "2020-02-04",
        ]
    )

    daily = np.array(
        [
            0.0,
            0.2,
            0.1,
            0.3,
        ]
    )

    # January = 0.2
    # February = 0.4
    # average = 0.3
    result = (
        average_monthly_one_way_turnover(
            dates,
            daily,
        )
    )

    assert result == pytest.approx(
        0.3
    )


def test_maximum_drawdown_has_known_value():
    growth = np.array(
        [
            1.00,
            1.10,
            0.88,
            0.99,
        ]
    )

    # Peak = 1.10, trough = 0.88
    # drawdown = 0.88 / 1.10 - 1 = -0.20
    result = maximum_drawdown(
        growth
    )

    assert result == pytest.approx(
        -0.20
    )


def test_cumulative_growth_compounds_returns():
    returns = np.array(
        [
            0.10,
            -0.10,
        ]
    )

    growth = cumulative_growth(
        returns
    )

    np.testing.assert_allclose(
        growth,
        [
            1.10,
            0.99,
        ],
    )


def test_non_unit_sum_weights_are_rejected():
    dates = pd.date_range(
        "2020-01-01",
        periods=3,
        freq="B",
    )

    returns = np.zeros(
        (3, 2)
    )

    weights = np.array(
        [
            [0.8, 0.8],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        evaluate_portfolio(
            returns,
            weights,
            dates,
        )


def test_negative_weights_rejected_for_long_only():
    dates = pd.date_range(
        "2020-01-01",
        periods=2,
        freq="B",
    )

    returns = np.zeros(
        (2, 2)
    )

    weights = np.array(
        [
            [1.1, -0.1],
            [1.1, -0.1],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Negative weights",
    ):
        evaluate_portfolio(
            returns,
            weights,
            dates,
            allow_short=False,
        )


def test_short_weights_can_be_allowed_explicitly():
    dates = pd.date_range(
        "2020-01-01",
        periods=2,
        freq="B",
    )

    returns = np.array(
        [
            [0.01, -0.01],
            [0.02, 0.00],
        ]
    )

    weights = np.array(
        [
            [1.1, -0.1],
            [1.1, -0.1],
        ]
    )

    result = evaluate_portfolio(
        returns,
        weights,
        dates,
        allow_short=True,
    )

    assert len(
        result.excess_returns
    ) == 2