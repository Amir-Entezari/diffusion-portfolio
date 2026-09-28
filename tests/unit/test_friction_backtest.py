import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.evaluation import (
    evaluate_rebalanced_portfolio,
)


def test_weights_drift_between_rebalances():
    dates = pd.date_range(
        "2020-01-01",
        periods=2,
        freq="B",
    )

    total = np.array(
        [
            [0.10, 0.00],
            [0.00, 0.00],
        ]
    )

    excess = total.copy()

    targets = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    result = evaluate_rebalanced_portfolio(
        excess,
        total,
        targets,
        dates,
        rebalance_interval=5,
        transaction_cost_bps=0.0,
    )

    expected = np.array(
        [
            0.55 / 1.05,
            0.50 / 1.05,
        ]
    )

    np.testing.assert_allclose(
        result.applied_weights[1],
        expected,
    )


def test_only_scheduled_days_rebalance():
    dates = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="B",
    )

    total = np.zeros(
        (4, 2)
    )

    excess = np.zeros(
        (4, 2)
    )

    targets = np.array(
        [
            [0.5, 0.5],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = evaluate_rebalanced_portfolio(
        excess,
        total,
        targets,
        dates,
        rebalance_interval=2,
        transaction_cost_bps=0.0,
    )

    np.testing.assert_allclose(
        result.applied_weights[0],
        [0.5, 0.5],
    )

    np.testing.assert_allclose(
        result.applied_weights[1],
        [0.5, 0.5],
    )

    np.testing.assert_allclose(
        result.applied_weights[2],
        [1.0, 0.0],
    )

    np.testing.assert_allclose(
        result.applied_weights[3],
        [1.0, 0.0],
    )


def test_transaction_cost_matches_turnover():
    dates = pd.date_range(
        "2020-01-01",
        periods=3,
        freq="B",
    )

    total = np.zeros(
        (3, 2)
    )

    excess = np.zeros(
        (3, 2)
    )

    targets = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 0.0],
        ]
    )

    result = evaluate_rebalanced_portfolio(
        excess,
        total,
        targets,
        dates,
        rebalance_interval=2,
        transaction_cost_bps=10.0,
    )

    # 0.5 * (|1 - .5| + |0 - .5|) = 0.5
    assert (
        result.one_way_turnover[2]
        == pytest.approx(0.5)
    )

    # 10 bps = 0.001.
    # cost = 0.001 * 0.5 = 0.0005.
    assert (
        result.transaction_costs[2]
        == pytest.approx(0.0005)
    )

    assert (
        result.net_excess_returns[2]
        == pytest.approx(-0.0005)
    )


def test_initial_portfolio_is_not_charged():
    dates = pd.date_range(
        "2020-01-01",
        periods=2,
        freq="B",
    )

    returns = np.zeros(
        (2, 2)
    )

    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = evaluate_rebalanced_portfolio(
        returns,
        returns,
        targets,
        dates,
        rebalance_interval=1,
        transaction_cost_bps=10.0,
    )

    assert result.one_way_turnover[0] == 0.0
    assert result.transaction_costs[0] == 0.0


def test_zero_cost_makes_net_equal_gross():
    rng = np.random.default_rng(
        42
    )

    dates = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="B",
    )

    total = rng.normal(
        0.001,
        0.01,
        size=(10, 3),
    )

    excess = total.copy()

    targets = np.full(
        (10, 3),
        1.0 / 3.0,
    )

    result = evaluate_rebalanced_portfolio(
        excess,
        total,
        targets,
        dates,
        rebalance_interval=5,
        transaction_cost_bps=0.0,
    )

    np.testing.assert_allclose(
        result.net_excess_returns,
        result.gross_excess_returns,
    )


def test_costs_cannot_improve_net_return():
    dates = pd.date_range(
        "2020-01-01",
        periods=3,
        freq="B",
    )

    total = np.zeros(
        (3, 2)
    )

    excess = np.zeros(
        (3, 2)
    )

    targets = np.array(
        [
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = evaluate_rebalanced_portfolio(
        excess,
        total,
        targets,
        dates,
        rebalance_interval=1,
        transaction_cost_bps=10.0,
    )

    assert np.all(
        result.net_excess_returns
        <= result.gross_excess_returns
    )