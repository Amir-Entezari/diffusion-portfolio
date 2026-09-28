import numpy as np
import pytest

from diffusion_portfolio.portfolio import (
    equal_weight,
    estimate_sample_moments,
    historical_portfolio_weights,
    solve_long_only_minimum_variance,
    solve_long_only_tangency,
)


def test_equal_weight_is_fully_invested():
    weights = equal_weight(
        4
    )

    np.testing.assert_allclose(
        weights,
        [0.25, 0.25, 0.25, 0.25],
    )

    assert weights.sum() == pytest.approx(
        1.0
    )


def test_minimum_variance_prefers_lower_variance_asset():
    covariance = np.array(
        [
            [1.0, 0.0],
            [0.0, 4.0],
        ]
    )

    weights = (
        solve_long_only_minimum_variance(
            covariance,
        )
    )

    assert weights[0] > weights[1]

    assert weights.sum() == pytest.approx(
        1.0
    )

    assert np.all(
        weights >= 0.0
    )


def test_tangency_prefers_higher_mean_when_risk_is_equal():
    expected_returns = np.array(
        [
            0.02,
            0.01,
        ]
    )

    covariance = np.eye(
        2
    )

    weights = solve_long_only_tangency(
        expected_returns,
        covariance,
    )

    assert weights[0] > weights[1]

    assert weights.sum() == pytest.approx(
        1.0
    )

    assert np.all(
        weights >= 0.0
    )


def test_sample_moments_have_correct_shapes():
    rng = np.random.default_rng(
        42
    )

    history = rng.normal(
        size=(60, 12)
    )

    mean, covariance = (
        estimate_sample_moments(
            history
        )
    )

    assert mean.shape == (
        12,
    )

    assert covariance.shape == (
        12,
        12,
    )


def test_historical_equal_weight_shape():
    rng = np.random.default_rng(
        42
    )

    histories = rng.normal(
        size=(20, 60, 12)
    )

    weights = (
        historical_portfolio_weights(
            histories,
            method="equal_weight",
        )
    )

    assert weights.shape == (
        20,
        12,
    )

    np.testing.assert_allclose(
        weights.sum(axis=1),
        np.ones(20),
    )


def test_historical_min_variance_weights_are_long_only():
    rng = np.random.default_rng(
        42
    )

    histories = rng.normal(
        size=(5, 60, 4)
    )

    weights = (
        historical_portfolio_weights(
            histories,
            method=(
                "historical_min_variance"
            ),
        )
    )

    assert np.all(
        weights >= -1e-10
    )

    np.testing.assert_allclose(
        weights.sum(axis=1),
        np.ones(5),
        atol=1e-8,
    )


def test_historical_tangency_weights_are_long_only():
    rng = np.random.default_rng(
        42
    )

    histories = rng.normal(
        loc=0.001,
        scale=0.01,
        size=(5, 60, 4),
    )

    weights = (
        historical_portfolio_weights(
            histories,
            method="historical_tangency",
        )
    )

    assert np.all(
        weights >= -1e-10
    )

    np.testing.assert_allclose(
        weights.sum(axis=1),
        np.ones(5),
        atol=1e-8,
    )


def test_unknown_baseline_is_rejected():
    histories = np.zeros(
        (5, 60, 12)
    )

    with pytest.raises(
        ValueError,
        match="Unknown baseline",
    ):
        historical_portfolio_weights(
            histories,
            method="magic",
        )