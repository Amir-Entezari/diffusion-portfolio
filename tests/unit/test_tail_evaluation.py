import numpy as np
import pytest

from diffusion_portfolio.evaluation import (
    brier_score,
    fit_stress_threshold,
    quantile_tail_metrics,
    stress_event_probabilities,
)


def test_quantile_coverage_known_example():
    samples = np.array(
        [
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ],
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ],
        ]
    )

    observed = np.array(
        [
            [0.5],
            [2.5],
        ]
    )

    coverage, error, pinball = (
        quantile_tail_metrics(
            samples,
            observed,
            levels=(
                0.50,
            ),
        )
    )

    assert coverage.shape == (
        1,
        1,
    )

    assert error.shape == (
        1,
        1,
    )

    assert pinball.shape == (
        1,
        1,
    )

    assert coverage[
        0,
        0,
    ] == pytest.approx(
        0.50
    )

    assert error[
        0,
        0,
    ] == pytest.approx(
        0.0
    )


def test_pinball_loss_is_zero_for_exact_quantile():
    samples = np.array(
        [
            [
                [1.0],
                [1.0],
                [1.0],
            ]
        ]
    )

    observed = np.array(
        [
            [1.0]
        ]
    )

    _, _, pinball = (
        quantile_tail_metrics(
            samples,
            observed,
            levels=(
                0.05,
                0.95,
            ),
        )
    )

    np.testing.assert_allclose(
        pinball,
        0.0,
        atol=1e-12,
    )


def test_stress_threshold_uses_training_targets():
    train = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    threshold = fit_stress_threshold(
        train,
        quantile=0.50,
    )

    assert threshold == pytest.approx(
        2.5
    )


def test_stress_probability_known_example():
    samples = np.array(
        [
            [
                [0.0, 0.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [0.5, 0.5],
            ]
        ]
    )

    probabilities = (
        stress_event_probabilities(
            samples,
            threshold=1.0,
        )
    )

    assert probabilities.shape == (
        1,
    )

    assert probabilities[
        0
    ] == pytest.approx(
        0.50
    )


def test_brier_score_known_example():
    probabilities = np.array(
        [
            0.0,
            0.5,
            1.0,
        ]
    )

    events = np.array(
        [
            0.0,
            1.0,
            1.0,
        ]
    )

    result = brier_score(
        probabilities,
        events,
    )

    assert result == pytest.approx(
        1.0 / 12.0
    )