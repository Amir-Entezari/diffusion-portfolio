import numpy as np

from diffusion_portfolio.evaluation.regimes import (
    assign_regime_labels,
    cross_sectional_rms,
    fit_regime_thresholds,
)


def test_stress_score():
    returns = np.array(
        [
            [1.0, 1.0],
            [3.0, 4.0],
        ]
    )

    scores = cross_sectional_rms(
        returns
    )

    assert np.isclose(
        scores[0],
        1.0,
    )

    assert np.isclose(
        scores[1],
        np.sqrt(12.5),
    )


def test_regime_labels():
    scores = np.array(
        [
            0.1,
            0.3,
            0.6,
            0.9,
        ]
    )

    labels = assign_regime_labels(
        scores,
        thresholds=(
            0.3,
            0.6,
        ),
    )

    assert labels.tolist() == [
        0,
        0,
        1,
        2,
    ]
    
    
def test_thresholds_are_fitted_from_scores():
    scores = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]
    )

    low, high = (
        fit_regime_thresholds(
            scores,
            quantiles=(
                1.0 / 3.0,
                2.0 / 3.0,
            ),
        )
    )

    assert low < high
    assert scores.min() < low
    assert high < scores.max()