import numpy as np
import pytest

from diffusion_portfolio.evaluation import (
    DEFAULT_COVERAGE_LEVELS,
    energy_score,
    evaluate_probabilistic_forecast,
    marginal_crps,
    prediction_interval_calibration,
)


def test_crps_known_one_dimensional_example():
    # Forecast samples: {0, 2}
    # Observation: 1
    #
    # E|X-y| = 1
    #
    # 1/2 E|X-X'|
    # = 1/2 * (4 / 4)
    # = 0.5
    #
    # CRPS = 0.5
    samples = np.array(
        [
            [
                [0.0],
                [2.0],
            ]
        ]
    )

    observed = np.array(
        [
            [1.0]
        ]
    )

    result = marginal_crps(
        samples,
        observed,
    )

    assert result.shape == (1,)

    assert result[0] == pytest.approx(
        0.5
    )


def test_energy_score_matches_crps_in_one_dimension():
    samples = np.array(
        [
            [
                [0.0],
                [2.0],
            ]
        ]
    )

    observed = np.array(
        [
            [1.0]
        ]
    )

    result = energy_score(
        samples,
        observed,
    )

    assert result == pytest.approx(
        0.5
    )


def test_perfect_empirical_forecast_has_zero_scores():
    observed = np.array(
        [
            [0.01, -0.02],
            [0.03, 0.04],
        ]
    )

    samples = np.repeat(
        observed[:, None, :],
        repeats=5,
        axis=1,
    )

    crps = marginal_crps(
        samples,
        observed,
    )

    es = energy_score(
        samples,
        observed,
    )

    np.testing.assert_allclose(
        crps,
        np.zeros(2),
        atol=1e-12,
    )

    assert es == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_picp_and_ace_known_example():
    # Same empirical distribution at two time points.
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

    # For a 50% central interval, one observation is covered
    # and one is deliberately outside.
    observed = np.array(
        [
            [1.5],
            [10.0],
        ]
    )

    result = prediction_interval_calibration(
        samples,
        observed,
        levels=(0.50,),
    )

    assert result.picp[0] == pytest.approx(
        0.50
    )

    assert result.ace[0] == pytest.approx(
        0.0
    )


def test_default_calibration_levels_match_benchmark_levels():
    observed = np.zeros(
        (3, 2)
    )

    samples = np.zeros(
        (3, 4, 2)
    )

    result = prediction_interval_calibration(
        samples,
        observed,
    )

    np.testing.assert_allclose(
        result.levels,
        np.asarray(
            DEFAULT_COVERAGE_LEVELS
        ),
    )


def test_summary_reports_crps_mean_and_std_across_assets():
    observed = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    samples = np.array(
        [
            [
                [-1.0, 1.0],
                [1.0, 1.0],
            ],
            [
                [-1.0, 1.0],
                [1.0, 1.0],
            ],
        ]
    )

    result = evaluate_probabilistic_forecast(
        samples,
        observed,
        coverage_levels=(0.5,),
    )

    assert result.crps_by_asset.shape == (
        2,
    )

    assert result.crps_mean == pytest.approx(
        result.crps_by_asset.mean()
    )

    assert result.crps_std == pytest.approx(
        result.crps_by_asset.std(
            ddof=0
        )
    )


def test_invalid_forecast_shape_is_rejected():
    samples = np.zeros(
        (10, 12)
    )

    observed = np.zeros(
        (10, 12)
    )

    with pytest.raises(
        ValueError,
        match="forecast_samples",
    ):
        marginal_crps(
            samples,
            observed,
        )