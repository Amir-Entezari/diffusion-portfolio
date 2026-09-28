"""Evaluation metrics for probabilistic multivariate return forecasts.

Forecast samples follow the contract:

    forecast_samples: [time, samples, assets]
    observed:         [time, assets]

All values should be expressed in raw decimal excess-return units before
calling these metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist


DEFAULT_COVERAGE_LEVELS = (
    0.50,
    0.80,
    0.90,
    0.95,
    0.99,
)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Prediction-interval calibration statistics."""

    levels: np.ndarray
    picp: np.ndarray
    ace: np.ndarray


@dataclass(frozen=True)
class ProbabilisticMetrics:
    """Summary of probabilistic forecast quality."""

    crps_by_asset: np.ndarray
    crps_mean: float
    crps_std: float
    energy_score: float
    calibration: CalibrationMetrics


def _validate_forecasts(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the probabilistic-forecast tensor contract."""

    forecast_samples = np.asarray(
        forecast_samples,
        dtype=np.float64,
    )

    observed = np.asarray(
        observed,
        dtype=np.float64,
    )

    if forecast_samples.ndim != 3:
        raise ValueError(
            "forecast_samples must have shape "
            "[time, samples, assets]"
        )

    if observed.ndim != 2:
        raise ValueError(
            "observed must have shape [time, assets]"
        )

    if (
        forecast_samples.shape[0]
        != observed.shape[0]
    ):
        raise ValueError(
            "Forecast and observed time dimensions differ"
        )

    if (
        forecast_samples.shape[2]
        != observed.shape[1]
    ):
        raise ValueError(
            "Forecast and observed asset dimensions differ"
        )

    if forecast_samples.shape[1] < 2:
        raise ValueError(
            "At least two forecast samples are required"
        )

    if not np.isfinite(
        forecast_samples
    ).all():
        raise ValueError(
            "forecast_samples contain NaN or infinite values"
        )

    if not np.isfinite(
        observed
    ).all():
        raise ValueError(
            "observed contains NaN or infinite values"
        )

    return (
        forecast_samples,
        observed,
    )


def marginal_crps(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Compute ensemble CRPS separately for each asset.

    For empirical forecast samples X_1, ..., X_M:

        CRPS =
            mean_m |X_m - y|
            - 1/2 mean_{m,m'} |X_m - X_m'|

    The pairwise term is evaluated using the sorted-sample identity, avoiding
    an O(M^2) pairwise tensor.

    Returns
    -------
    np.ndarray
        Mean CRPS over time for each asset, shape [assets].
    """

    (
        forecast_samples,
        observed,
    ) = _validate_forecasts(
        forecast_samples,
        observed,
    )

    n_samples = (
        forecast_samples.shape[1]
    )

    absolute_error = np.abs(
        forecast_samples
        - observed[:, None, :]
    ).mean(axis=1)

    sorted_samples = np.sort(
        forecast_samples,
        axis=1,
    )

    ranks = np.arange(
        1,
        n_samples + 1,
        dtype=np.float64,
    )

    coefficients = (
        2.0 * ranks
        - n_samples
        - 1.0
    )

    dispersion = np.sum(
        sorted_samples
        * coefficients[
            None,
            :,
            None,
        ],
        axis=1,
    ) / (
        n_samples ** 2
    )

    crps = (
        absolute_error
        - dispersion
    )

    return crps.mean(
        axis=0
    )


def energy_score(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Compute the multivariate Energy Score.

    For forecast vector X and observation y:

        ES =
            E ||X - y||_2
            - 1/2 E ||X - X'||_2

    The score is calculated independently at each time step and then averaged
    over time.

    Lower values indicate better multivariate probabilistic forecasts.
    """

    (
        forecast_samples,
        observed,
    ) = _validate_forecasts(
        forecast_samples,
        observed,
    )

    n_time = (
        forecast_samples.shape[0]
    )

    n_samples = (
        forecast_samples.shape[1]
    )

    scores = np.empty(
        n_time,
        dtype=np.float64,
    )

    for t in range(n_time):
        samples_t = (
            forecast_samples[t]
        )

        observed_t = observed[t]

        observation_term = np.linalg.norm(
            samples_t
            - observed_t[None, :],
            axis=1,
        ).mean()

        # scipy.pdist contains each unordered pair once.
        #
        # The Energy Score pairwise term is:
        #
        #   1 / (2 M^2) sum_{i,j} ||X_i - X_j||
        #
        # Because every non-diagonal unordered pair occurs twice
        # in the full i,j sum, this simplifies to:
        #
        #   sum(pdist(X)) / M^2
        pairwise_term = (
            pdist(
                samples_t,
                metric="euclidean",
            ).sum()
            / (n_samples ** 2)
        )

        scores[t] = (
            observation_term
            - pairwise_term
        )

    return float(
        scores.mean()
    )


def prediction_interval_calibration(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
    *,
    levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
) -> CalibrationMetrics:
    """Compute PICP and signed ACE for central prediction intervals.

    For target coverage c, the central interval is:

        [(1-c)/2, 1-(1-c)/2]

    PICP is averaged across every time point and asset.

    ACE is defined as:

        ACE = PICP - target_coverage

    Therefore:
        ACE < 0  -> under-coverage
        ACE > 0  -> over-coverage
    """

    (
        forecast_samples,
        observed,
    ) = _validate_forecasts(
        forecast_samples,
        observed,
    )

    levels_array = np.asarray(
        levels,
        dtype=np.float64,
    )

    if levels_array.ndim != 1:
        raise ValueError(
            "levels must be one-dimensional"
        )

    if (
        len(levels_array) == 0
        or np.any(levels_array <= 0.0)
        or np.any(levels_array >= 1.0)
    ):
        raise ValueError(
            "coverage levels must lie strictly between 0 and 1"
        )

    picp = np.empty(
        len(levels_array),
        dtype=np.float64,
    )

    for idx, level in enumerate(
        levels_array
    ):
        tail = (
            1.0 - level
        ) / 2.0

        lower = np.quantile(
            forecast_samples,
            tail,
            axis=1,
        )

        upper = np.quantile(
            forecast_samples,
            1.0 - tail,
            axis=1,
        )

        covered = (
            (observed >= lower)
            & (observed <= upper)
        )

        picp[idx] = covered.mean()

    ace = (
        picp
        - levels_array
    )

    return CalibrationMetrics(
        levels=levels_array,
        picp=picp,
        ace=ace,
    )


def evaluate_probabilistic_forecast(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
    *,
    coverage_levels: tuple[
        float,
        ...
    ] = DEFAULT_COVERAGE_LEVELS,
) -> ProbabilisticMetrics:
    """Evaluate one multivariate empirical predictive distribution."""

    crps_by_asset = marginal_crps(
        forecast_samples,
        observed,
    )

    es = energy_score(
        forecast_samples,
        observed,
    )

    calibration = (
        prediction_interval_calibration(
            forecast_samples,
            observed,
            levels=coverage_levels,
        )
    )

    return ProbabilisticMetrics(
        crps_by_asset=crps_by_asset,
        crps_mean=float(
            crps_by_asset.mean()
        ),
        crps_std=float(
            crps_by_asset.std(
                ddof=0
            )
        ),
        energy_score=es,
        calibration=calibration,
    )