"""Tail-focused diagnostics for probabilistic return forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffusion_portfolio.evaluation.probabilistic import (
    energy_score,
    marginal_crps,
)
from diffusion_portfolio.evaluation.regimes import (
    cross_sectional_rms,
)


DEFAULT_TAIL_LEVELS = (
    0.01,
    0.05,
    0.95,
    0.99,
)


@dataclass(frozen=True)
class TailForecastMetrics:
    """Tail-sensitive validation diagnostics."""

    levels: np.ndarray

    quantile_coverage_by_asset: (
        np.ndarray
    )

    quantile_calibration_error_by_asset: (
        np.ndarray
    )

    pinball_loss_by_asset: (
        np.ndarray
    )

    stress_threshold: float
    stress_base_rate: float
    mean_predicted_stress_probability: float
    stress_brier_score: float

    n_stress_dates: int
    stress_crps_mean: float
    stress_energy_score: float


def _validate_inputs(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    samples = np.asarray(
        forecast_samples,
        dtype=np.float64,
    )

    targets = np.asarray(
        observed,
        dtype=np.float64,
    )

    if samples.ndim != 3:
        raise ValueError(
            "forecast_samples must have shape "
            "[time, scenarios, assets]"
        )

    if targets.ndim != 2:
        raise ValueError(
            "observed must have shape "
            "[time, assets]"
        )

    if (
        samples.shape[0]
        != targets.shape[0]
    ):
        raise ValueError(
            "forecast and observed time "
            "dimensions differ"
        )

    if (
        samples.shape[2]
        != targets.shape[1]
    ):
        raise ValueError(
            "forecast and observed asset "
            "dimensions differ"
        )

    if samples.shape[1] <= 0:
        raise ValueError(
            "forecast_samples must contain "
            "at least one scenario"
        )

    if not np.isfinite(
        samples
    ).all():
        raise ValueError(
            "forecast_samples contain "
            "non-finite values"
        )

    if not np.isfinite(
        targets
    ).all():
        raise ValueError(
            "observed contains "
            "non-finite values"
        )

    return (
        samples,
        targets,
    )


def quantile_tail_metrics(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
    *,
    levels: tuple[
        float,
        ...,
    ] = DEFAULT_TAIL_LEVELS,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate marginal tail quantiles.

    Returns
    -------
    coverage_by_asset:
        Shape [levels, assets].

        Fraction of observations satisfying

            y <= Q_q

        for each forecast quantile q.

    calibration_error_by_asset:
        coverage - nominal quantile level.

    pinball_loss_by_asset:
        Mean quantile loss over validation dates,
        shape [levels, assets].
    """

    (
        samples,
        targets,
    ) = _validate_inputs(
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
        or np.any(
            levels_array <= 0.0
        )
        or np.any(
            levels_array >= 1.0
        )
    ):
        raise ValueError(
            "levels must lie strictly "
            "between 0 and 1"
        )

    quantiles = np.quantile(
        samples,
        levels_array,
        axis=1,
    )

    # quantiles:
    # [levels, time, assets]

    coverage = (
        targets[
            None,
            :,
            :,
        ]
        <= quantiles
    ).mean(
        axis=1
    )

    calibration_error = (
        coverage
        - levels_array[
            :,
            None,
        ]
    )

    residual = (
        targets[
            None,
            :,
            :,
        ]
        - quantiles
    )

    pinball = np.maximum(
        levels_array[
            :,
            None,
            None,
        ]
        * residual,
        (
            levels_array[
                :,
                None,
                None,
            ]
            - 1.0
        )
        * residual,
    )

    pinball_by_asset = (
        pinball.mean(
            axis=1
        )
    )

    return (
        coverage,
        calibration_error,
        pinball_by_asset,
    )


def fit_stress_threshold(
    train_targets: np.ndarray,
    *,
    quantile: float = 0.95,
) -> float:
    """Fit a multivariate stress threshold on training targets only."""

    targets = np.asarray(
        train_targets,
        dtype=np.float64,
    )

    if targets.ndim != 2:
        raise ValueError(
            "train_targets must have shape "
            "[time, assets]"
        )

    if not (
        0.0
        < quantile
        < 1.0
    ):
        raise ValueError(
            "quantile must lie in (0, 1)"
        )

    scores = cross_sectional_rms(
        targets
    )

    return float(
        np.quantile(
            scores,
            quantile,
        )
    )


def stress_event_probabilities(
    forecast_samples: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Forecast probability of a high-magnitude multivariate return event."""

    samples = np.asarray(
        forecast_samples,
        dtype=np.float64,
    )

    if samples.ndim != 3:
        raise ValueError(
            "forecast_samples must have shape "
            "[time, scenarios, assets]"
        )

    if threshold <= 0:
        raise ValueError(
            "threshold must be positive"
        )

    if not np.isfinite(
        samples
    ).all():
        raise ValueError(
            "forecast_samples contain "
            "non-finite values"
        )

    scenario_rms = np.sqrt(
        np.mean(
            np.square(
                samples
            ),
            axis=2,
        )
    )

    return (
        scenario_rms
        > threshold
    ).mean(
        axis=1
    )


def brier_score(
    probabilities: np.ndarray,
    events: np.ndarray,
) -> float:
    """Binary-event Brier score."""

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    events = np.asarray(
        events,
        dtype=np.float64,
    )

    if (
        probabilities.ndim != 1
        or events.ndim != 1
    ):
        raise ValueError(
            "probabilities and events must "
            "be one-dimensional"
        )

    if probabilities.shape != (
        events.shape
    ):
        raise ValueError(
            "probabilities and events must "
            "have the same shape"
        )

    if np.any(
        (
            probabilities < 0.0
        )
        | (
            probabilities > 1.0
        )
    ):
        raise ValueError(
            "probabilities must lie in [0, 1]"
        )

    return float(
        np.mean(
            np.square(
                probabilities
                - events
            )
        )
    )


def evaluate_tail_forecast(
    forecast_samples: np.ndarray,
    observed: np.ndarray,
    train_targets: np.ndarray,
    *,
    levels: tuple[
        float,
        ...,
    ] = DEFAULT_TAIL_LEVELS,
    stress_quantile: float = 0.95,
) -> TailForecastMetrics:
    """Evaluate tail calibration and stress-event fidelity.

    The stress threshold is fitted only from training targets.
    """

    (
        samples,
        targets,
    ) = _validate_inputs(
        forecast_samples,
        observed,
    )

    (
        coverage,
        calibration_error,
        pinball,
    ) = quantile_tail_metrics(
        samples,
        targets,
        levels=levels,
    )

    threshold = fit_stress_threshold(
        train_targets,
        quantile=stress_quantile,
    )

    stress_probability = (
        stress_event_probabilities(
            samples,
            threshold=threshold,
        )
    )

    observed_stress = (
        cross_sectional_rms(
            targets
        )
        > threshold
    )

    stress_brier = brier_score(
        stress_probability,
        observed_stress.astype(
            np.float64
        ),
    )

    stress_mask = (
        observed_stress
    )

    n_stress_dates = int(
        stress_mask.sum()
    )

    if n_stress_dates == 0:
        stress_crps_mean = float(
            "nan"
        )

        stress_energy = float(
            "nan"
        )

    else:
        stress_crps_mean = float(
            marginal_crps(
                samples[
                    stress_mask
                ],
                targets[
                    stress_mask
                ],
            ).mean()
        )

        stress_energy = float(
            energy_score(
                samples[
                    stress_mask
                ],
                targets[
                    stress_mask
                ],
            )
        )

    return TailForecastMetrics(
        levels=np.asarray(
            levels,
            dtype=np.float64,
        ),
        quantile_coverage_by_asset=(
            coverage
        ),
        quantile_calibration_error_by_asset=(
            calibration_error
        ),
        pinball_loss_by_asset=(
            pinball
        ),
        stress_threshold=threshold,
        stress_base_rate=float(
            observed_stress.mean()
        ),
        mean_predicted_stress_probability=float(
            stress_probability.mean()
        ),
        stress_brier_score=stress_brier,
        n_stress_dates=n_stress_dates,
        stress_crps_mean=(
            stress_crps_mean
        ),
        stress_energy_score=(
            stress_energy
        ),
    )