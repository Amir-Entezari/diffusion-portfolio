"""Market-stress regime utilities for regime probe."""

from __future__ import annotations

import numpy as np


def cross_sectional_rms(
    returns: np.ndarray,
) -> np.ndarray:
    """Compute cross-sectional RMS return magnitude per date."""

    values = np.asarray(
        returns,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "returns must have shape "
            "[samples, assets]"
        )

    if values.shape[1] <= 0:
        raise ValueError(
            "returns must contain assets"
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "returns contain non-finite values"
        )

    return np.sqrt(
        np.mean(
            np.square(
                values
            ),
            axis=1,
        )
    )

def trailing_cross_sectional_rms(
    history: np.ndarray,
    *,
    recent_days: int = 20,
) -> np.ndarray:
    """Compute trailing market stress from historical returns only."""

    values = np.asarray(
        history,
        dtype=np.float64,
    )

    if values.ndim != 3:
        raise ValueError(
            "history must have shape "
            "[samples, lookback, assets]"
        )

    if recent_days <= 0:
        raise ValueError(
            "recent_days must be positive"
        )

    if recent_days > values.shape[1]:
        raise ValueError(
            "recent_days cannot exceed lookback"
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "history contains non-finite values"
        )

    recent = values[
        :,
        -recent_days:,
        :,
    ]

    return np.sqrt(
        np.mean(
            np.square(
                recent
            ),
            axis=(
                1,
                2,
            ),
        )
    )

def fit_regime_thresholds(
    train_scores: np.ndarray,
    *,
    quantiles: tuple[
        float,
        float,
    ] = (
        1.0 / 3.0,
        2.0 / 3.0,
    ),
) -> tuple[
    float,
    float,
]:
    """Fit two regime thresholds using training scores only."""

    scores = np.asarray(
        train_scores,
        dtype=np.float64,
    )

    if scores.ndim != 1:
        raise ValueError(
            "train_scores must be one-dimensional"
        )

    if len(scores) == 0:
        raise ValueError(
            "train_scores cannot be empty"
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "train_scores contain non-finite values"
        )

    low_q, high_q = quantiles

    if not (
        0.0
        < low_q
        < high_q
        < 1.0
    ):
        raise ValueError(
            "quantiles must satisfy "
            "0 < low < high < 1"
        )

    low, high = np.quantile(
        scores,
        [
            low_q,
            high_q,
        ],
    )

    return (
        float(low),
        float(high),
    )


def assign_regime_labels(
    scores: np.ndarray,
    *,
    thresholds: tuple[
        float,
        float,
    ],
) -> np.ndarray:
    """Assign calm=0, normal=1, stressed=2."""

    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional"
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "scores contain non-finite values"
        )

    low, high = thresholds

    if not low < high:
        raise ValueError(
            "thresholds must satisfy low < high"
        )

    return np.digitize(
        values,
        bins=[
            low,
            high,
        ],
        right=True,
    ).astype(
        np.int64
    )