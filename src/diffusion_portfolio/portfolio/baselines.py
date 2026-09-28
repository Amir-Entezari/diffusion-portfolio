"""Classical portfolio baselines derived only from historical returns."""

from __future__ import annotations

import numpy as np

from diffusion_portfolio.portfolio.optimization import (
    equal_weight,
    solve_long_only_mean_cvar,
    solve_long_only_minimum_variance,
    solve_long_only_tangency,
)


SUPPORTED_BASELINES = (
    "equal_weight",
    "historical_min_variance",
    "historical_tangency",
    "historical_mean_cvar",
)


def estimate_sample_moments(
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate sample mean and covariance from one historical window.

    Parameters
    ----------
    history:
        Raw decimal excess returns with shape [lookback, assets].
    """

    history = np.asarray(
        history,
        dtype=np.float64,
    )

    if history.ndim != 2:
        raise ValueError(
            "history must have shape [lookback, assets]"
        )

    if history.shape[0] < 2:
        raise ValueError(
            "At least two historical observations are required"
        )

    if not np.isfinite(
        history
    ).all():
        raise ValueError(
            "history contains NaN or infinite values"
        )

    mean = history.mean(
        axis=0
    )

    covariance = np.cov(
        history,
        rowvar=False,
        ddof=1,
    )

    return (
        mean,
        covariance,
    )


def historical_portfolio_weights(
    histories: np.ndarray,
    *,
    method: str,
    ridge_multiplier: float = 1e-6,
    cvar_confidence_level: float = 0.95,
) -> np.ndarray:
    """Construct one portfolio from each historical window.

    Parameters
    ----------
    histories:
        Raw decimal excess-return histories with shape:

            [time, lookback, assets]

        The window for time t must contain observations strictly before the
        realized return at t.

    method:
        One of:

            equal_weight
            historical_min_variance
            historical_tangency

    Returns
    -------
    np.ndarray
        Portfolio weights with shape [time, assets].
    """

    histories = np.asarray(
        histories,
        dtype=np.float64,
    )

    if histories.ndim != 3:
        raise ValueError(
            "histories must have shape "
            "[time, lookback, assets]"
        )

    if method not in SUPPORTED_BASELINES:
        raise ValueError(
            f"Unknown baseline method: {method}"
        )

    n_time, _, n_assets = (
        histories.shape
    )

    weights = np.empty(
        (n_time, n_assets),
        dtype=np.float64,
    )

    if method == "equal_weight":
        weights[:] = equal_weight(
            n_assets
        )

        return weights

    for t in range(n_time):
        mean, covariance = (
            estimate_sample_moments(
                histories[t]
            )
        )

        if (
            method
            == "historical_min_variance"
        ):
            weights[t] = (
                solve_long_only_minimum_variance(
                    covariance,
                    ridge_multiplier=(
                        ridge_multiplier
                    ),
                )
            )

        elif (
            method
            == "historical_tangency"
        ):
            weights[t] = (
                solve_long_only_tangency(
                    mean,
                    covariance,
                    ridge_multiplier=(
                        ridge_multiplier
                    ),
                )
            )
        elif (
            method
            == "historical_mean_cvar"
        ):
            # Estimated return of the equal-weight portfolio
            # over exactly the same historical information set.
            equal_weight_mean = float(
                mean.mean()
            )

            weights[t] = (
                solve_long_only_mean_cvar(
                    histories[t],
                    confidence_level=(
                        cvar_confidence_level
                    ),
                    minimum_expected_return=(
                        equal_weight_mean
                    ),
                )
            )
    return weights