"""Classical long-only portfolio optimization utilities.

These functions operate on expected EXCESS returns and covariance matrices.

The initial MVP baselines deliberately remain simple:
- equal weight;
- global minimum variance;
- maximum-Sharpe / tangency portfolio.

No transaction-cost penalty is applied here yet.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _validate_moments(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate expected-return and covariance inputs."""

    expected_returns = np.asarray(
        expected_returns,
        dtype=np.float64,
    )

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    if expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must have shape [assets]"
        )

    n_assets = len(expected_returns)

    if covariance.shape != (
        n_assets,
        n_assets,
    ):
        raise ValueError(
            "covariance must have shape [assets, assets]"
        )

    if not np.isfinite(
        expected_returns
    ).all():
        raise ValueError(
            "expected_returns contain NaN or infinite values"
        )

    if not np.isfinite(
        covariance
    ).all():
        raise ValueError(
            "covariance contains NaN or infinite values"
        )

    if not np.allclose(
        covariance,
        covariance.T,
        atol=1e-10,
    ):
        raise ValueError(
            "covariance must be symmetric"
        )

    return (
        expected_returns,
        covariance,
    )


def regularize_covariance(
    covariance: np.ndarray,
    *,
    ridge_multiplier: float = 1e-6,
) -> np.ndarray:
    """Add a tiny scale-aware diagonal ridge.

    The ridge is intended for numerical conditioning only, not as a tuned
    shrinkage estimator.

    The added value is:

        ridge_multiplier * average_variance

    with a small numerical floor when the covariance scale is zero.
    """

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    if covariance.ndim != 2:
        raise ValueError(
            "covariance must be two-dimensional"
        )

    if (
        covariance.shape[0]
        != covariance.shape[1]
    ):
        raise ValueError(
            "covariance must be square"
        )

    if ridge_multiplier < 0.0:
        raise ValueError(
            "ridge_multiplier cannot be negative"
        )

    n_assets = covariance.shape[0]

    average_variance = (
        np.trace(covariance)
        / n_assets
    )

    scale = max(
        float(average_variance),
        1e-12,
    )

    return (
        covariance
        + ridge_multiplier
        * scale
        * np.eye(n_assets)
    )


def equal_weight(
    n_assets: int,
) -> np.ndarray:
    """Return a fully-invested equal-weight portfolio."""

    if n_assets <= 0:
        raise ValueError(
            "n_assets must be positive"
        )

    return np.full(
        n_assets,
        1.0 / n_assets,
        dtype=np.float64,
    )


def solve_long_only_minimum_variance(
    covariance: np.ndarray,
    *,
    ridge_multiplier: float = 1e-6,
) -> np.ndarray:
    """Solve the long-only global minimum-variance portfolio.

    min_w  w' Sigma w

    subject to:

        sum(w) = 1
        w >= 0
    """

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    if (
        covariance.ndim != 2
        or covariance.shape[0]
        != covariance.shape[1]
    ):
        raise ValueError(
            "covariance must be square"
        )

    if not np.isfinite(
        covariance
    ).all():
        raise ValueError(
            "covariance contains NaN or infinite values"
        )

    if not np.allclose(
        covariance,
        covariance.T,
        atol=1e-10,
    ):
        raise ValueError(
            "covariance must be symmetric"
        )

    covariance = regularize_covariance(
        covariance,
        ridge_multiplier=ridge_multiplier,
    )

    n_assets = covariance.shape[0]

    initial = equal_weight(
        n_assets
    )

    def objective(
        weights: np.ndarray,
    ) -> float:
        return float(
            weights
            @ covariance
            @ weights
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[
            (0.0, 1.0)
            for _ in range(n_assets)
        ],
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: (
                    np.sum(w) - 1.0
                ),
            }
        ],
        options={
            "maxiter": 500,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Minimum-variance optimization failed: "
            f"{result.message}"
        )

    weights = np.clip(
        result.x,
        0.0,
        1.0,
    )

    weights /= weights.sum()

    return weights


def solve_long_only_tangency(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    ridge_multiplier: float = 1e-6,
) -> np.ndarray:
    """Solve the long-only maximum-Sharpe tangency portfolio.

    Because expected returns are excess returns, the risk-free rate in the
    Sharpe numerator is zero:

        SR(w) =
            w' mu / sqrt(w' Sigma w)

    subject to:

        sum(w) = 1
        w >= 0
    """

    (
        expected_returns,
        covariance,
    ) = _validate_moments(
        expected_returns,
        covariance,
    )

    covariance = regularize_covariance(
        covariance,
        ridge_multiplier=ridge_multiplier,
    )

    n_assets = len(
        expected_returns
    )

    initial = equal_weight(
        n_assets
    )

    def negative_sharpe(
        weights: np.ndarray,
    ) -> float:
        mean = float(
            weights
            @ expected_returns
        )

        variance = float(
            weights
            @ covariance
            @ weights
        )

        if variance <= 0.0:
            return 1e12

        return -(
            mean
            / np.sqrt(variance)
        )

    result = minimize(
        negative_sharpe,
        initial,
        method="SLSQP",
        bounds=[
            (0.0, 1.0)
            for _ in range(n_assets)
        ],
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: (
                    np.sum(w) - 1.0
                ),
            }
        ],
        options={
            "maxiter": 500,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Tangency optimization failed: "
            f"{result.message}"
        )

    weights = np.clip(
        result.x,
        0.0,
        1.0,
    )

    weights /= weights.sum()

    return weights