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
from scipy.optimize import linprog, minimize


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




def solve_long_only_mean_cvar(
    scenarios: np.ndarray,
    *,
    confidence_level: float = 0.95,
    minimum_expected_return: float | None = None,
) -> np.ndarray:
    """Solve a long-only mean-CVaR portfolio from empirical scenarios.

    Parameters
    ----------
    scenarios:
        Scenario excess returns with shape [scenarios, assets].

    confidence_level:
        CVaR confidence level, e.g. 0.95.

    minimum_expected_return:
        Optional lower bound on the portfolio's sample expected return.

        If supplied:

            mean(scenarios) @ weights >= minimum_expected_return

    Notes
    -----
    Loss in scenario s is:

        L_s(w) = -r_s @ w

    Using the Rockafellar-Uryasev representation:

        CVaR_alpha =
            zeta
            + 1 / ((1-alpha) * S) * sum_s u_s

    subject to:

        u_s >= L_s(w) - zeta
        u_s >= 0

    The full problem is therefore a linear program.
    """

    scenarios = np.asarray(
        scenarios,
        dtype=np.float64,
    )

    if scenarios.ndim != 2:
        raise ValueError(
            "scenarios must have shape [scenarios, assets]"
        )

    n_scenarios, n_assets = scenarios.shape

    if n_scenarios < 2:
        raise ValueError(
            "At least two scenarios are required"
        )

    if n_assets < 1:
        raise ValueError(
            "At least one asset is required"
        )

    if not np.isfinite(
        scenarios
    ).all():
        raise ValueError(
            "scenarios contain NaN or infinite values"
        )

    if not (
        0.0 < confidence_level < 1.0
    ):
        raise ValueError(
            "confidence_level must lie in (0, 1)"
        )

    mean_returns = scenarios.mean(
        axis=0
    )

    if (
        minimum_expected_return is not None
        and minimum_expected_return
        > mean_returns.max() + 1e-12
    ):
        raise ValueError(
            "minimum_expected_return is infeasible"
        )

    # Variables:
    #
    # [w_1, ..., w_N, zeta, u_1, ..., u_S]
    #
    n_variables = (
        n_assets
        + 1
        + n_scenarios
    )

    zeta_index = n_assets
    u_start = n_assets + 1

    # ---------------------------------------------------------
    # Objective
    # ---------------------------------------------------------
    objective = np.zeros(
        n_variables,
        dtype=np.float64,
    )

    objective[zeta_index] = 1.0

    objective[u_start:] = (
        1.0
        / (
            (1.0 - confidence_level)
            * n_scenarios
        )
    )

    # ---------------------------------------------------------
    # Inequality constraints
    # ---------------------------------------------------------
    #
    # u_s >= -r_s @ w - zeta
    #
    # equivalent to:
    #
    # -r_s @ w - zeta - u_s <= 0
    #
    rows = []
    bounds = []

    for scenario_idx in range(
        n_scenarios
    ):
        row = np.zeros(
            n_variables,
            dtype=np.float64,
        )

        row[:n_assets] = (
            -scenarios[scenario_idx]
        )

        row[zeta_index] = -1.0

        row[
            u_start + scenario_idx
        ] = -1.0

        rows.append(row)
        bounds.append(0.0)

    # Optional mean-return constraint:
    #
    # mu @ w >= target
    #
    # -> -mu @ w <= -target
    if minimum_expected_return is not None:
        row = np.zeros(
            n_variables,
            dtype=np.float64,
        )

        row[:n_assets] = (
            -mean_returns
        )

        rows.append(row)

        bounds.append(
            -float(
                minimum_expected_return
            )
        )

    A_ub = np.asarray(
        rows,
        dtype=np.float64,
    )

    b_ub = np.asarray(
        bounds,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Fully invested
    # ---------------------------------------------------------
    A_eq = np.zeros(
        (1, n_variables),
        dtype=np.float64,
    )

    A_eq[
        0,
        :n_assets,
    ] = 1.0

    b_eq = np.array(
        [1.0],
        dtype=np.float64,
    )

    # Long-only weights.
    #
    # zeta must be unbounded because VaR/loss may be negative.
    #
    # u_s >= 0.
    variable_bounds = (
        [(0.0, 1.0)] * n_assets
        + [(None, None)]
        + [(0.0, None)] * n_scenarios
    )

    result = linprog(
        c=objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=variable_bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "Mean-CVaR optimization failed: "
            f"{result.message}"
        )

    weights = result.x[
        :n_assets
    ]

    # Remove tiny numerical LP violations.
    weights = np.clip(
        weights,
        0.0,
        1.0,
    )

    weights /= weights.sum()

    return weights