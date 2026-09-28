"""Portfolio backtesting and benchmark-aligned performance metrics.

The evaluator operates on realized asset EXCESS returns.

For each date t:

    portfolio_excess_return[t]
        = weights[t] @ asset_excess_returns[t]

The primary benchmark experiment is frictionless. Transaction-cost accounting
is intentionally handled separately so the core evaluator remains unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioMetrics:
    """Summary statistics for a portfolio excess-return stream."""

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    average_monthly_turnover: float


@dataclass(frozen=True)
class BacktestResult:
    """Complete result of a portfolio backtest."""

    dates: pd.DatetimeIndex
    weights: np.ndarray
    excess_returns: np.ndarray
    cumulative_growth: np.ndarray
    one_way_turnover: np.ndarray
    metrics: PortfolioMetrics


def _validate_inputs(
    asset_excess_returns: np.ndarray,
    weights: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    allow_short: bool,
    weight_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the common portfolio-backtest contract."""

    asset_excess_returns = np.asarray(
        asset_excess_returns,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if asset_excess_returns.ndim != 2:
        raise ValueError(
            "asset_excess_returns must have shape [time, assets]"
        )

    if weights.ndim != 2:
        raise ValueError(
            "weights must have shape [time, assets]"
        )

    if asset_excess_returns.shape != weights.shape:
        raise ValueError(
            "asset_excess_returns and weights must have "
            "identical [time, assets] shapes"
        )

    if len(dates) != asset_excess_returns.shape[0]:
        raise ValueError(
            "dates length must equal the number of time steps"
        )

    if not dates.is_monotonic_increasing:
        raise ValueError(
            "dates must be chronological"
        )

    if dates.has_duplicates:
        raise ValueError(
            "dates must not contain duplicates"
        )

    if not np.isfinite(
        asset_excess_returns
    ).all():
        raise ValueError(
            "asset_excess_returns contain NaN or infinite values"
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            "weights contain NaN or infinite values"
        )

    weight_sums = weights.sum(
        axis=1
    )

    if not np.allclose(
        weight_sums,
        1.0,
        atol=weight_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "Portfolio weights must sum to one at every time step"
        )

    if (
        not allow_short
        and np.any(
            weights < -weight_tolerance
        )
    ):
        raise ValueError(
            "Negative weights are not allowed "
            "for a long-only portfolio"
        )

    return asset_excess_returns, weights


def compute_one_way_turnover(
    weights: np.ndarray,
) -> np.ndarray:
    """Compute daily one-way target-weight turnover.

    For t > 0:

        turnover[t]
            = 0.5 * sum_i |w[t,i] - w[t-1,i]|

    The initial portfolio establishment is not counted as turnover, so
    turnover[0] = 0.

    This measures changes in target allocations. It intentionally does not
    model intra-period weight drift; that belongs to the later transaction-
    cost accounting layer.
    """

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if weights.ndim != 2:
        raise ValueError(
            "weights must have shape [time, assets]"
        )

    turnover = np.zeros(
        weights.shape[0],
        dtype=np.float64,
    )

    if len(weights) > 1:
        turnover[1:] = (
            0.5
            * np.abs(
                weights[1:]
                - weights[:-1]
            ).sum(axis=1)
        )

    return turnover


def average_monthly_one_way_turnover(
    dates: pd.DatetimeIndex,
    daily_turnover: np.ndarray,
) -> float:
    """Average monthly one-way turnover.

    Daily turnover is summed within each calendar month and then averaged
    across months.
    """

    daily_turnover = np.asarray(
        daily_turnover,
        dtype=np.float64,
    )

    if daily_turnover.ndim != 1:
        raise ValueError(
            "daily_turnover must have shape [time]"
        )

    if len(dates) != len(
        daily_turnover
    ):
        raise ValueError(
            "dates and daily_turnover must have the same length"
        )

    series = pd.Series(
        daily_turnover,
        index=dates,
    )

    monthly = series.groupby(
        series.index.to_period("M")
    ).sum()

    return float(
        monthly.mean()
    )


def cumulative_growth(
    returns: np.ndarray,
) -> np.ndarray:
    """Compute cumulative growth from a return stream.

    growth[t] = product_{s <= t} (1 + return[s])

    Here the input is the portfolio excess-return stream, so this is a
    benchmark-style cumulative excess-return growth index rather than a
    literal total-wealth process including the risk-free return.
    """

    returns = np.asarray(
        returns,
        dtype=np.float64,
    )

    if returns.ndim != 1:
        raise ValueError(
            "returns must have shape [time]"
        )

    if np.any(
        returns <= -1.0
    ):
        raise ValueError(
            "Cannot compound returns <= -100%"
        )

    return np.cumprod(
        1.0 + returns
    )


def maximum_drawdown(
    growth: np.ndarray,
) -> float:
    """Maximum peak-to-trough drawdown of a cumulative growth index."""

    growth = np.asarray(
        growth,
        dtype=np.float64,
    )

    if growth.ndim != 1:
        raise ValueError(
            "growth must have shape [time]"
        )

    if len(growth) == 0:
        raise ValueError(
            "growth cannot be empty"
        )

    running_peak = np.maximum.accumulate(
        growth
    )

    drawdowns = (
        growth / running_peak
        - 1.0
    )

    return float(
        drawdowns.min()
    )


def compute_portfolio_metrics(
    excess_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    one_way_turnover: np.ndarray,
    *,
    annualization_factor: int = 252,
) -> PortfolioMetrics:
    """Calculate benchmark-aligned portfolio summary statistics."""

    excess_returns = np.asarray(
        excess_returns,
        dtype=np.float64,
    )

    if excess_returns.ndim != 1:
        raise ValueError(
            "excess_returns must have shape [time]"
        )

    if len(excess_returns) < 2:
        raise ValueError(
            "At least two return observations are required"
        )

    if annualization_factor <= 0:
        raise ValueError(
            "annualization_factor must be positive"
        )

    if not np.isfinite(
        excess_returns
    ).all():
        raise ValueError(
            "excess_returns contain NaN or infinite values"
        )

    annualized_return = float(
        excess_returns.mean()
        * annualization_factor
    )

    annualized_volatility = float(
        excess_returns.std(
            ddof=1
        )
        * np.sqrt(
            annualization_factor
        )
    )

    if annualized_volatility > 0.0:
        sharpe_ratio = (
            annualized_return
            / annualized_volatility
        )
    else:
        sharpe_ratio = np.nan

    growth = cumulative_growth(
        excess_returns
    )

    max_drawdown = maximum_drawdown(
        growth
    )

    if max_drawdown < 0.0:
        calmar_ratio = (
            annualized_return
            / abs(max_drawdown)
        )
    else:
        calmar_ratio = np.nan

    monthly_turnover = (
        average_monthly_one_way_turnover(
            dates,
            one_way_turnover,
        )
    )

    return PortfolioMetrics(
        annualized_return=float(
            annualized_return
        ),
        annualized_volatility=float(
            annualized_volatility
        ),
        sharpe_ratio=float(
            sharpe_ratio
        ),
        max_drawdown=float(
            max_drawdown
        ),
        calmar_ratio=float(
            calmar_ratio
        ),
        average_monthly_turnover=float(
            monthly_turnover
        ),
    )


def evaluate_portfolio(
    asset_excess_returns: np.ndarray,
    weights: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    annualization_factor: int = 252,
    allow_short: bool = False,
    weight_tolerance: float = 1e-6,
) -> BacktestResult:
    """Evaluate a sequence of portfolio allocations.

    Parameters
    ----------
    asset_excess_returns:
        Realized decimal excess returns, shape [T, N].

    weights:
        Portfolio weights applied to each corresponding realized return,
        shape [T, N].

        ``weights[t]`` must have been determined using information available
        before observing ``asset_excess_returns[t]``.

    dates:
        Corresponding target dates.

    annualization_factor:
        Usually 252 for daily U.S. equity data.

    allow_short:
        Whether negative portfolio weights are permitted.
    """

    (
        asset_excess_returns,
        weights,
    ) = _validate_inputs(
        asset_excess_returns,
        weights,
        dates,
        allow_short=allow_short,
        weight_tolerance=weight_tolerance,
    )

    portfolio_excess_returns = np.sum(
        weights
        * asset_excess_returns,
        axis=1,
    )

    turnover = compute_one_way_turnover(
        weights
    )

    growth = cumulative_growth(
        portfolio_excess_returns
    )

    metrics = compute_portfolio_metrics(
        portfolio_excess_returns,
        dates,
        turnover,
        annualization_factor=annualization_factor,
    )

    return BacktestResult(
        dates=dates.copy(),
        weights=weights.astype(
            np.float32
        ),
        excess_returns=(
            portfolio_excess_returns.astype(
                np.float32
            )
        ),
        cumulative_growth=growth.astype(
            np.float64
        ),
        one_way_turnover=turnover.astype(
            np.float64
        ),
        metrics=metrics,
    )