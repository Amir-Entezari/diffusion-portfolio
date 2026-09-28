"""Rebalancing and transaction-cost aware portfolio evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diffusion_portfolio.evaluation.portfolio import (
    PortfolioMetrics,
    compute_portfolio_metrics,
    cumulative_growth,
)


@dataclass(frozen=True)
class FrictionBacktestResult:
    """Portfolio result with explicit rebalancing and trading costs."""

    dates: pd.DatetimeIndex

    target_weights: np.ndarray
    applied_weights: np.ndarray
    pre_trade_weights: np.ndarray

    gross_excess_returns: np.ndarray
    net_excess_returns: np.ndarray

    one_way_turnover: np.ndarray
    transaction_costs: np.ndarray

    gross_cumulative_growth: np.ndarray
    net_cumulative_growth: np.ndarray

    gross_metrics: PortfolioMetrics
    net_metrics: PortfolioMetrics


def evaluate_rebalanced_portfolio(
    asset_excess_returns: np.ndarray,
    asset_total_returns: np.ndarray,
    target_weights: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    rebalance_interval: int,
    transaction_cost_bps: float,
    annualization_factor: int = 252,
    allow_short: bool = False,
    weight_tolerance: float = 1e-6,
) -> FrictionBacktestResult:
    """Evaluate target portfolios with weight drift and transaction costs.

    Parameters
    ----------
    asset_excess_returns:
        Realized asset excess returns, shape [T, N].

    asset_total_returns:
        Realized total asset returns, shape [T, N].

        Total returns are needed to propagate actual portfolio weights
        between rebalance dates.

    target_weights:
        Desired portfolio at every date, shape [T, N].

        Only rows corresponding to rebalance dates are actually applied.

    rebalance_interval:
        Rebalance every N trading days.

        1 -> daily
        5 -> every five trading days

    transaction_cost_bps:
        Cost in basis points per unit of one-way turnover.

        With one-way turnover:

            0.5 * sum_i |w_new_i - w_old_i|

        the cost deducted from portfolio excess return is:

            cost_rate * one_way_turnover

    Notes
    -----
    The initial portfolio establishment is not charged transaction costs.

    This implementation models realized execution costs. It does not place
    a trading-cost penalty inside the optimizer itself.
    """

    excess = np.asarray(
        asset_excess_returns,
        dtype=np.float64,
    )

    total = np.asarray(
        asset_total_returns,
        dtype=np.float64,
    )

    targets = np.asarray(
        target_weights,
        dtype=np.float64,
    )

    if excess.ndim != 2:
        raise ValueError(
            "asset_excess_returns must have shape [time, assets]"
        )

    if total.shape != excess.shape:
        raise ValueError(
            "asset_total_returns must match excess-return shape"
        )

    if targets.shape != excess.shape:
        raise ValueError(
            "target_weights must match return shape"
        )

    if len(dates) != excess.shape[0]:
        raise ValueError(
            "dates length must match return observations"
        )

    if rebalance_interval <= 0:
        raise ValueError(
            "rebalance_interval must be positive"
        )

    if transaction_cost_bps < 0.0:
        raise ValueError(
            "transaction_cost_bps cannot be negative"
        )

    if not dates.is_monotonic_increasing:
        raise ValueError(
            "dates must be chronological"
        )

    if dates.has_duplicates:
        raise ValueError(
            "dates must not contain duplicates"
        )

    if not np.isfinite(excess).all():
        raise ValueError(
            "asset_excess_returns contain NaN or infinite values"
        )

    if not np.isfinite(total).all():
        raise ValueError(
            "asset_total_returns contain NaN or infinite values"
        )

    if not np.isfinite(targets).all():
        raise ValueError(
            "target_weights contain NaN or infinite values"
        )

    if np.any(total <= -1.0):
        raise ValueError(
            "asset_total_returns cannot be <= -100%"
        )

    if not np.allclose(
        targets.sum(axis=1),
        1.0,
        atol=weight_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "target weights must sum to one"
        )

    if (
        not allow_short
        and np.any(
            targets < -weight_tolerance
        )
    ):
        raise ValueError(
            "Negative target weights are not allowed"
        )

    n_time, n_assets = excess.shape

    pre_trade = np.empty(
        (n_time, n_assets),
        dtype=np.float64,
    )

    applied = np.empty_like(
        pre_trade
    )

    turnover = np.zeros(
        n_time,
        dtype=np.float64,
    )

    costs = np.zeros(
        n_time,
        dtype=np.float64,
    )

    gross_excess = np.zeros(
        n_time,
        dtype=np.float64,
    )

    net_excess = np.zeros(
        n_time,
        dtype=np.float64,
    )

    cost_rate = (
        transaction_cost_bps
        / 10_000.0
    )

    # Initial establishment:
    # target portfolio is applied without charging turnover.
    current_weights = targets[0].copy()

    for t in range(n_time):
        if t == 0:
            pre_trade[t] = (
                current_weights
            )

            applied[t] = (
                targets[t]
            )

        else:
            pre_trade[t] = (
                current_weights
            )

            if (
                t % rebalance_interval
                == 0
            ):
                applied[t] = (
                    targets[t]
                )

                turnover[t] = (
                    0.5
                    * np.abs(
                        applied[t]
                        - pre_trade[t]
                    ).sum()
                )

            else:
                applied[t] = (
                    pre_trade[t]
                )

        costs[t] = (
            cost_rate
            * turnover[t]
        )

        gross_excess[t] = float(
            applied[t]
            @ excess[t]
        )

        net_excess[t] = (
            gross_excess[t]
            - costs[t]
        )

        # Let the actual portfolio weights drift according to
        # realized TOTAL asset returns.
        asset_growth = (
            1.0
            + total[t]
        )

        end_values = (
            applied[t]
            * asset_growth
        )

        portfolio_growth = (
            end_values.sum()
        )

        if portfolio_growth <= 0.0:
            raise ValueError(
                "Portfolio total value became non-positive"
            )

        current_weights = (
            end_values
            / portfolio_growth
        )

    gross_growth = cumulative_growth(
        gross_excess
    )

    net_growth = cumulative_growth(
        net_excess
    )

    gross_metrics = compute_portfolio_metrics(
        gross_excess,
        dates,
        turnover,
        annualization_factor=annualization_factor,
    )

    net_metrics = compute_portfolio_metrics(
        net_excess,
        dates,
        turnover,
        annualization_factor=annualization_factor,
    )

    return FrictionBacktestResult(
        dates=dates.copy(),
        target_weights=targets.astype(
            np.float32
        ),
        applied_weights=applied.astype(
            np.float64
        ),
        pre_trade_weights=pre_trade.astype(
            np.float64
        ),
        gross_excess_returns=gross_excess,
        net_excess_returns=net_excess,
        one_way_turnover=turnover,
        transaction_costs=costs,
        gross_cumulative_growth=gross_growth,
        net_cumulative_growth=net_growth,
        gross_metrics=gross_metrics,
        net_metrics=net_metrics,
    )