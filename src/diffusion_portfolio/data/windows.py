"""Sliding-window utilities for return forecasting/generation."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from diffusion_portfolio.data.core import ReturnTable


@dataclass(frozen=True)
class WindowedReturns:
    """Supervised windows built from a chronological return table."""

    history: np.ndarray
    target: np.ndarray
    target_dates: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if self.history.ndim != 3:
            raise ValueError("history must have shape [samples, lookback, assets]")

        if self.target.ndim != 3:
            raise ValueError("target must have shape [samples, horizon, assets]")

        if self.history.shape[0] != self.target.shape[0]:
            raise ValueError("history and target sample counts differ")

        if self.target.shape[0] != len(self.target_dates):
            raise ValueError("target_dates length does not match samples")

        if self.history.shape[2] != self.target.shape[2]:
            raise ValueError("history and target asset dimensions differ")


def make_windows(
    table: ReturnTable,
    lookback: int,
    horizon: int = 1,
    *,
    max_gap_days: int = 7,
) -> WindowedReturns:
    """Create chronological history/target windows.

    For each sample:

        history = returns[t-lookback:t]
        target  = returns[t:t+horizon]

    Therefore every target observation occurs strictly after its history.

    Parameters
    ----------
    table:
        Chronological return table.

    lookback:
        Number of historical trading days available to the model.

    horizon:
        Number of future trading days to predict/generate.

    max_gap_days:
        Maximum elapsed calendar days between adjacent observations within
        one history-plus-target window. Longer gaps start a new segment that
        needs its own complete lookback. Seven days permits weekends and
        observed KF12 market closures; this is not a trading-calendar check.

    Returns
    -------
    WindowedReturns
        history:
            [samples, lookback, assets]

        target:
            [samples, horizon, assets]

        target_dates:
            Date corresponding to the FIRST target observation.
    """

    if lookback <= 0:
        raise ValueError("lookback must be positive")

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if isinstance(max_gap_days, bool) or not isinstance(max_gap_days, int) or max_gap_days <= 0:
        raise ValueError("max_gap_days must be a positive integer")

    returns = table.returns
    dates = table.dates

    n_time, n_assets = returns.shape

    n_candidates = n_time - lookback - horizon + 1

    if n_candidates <= 0:
        raise ValueError(
            "Not enough observations for requested "
            f"lookback={lookback}, horizon={horizon}"
        )

    # A window is valid only when its first history and last target observation
    # belong to the same continuous segment. This covers the history/target
    # join and gaps inside multi-step targets, without padding missing returns.
    breaks = (dates[1:] - dates[:-1]) > pd.Timedelta(days=max_gap_days)
    segments = np.r_[0, np.cumsum(breaks)]
    starts = np.arange(lookback, n_time - horizon + 1)
    starts = starts[segments[starts - lookback] == segments[starts + horizon - 1]]
    n_samples = len(starts)
    if n_samples == 0:
        raise ValueError(
            "No windows remain after continuity segmentation: each segment "
            f"needs lookback={lookback} plus horizon={horizon} observations "
            f"with adjacent gaps <= {max_gap_days} calendar days"
        )
    if n_samples < n_candidates:
        warnings.warn(
            f"Window continuity excluded {n_candidates - n_samples}/{n_candidates} "
            f"candidate windows across {int(breaks.sum())} gaps exceeding "
            f"{max_gap_days} calendar days; retained {n_samples}",
            UserWarning,
        )

    history = np.empty(
        (n_samples, lookback, n_assets),
        dtype=np.float32,
    )

    target = np.empty(
        (n_samples, horizon, n_assets),
        dtype=np.float32,
    )

    target_dates: list[pd.Timestamp] = []

    for sample_idx, target_start in enumerate(starts):
        target_end = target_start + horizon

        history[sample_idx] = returns[
            target_start - lookback : target_start
        ]

        target[sample_idx] = returns[
            target_start:target_end
        ]

        target_dates.append(dates[target_start])

    return WindowedReturns(
        history=history,
        target=target,
        target_dates=pd.DatetimeIndex(target_dates),
    )
