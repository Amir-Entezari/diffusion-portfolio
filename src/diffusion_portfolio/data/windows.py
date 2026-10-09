"""Sliding-window utilities for return forecasting/generation."""

from __future__ import annotations

from dataclasses import dataclass

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

    returns = table.returns
    dates = table.dates

    n_time, n_assets = returns.shape

    n_samples = n_time - lookback - horizon + 1

    if n_samples <= 0:
        raise ValueError(
            "Not enough observations for requested "
            f"lookback={lookback}, horizon={horizon}"
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

    for sample_idx in range(n_samples):
        target_start = sample_idx + lookback
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