"""Return transformations used by the portfolio experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from diffusion_portfolio.data.sources.fama_french import (
    RiskFreeSeries,
)
from diffusion_portfolio.data.core import (
    ReturnTable,
)


def to_excess_returns(
    table: ReturnTable,
    risk_free: RiskFreeSeries,
) -> ReturnTable:
    """Convert asset returns to excess returns.

    For every trading day t and asset i:

        excess_return[t, i]
            = asset_return[t, i] - risk_free[t]

    Exact calendar alignment is required. Missing risk-free dates are
    considered an error rather than silently forward-filled.

    Parameters
    ----------
    table:
        Asset returns with shape [T, N].

    risk_free:
        Daily risk-free returns.

    Returns
    -------
    ReturnTable
        Excess returns with the exact same dates and columns as ``table``.
    """

    rf_by_date = pd.Series(
        risk_free.returns,
        index=risk_free.dates,
    )

    aligned_rf = rf_by_date.reindex(
        table.dates
    )

    if aligned_rf.isna().any():
        missing_dates = table.dates[
            aligned_rf.isna().to_numpy()
        ]

        preview = ", ".join(
            str(date.date())
            for date in missing_dates[:5]
        )

        raise ValueError(
            "Risk-free series is missing dates required "
            f"by the asset return table: {preview}"
        )

    rf_values = aligned_rf.to_numpy(
        dtype=np.float32
    )

    excess = (
        table.returns
        - rf_values[:, None]
    )

    if not np.isfinite(excess).all():
        raise ValueError(
            "Excess-return construction produced "
            "NaN or infinite values"
        )

    return ReturnTable(
        dates=table.dates.copy(),
        returns=excess.astype(np.float32),
        columns=table.columns,
    )