"""Generic return-panel contract and inclusive calendar slicing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnTable:
    """Chronological return panel for any number of assets.

    Attributes
    ----------
    dates:
        Trading dates, strictly increasing.
    returns:
        Decimal returns with shape [time, assets].
        Example: 1.25% is represented as 0.0125.
    columns:
        Asset names corresponding to the second axis of ``returns``.
    """

    dates: pd.DatetimeIndex
    returns: np.ndarray
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.returns.ndim != 2:
            raise ValueError("returns must have shape [time, assets]")

        if self.returns.shape[1] != len(self.columns):
            raise ValueError(
                f"Expected {len(self.columns)} assets, "
                f"got {self.returns.shape[1]}"
            )

        if len(self.dates) != self.returns.shape[0]:
            raise ValueError("dates and returns must have the same length")

        if not self.dates.is_monotonic_increasing:
            raise ValueError("dates must be strictly chronological")

        if self.dates.has_duplicates:
            raise ValueError("dates must not contain duplicates")

        if not np.isfinite(self.returns).all():
            raise ValueError("returns contain NaN or infinite values")


def slice_return_table(
    table: ReturnTable,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> ReturnTable:
    """Restrict a return table to an inclusive calendar period.

    Slicing is performed before sliding windows are constructed so that
    histories cannot reach outside the declared experimental sample.

    Parameters
    ----------
    table:
        Full chronological return table.

    start:
        Inclusive first date.

    end:
        Inclusive final date.
    """

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if start > end:
        raise ValueError("start date must not be after end date")

    mask = (
        (table.dates >= start)
        & (table.dates <= end)
    )

    if not mask.any():
        raise ValueError(
            f"No observations found between {start.date()} "
            f"and {end.date()}"
        )

    return ReturnTable(
        dates=table.dates[mask],
        returns=table.returns[mask].copy(),
        columns=table.columns,
    )
