"""Kenneth French 12 Industry Portfolios data utilities.

Stage 2 of the scientific MVP uses the daily value-weighted returns of the
12 Industry Portfolios from the Kenneth R. French Data Library.

This module is intentionally responsible only for the raw tabular contract:
dates + twelve return series.

It does NOT:
- normalize the returns;
- create train/validation/test splits;
- create sliding windows;
- create PyTorch tensors.

Those operations are kept separate so leakage is easier to detect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


INDUSTRY_COLUMNS: tuple[str, ...] = (
    "NoDur",
    "Durbl",
    "Manuf",
    "Enrgy",
    "Chems",
    "BusEq",
    "Telcm",
    "Utils",
    "Shops",
    "Hlth",
    "Money",
    "Other",
)


@dataclass(frozen=True)
class ReturnTable:
    """Clean daily return table.

    Attributes
    ----------
    dates:
        Trading dates, strictly increasing.
    returns:
        Decimal returns with shape [T, 12].
        Example: 1.25% is represented as 0.0125.
    columns:
        Industry names corresponding to the second axis of ``returns``.
    """

    dates: pd.DatetimeIndex
    returns: np.ndarray
    columns: tuple[str, ...] = INDUSTRY_COLUMNS

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


def dataframe_to_return_table(frame: pd.DataFrame) -> ReturnTable:
    """Validate a clean dataframe and convert percentages to decimal returns.

    Expected input
    --------------
    Index:
        pandas DatetimeIndex.

    Columns:
        exactly the twelve Kenneth French industry columns.

    Values:
        percentage returns, as distributed by the French data library.

        For example:
            1.25  -> 0.0125
           -0.50  -> -0.0050

    Notes
    -----
    Kenneth French files use sentinel values such as -99.99 and -999 for
    missing observations. They are rejected rather than silently treated as
    extremely negative returns.
    """

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame index must be a pandas DatetimeIndex")

    missing_columns = set(INDUSTRY_COLUMNS) - set(frame.columns)
    extra_columns = set(frame.columns) - set(INDUSTRY_COLUMNS)

    if missing_columns or extra_columns:
        raise ValueError(
            "Unexpected industry columns. "
            f"Missing={sorted(missing_columns)}, "
            f"extra={sorted(extra_columns)}"
        )

    # Fix column order even if the source dataframe is shuffled.
    frame = frame.loc[:, INDUSTRY_COLUMNS].copy()

    if not frame.index.is_monotonic_increasing:
        raise ValueError("input dates are not chronological")

    if frame.index.has_duplicates:
        raise ValueError("input contains duplicate dates")

    values = frame.to_numpy(dtype=np.float64)

    # French Data Library sentinel missing values.
    if np.any(values <= -99.0):
        raise ValueError(
            "Kenneth French missing-value sentinel detected "
            "(-99.99 or -999)."
        )

    if not np.isfinite(values).all():
        raise ValueError("raw returns contain NaN or infinite values")

    # French library returns are reported in percent.
    values = values / 100.0

    return ReturnTable(
        dates=frame.index.copy(),
        returns=values.astype(np.float32),
    )