"""Diffolio-style systematic macroeconomic covariates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SYSTEMATIC_COLUMNS = (
    "tbl",
    "dp",
    "ep",
    "bm",
    "tms",
    "dfy",
    "ntis",
    "svar",
)


@dataclass(frozen=True)
class SystematicCovariateTable:
    """Chronological systematic covariates.

    values has shape:

        [time, covariates]
    """

    dates: pd.DatetimeIndex
    values: np.ndarray
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                "values must have shape "
                "[time, covariates]"
            )

        if len(self.dates) != self.values.shape[0]:
            raise ValueError(
                "dates length does not match time dimension"
            )

        if len(self.columns) != self.values.shape[1]:
            raise ValueError(
                "column count does not match values"
            )

        if not self.dates.is_monotonic_increasing:
            raise ValueError(
                "dates must be chronological"
            )

        if self.dates.has_duplicates:
            raise ValueError(
                "dates must not contain duplicates"
            )

        if not np.isfinite(
            self.values
        ).all():
            raise ValueError(
                "systematic covariates contain "
                "NaN or infinite values"
            )


def _to_numeric(
    series: pd.Series,
) -> pd.Series:
    """Convert Goyal spreadsheet values to numeric."""

    cleaned = (
        series
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.strip()
    )

    cleaned = cleaned.replace(
        {
            "": np.nan,
            "NaN": np.nan,
            "nan": np.nan,
        }
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def build_monthly_systematic_covariates(
    raw: pd.DataFrame,
) -> SystematicCovariateTable:
    """Construct the eight Diffolio systematic covariates.

    Supports both:

    1. Current Goyal workbook schema, which already contains:
           d/p, e/p, tms, dfy

    2. Older/raw schema, where those predictors must be
       constructed from:
           Index/price, D12/d12, E12/e12,
           lty, tbl, BAA, AAA
    """

    # ---------------------------------------------------------
    # Variables required regardless of source-file version.
    # ---------------------------------------------------------
    base_required = (
        "yyyymm",
        "b/m",
        "tbl",
        "ntis",
        "svar",
    )

    missing_base = [
        column
        for column in base_required
        if column not in raw.columns
    ]

    if missing_base:
        raise ValueError(
            "Missing required Goyal columns: "
            f"{missing_base}"
        )

    # ---------------------------------------------------------
    # Current official workbook already provides the four
    # derived predictors directly.
    # ---------------------------------------------------------
    derived_columns = (
        "d/p",
        "e/p",
        "tms",
        "dfy",
    )

    has_direct_predictors = all(
        column in raw.columns
        for column in derived_columns
    )

    if has_direct_predictors:
        required = (
            *base_required,
            *derived_columns,
        )

        frame = raw.loc[
            :,
            required,
        ].copy()

    else:
        # -----------------------------------------------------
        # Backward compatibility with older Goyal files.
        # -----------------------------------------------------
        if "Index" in raw.columns:
            price_column = "Index"
        elif "price" in raw.columns:
            price_column = "price"
        else:
            raise ValueError(
                "Goyal data must contain either "
                "'Index' or 'price'"
            )

        if "D12" in raw.columns:
            dividend_column = "D12"
        elif "d12" in raw.columns:
            dividend_column = "d12"
        else:
            raise ValueError(
                "Goyal data must contain either "
                "'D12' or 'd12'"
            )

        if "E12" in raw.columns:
            earnings_column = "E12"
        elif "e12" in raw.columns:
            earnings_column = "e12"
        else:
            raise ValueError(
                "Goyal data must contain either "
                "'E12' or 'e12'"
            )

        extra_required = (
            price_column,
            dividend_column,
            earnings_column,
            "AAA",
            "BAA",
            "lty",
        )

        missing_extra = [
            column
            for column in extra_required
            if column not in raw.columns
        ]

        if missing_extra:
            raise ValueError(
                "Missing required Goyal columns: "
                f"{missing_extra}"
            )

        required = (
            *base_required,
            *extra_required,
        )

        frame = raw.loc[
            :,
            required,
        ].copy()

    # ---------------------------------------------------------
    # Parse monthly dates.
    # ---------------------------------------------------------
    yyyymm = _to_numeric(
        frame["yyyymm"]
    )

    date_strings = (
        yyyymm
        .astype("Int64")
        .astype(str)
    )

    dates = pd.to_datetime(
        date_strings,
        format="%Y%m",
        errors="coerce",
    )

    # Treat observation m as available at month-end.
    dates = (
        dates
        + pd.offsets.MonthEnd(0)
    )

    frame["date"] = dates

    # ---------------------------------------------------------
    # Parse every non-date source column numerically.
    # ---------------------------------------------------------
    numeric_columns = tuple(
        column
        for column in required
        if column != "yyyymm"
    )

    for column in numeric_columns:
        frame[column] = _to_numeric(
            frame[column]
        )

    complete = (
        frame["date"].notna()
        & frame[
            list(
                numeric_columns
            )
        ].notna().all(
            axis=1
        )
    )

    if not complete.any():
        raise ValueError(
            "No complete Goyal observations found"
        )

    # Early historical rows can legitimately be incomplete.
    # Once a fully usable sample starts, internal missing rows
    # are treated as data problems rather than silently skipped.
    first_complete_position = int(
        np.flatnonzero(
            complete.to_numpy()
        )[0]
    )

    frame = frame.iloc[
        first_complete_position:
    ].copy()

    complete = complete.iloc[
        first_complete_position:
    ]

    if not complete.all():
        first_bad = int(
            np.flatnonzero(
                ~complete.to_numpy()
            )[0]
        )

        bad_row = frame.iloc[
            first_bad
        ]

        raise ValueError(
            "Missing systematic covariate data after "
            "the usable sample begins near "
            f"{bad_row['yyyymm']}"
        )

    frame = frame.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    if frame[
        "date"
    ].duplicated().any():
        raise ValueError(
            "Duplicate monthly observations found"
        )

    # ---------------------------------------------------------
    # Variables present directly in every supported schema.
    # ---------------------------------------------------------
    tbl = frame[
        "tbl"
    ].to_numpy(
        dtype=np.float64
    )

    bm = frame[
        "b/m"
    ].to_numpy(
        dtype=np.float64
    )

    ntis = frame[
        "ntis"
    ].to_numpy(
        dtype=np.float64
    )

    svar = frame[
        "svar"
    ].to_numpy(
        dtype=np.float64
    )

    # ---------------------------------------------------------
    # Prefer the predictors supplied directly by the current
    # official Goyal workbook.
    # ---------------------------------------------------------
    if has_direct_predictors:
        dp = frame[
            "d/p"
        ].to_numpy(
            dtype=np.float64
        )

        ep = frame[
            "e/p"
        ].to_numpy(
            dtype=np.float64
        )

        tms = frame[
            "tms"
        ].to_numpy(
            dtype=np.float64
        )

        dfy = frame[
            "dfy"
        ].to_numpy(
            dtype=np.float64
        )

    else:
        # -----------------------------------------------------
        # Legacy reconstruction.
        #
        # Standard Welch-Goyal definitions:
        #
        # d/p = log(D12) - log(Index)
        # e/p = log(E12) - log(Index)
        # tms = lty - tbl
        # dfy = BAA - AAA
        # -----------------------------------------------------
        price = frame[
            price_column
        ].to_numpy(
            dtype=np.float64
        )

        dividends = frame[
            dividend_column
        ].to_numpy(
            dtype=np.float64
        )

        earnings = frame[
            earnings_column
        ].to_numpy(
            dtype=np.float64
        )

        if np.any(
            price <= 0.0
        ):
            raise ValueError(
                "Price/index must be positive"
            )

        if np.any(
            dividends <= 0.0
        ):
            raise ValueError(
                "D12/d12 must be positive"
            )

        if np.any(
            earnings <= 0.0
        ):
            raise ValueError(
                "E12/e12 must be positive"
            )

        dp = (
            np.log(
                dividends
            )
            - np.log(
                price
            )
        )

        ep = (
            np.log(
                earnings
            )
            - np.log(
                price
            )
        )

        tms = (
            frame[
                "lty"
            ].to_numpy(
                dtype=np.float64
            )
            - tbl
        )

        dfy = (
            frame[
                "BAA"
            ].to_numpy(
                dtype=np.float64
            )
            - frame[
                "AAA"
            ].to_numpy(
                dtype=np.float64
            )
        )

    values = np.column_stack(
        (
            tbl,
            dp,
            ep,
            bm,
            tms,
            dfy,
            ntis,
            svar,
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Systematic covariate construction produced "
            "NaN or infinite values"
        )

    return SystematicCovariateTable(
        dates=pd.DatetimeIndex(
            frame[
                "date"
            ]
        ),
        values=values.astype(
            np.float32
        ),
        columns=SYSTEMATIC_COLUMNS,
    )

def align_systematic_covariates_to_dates(
    monthly: SystematicCovariateTable,
    dates: pd.DatetimeIndex,
) -> SystematicCovariateTable:
    """Carry the latest available monthly value forward to daily dates.

    Importantly, an observation for month m is timestamped at
    month-end. Therefore dates before that month-end cannot see
    that month's macroeconomic observation.
    """

    dates = pd.DatetimeIndex(
        dates
    )

    if not dates.is_monotonic_increasing:
        raise ValueError(
            "target dates must be chronological"
        )

    if dates.has_duplicates:
        raise ValueError(
            "target dates must not contain duplicates"
        )

    positions = np.searchsorted(
        monthly.dates.values,
        dates.values,
        side="right",
    ) - 1

    if np.any(
        positions < 0
    ):
        first_missing = int(
            np.flatnonzero(
                positions < 0
            )[0]
        )

        raise ValueError(
            "No systematic observation available on or before "
            f"{dates[first_missing].date()}"
        )

    values = monthly.values[
        positions
    ]

    return SystematicCovariateTable(
        dates=dates.copy(),
        values=values.copy(),
        columns=monthly.columns,
    )