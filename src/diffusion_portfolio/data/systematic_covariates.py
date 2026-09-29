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

    Expected raw columns follow the Goyal-Welch predictor
    dataset:

        yyyymm
        Index
        D12
        E12
        b/m
        tbl
        AAA
        BAA
        lty
        ntis
        svar

    Derived variables:

        dp  = log(D12) - log(Index)
        ep  = log(E12) - log(Index)
        tms = lty - tbl
        dfy = BAA - AAA

    The other four variables are taken directly from
    the source data:

        tbl, b/m, ntis, svar
    """

    required = (
        "yyyymm",
        "Index",
        "D12",
        "E12",
        "b/m",
        "tbl",
        "AAA",
        "BAA",
        "lty",
        "ntis",
        "svar",
    )

    missing = [
        column
        for column in required
        if column not in raw.columns
    ]

    if missing:
        raise ValueError(
            "Missing required Goyal columns: "
            f"{missing}"
        )

    frame = raw.loc[
        :,
        required,
    ].copy()

    # ---------------------------------------------------------
    # Parse dates.
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

    # Treat each observation as available at month-end.
    dates = (
        dates
        + pd.offsets.MonthEnd(0)
    )

    # ---------------------------------------------------------
    # Parse numeric columns.
    # ---------------------------------------------------------
    numeric_columns = (
        "Index",
        "D12",
        "E12",
        "b/m",
        "tbl",
        "AAA",
        "BAA",
        "lty",
        "ntis",
        "svar",
    )

    for column in numeric_columns:
        frame[column] = _to_numeric(
            frame[column]
        )

    frame["date"] = dates

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

    # Historical Goyal files contain early periods before all
    # predictors become available. Ignore that initial warm-up,
    # but once the complete sample begins, do not silently skip
    # internal missing observations.
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
    # Validate variables entering logarithms.
    # ---------------------------------------------------------
    for column in (
        "Index",
        "D12",
        "E12",
    ):
        if (
            frame[column]
            <= 0.0
        ).any():
            raise ValueError(
                f"{column} must be positive "
                "for log-ratio construction"
            )

    # ---------------------------------------------------------
    # Welch-Goyal transformations used by Diffolio.
    # ---------------------------------------------------------
    tbl = frame[
        "tbl"
    ].to_numpy(
        dtype=np.float64
    )

    dp = (
        np.log(
            frame[
                "D12"
            ].to_numpy(
                dtype=np.float64
            )
        )
        - np.log(
            frame[
                "Index"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )

    ep = (
        np.log(
            frame[
                "E12"
            ].to_numpy(
                dtype=np.float64
            )
        )
        - np.log(
            frame[
                "Index"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )

    bm = frame[
        "b/m"
    ].to_numpy(
        dtype=np.float64
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