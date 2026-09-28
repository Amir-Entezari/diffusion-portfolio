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
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

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

KF12_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/12_Industry_Portfolios_daily_CSV.zip"
)

DEFAULT_KF12_CACHE = Path(
    "data/raw/12_Industry_Portfolios_daily_CSV.zip"
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
    
    
def download_kf12_daily(
    destination: str | Path = DEFAULT_KF12_CACHE,
    *,
    overwrite: bool = False,
) -> Path:
    """Download the official daily KF12 CSV archive.

    The downloaded ZIP is cached locally under ``data/raw`` by default.
    Repeated calls reuse the existing file unless ``overwrite=True``.
    """

    destination = Path(destination)

    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    request = Request(
        KF12_DAILY_URL,
        headers={
            "User-Agent": "diffusion-portfolio-research/0.1"
        },
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        with urlopen(request, timeout=30) as response:
            temporary_path.write_bytes(response.read())

        # Validate that the server actually returned a ZIP archive
        # before replacing the cache file.
        if not zipfile.is_zipfile(temporary_path):
            raise RuntimeError(
                "Kenneth French download did not produce a valid ZIP file."
            )

        temporary_path.replace(destination)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return destination


def _find_value_weighted_header(lines: list[str]) -> int:
    """Locate the header of the daily value-weighted return table.

    The Kenneth French archive contains explanatory text and can contain
    multiple tables. We therefore locate the section semantically instead
    of relying on a fixed number of skipped rows.

    Returns
    -------
    int
        Index of the CSV header row.
    """

    section_found = False

    for idx, line in enumerate(lines):
        cleaned = line.strip().lower()

        if "average value weighted returns" in cleaned:
            section_found = True
            continue

        if not section_found:
            continue

        cells = [cell.strip() for cell in line.split(",")]

        # First field is the date-column placeholder, followed by
        # the twelve industry names.
        if len(cells) >= 13 and tuple(cells[1:13]) == INDUSTRY_COLUMNS:
            return idx

    raise ValueError(
        "Could not find the value-weighted daily return table "
        "inside the Kenneth French file."
    )


def parse_kf12_daily_text(text: str) -> ReturnTable:
    """Parse the official Kenneth French daily CSV text.

    Only the first/value-weighted return table is extracted.

    Parameters
    ----------
    text:
        Decoded contents of the CSV inside the official ZIP archive.

    Returns
    -------
    ReturnTable
        Chronological daily decimal returns with shape [T, 12].
    """

    lines = text.splitlines()

    header_idx = _find_value_weighted_header(lines)

    dates: list[pd.Timestamp] = []
    rows: list[list[float]] = []

    data_started = False

    for line in lines[header_idx + 1 :]:
        if not line.strip():
            if data_started:
                break
            continue

        cells = [cell.strip() for cell in line.split(",")]

        if not cells:
            continue

        date_text = cells[0]

        # Daily observations use YYYYMMDD.
        if not (len(date_text) == 8 and date_text.isdigit()):
            if data_started:
                break
            continue

        if len(cells) < 13:
            raise ValueError(
                f"Malformed KF12 row for date {date_text}: "
                f"expected 12 returns, got {len(cells) - 1}"
            )

        try:
            date = pd.to_datetime(
                date_text,
                format="%Y%m%d",
            )

            values = [
                float(value)
                for value in cells[1:13]
            ]

        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Could not parse KF12 row for {date_text}"
            ) from exc

        dates.append(date)
        rows.append(values)
        data_started = True

    if not rows:
        raise ValueError(
            "No daily observations were found in the KF12 file."
        )

    frame = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(dates),
        columns=INDUSTRY_COLUMNS,
        dtype=np.float64,
    )

    return dataframe_to_return_table(frame)


def read_kf12_daily_zip(path: str | Path) -> ReturnTable:
    """Read a downloaded KF12 ZIP archive."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid ZIP archive: {path}")

    with zipfile.ZipFile(path) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                "Expected exactly one CSV inside KF12 archive, "
                f"found {csv_files}"
            )

        raw = archive.read(csv_files[0])

    # The French files are simple ASCII-compatible text, but latin-1
    # avoids failing on occasional metadata characters.
    text = raw.decode("latin-1")

    return parse_kf12_daily_text(text)


def load_kf12_daily(
    path: str | Path = DEFAULT_KF12_CACHE,
    *,
    download_if_missing: bool = True,
) -> ReturnTable:
    """Load the official daily value-weighted KF12 returns.

    Downloads and caches the official archive if necessary.
    """

    path = Path(path)

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(path)

        download_kf12_daily(path)

    return read_kf12_daily_zip(path)