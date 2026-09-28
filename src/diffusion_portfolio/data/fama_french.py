"""Daily Fama/French factor data utilities.

For the MVP we only require the daily risk-free return (RF), which is used
to convert the 12 industry portfolio returns into excess returns.

The source is the official Kenneth R. French Data Library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd


FF3_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)

DEFAULT_FF3_DAILY_CACHE = Path(
    "data/raw/F-F_Research_Data_Factors_daily_CSV.zip"
)


@dataclass(frozen=True)
class RiskFreeSeries:
    """Chronological daily risk-free returns.

    Returns are stored in decimal form.

    Example
    -------
    0.01 percent -> 0.0001 decimal return.
    """

    dates: pd.DatetimeIndex
    returns: np.ndarray

    def __post_init__(self) -> None:
        if self.returns.ndim != 1:
            raise ValueError("risk-free returns must have shape [time]")

        if len(self.dates) != len(self.returns):
            raise ValueError(
                "dates and risk-free returns must have the same length"
            )

        if not self.dates.is_monotonic_increasing:
            raise ValueError(
                "risk-free dates must be chronological"
            )

        if self.dates.has_duplicates:
            raise ValueError(
                "risk-free dates must not contain duplicates"
            )

        if not np.isfinite(self.returns).all():
            raise ValueError(
                "risk-free returns contain NaN or infinite values"
            )


def download_ff3_daily(
    destination: str | Path = DEFAULT_FF3_DAILY_CACHE,
    *,
    overwrite: bool = False,
) -> Path:
    """Download the official daily Fama/French factor archive."""

    destination = Path(destination)

    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = Request(
        FF3_DAILY_URL,
        headers={
            "User-Agent": "diffusion-portfolio-research/0.1"
        },
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:
            temporary_path.write_bytes(
                response.read()
            )

        if not zipfile.is_zipfile(temporary_path):
            raise RuntimeError(
                "Fama/French factor download did not "
                "produce a valid ZIP archive."
            )

        temporary_path.replace(destination)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return destination


def _find_factor_header(
    lines: list[str],
) -> int:
    """Find the Mkt-RF, SMB, HML, RF header row."""

    for idx, line in enumerate(lines):
        cells = [
            cell.strip()
            for cell in line.split(",")
        ]

        if len(cells) >= 5 and tuple(cells[1:5]) == (
            "Mkt-RF",
            "SMB",
            "HML",
            "RF",
        ):
            return idx

    raise ValueError(
        "Could not find the daily Fama/French factor header."
    )


def parse_ff3_daily_text(
    text: str,
) -> RiskFreeSeries:
    """Extract the daily RF column from the official factor CSV."""

    lines = text.splitlines()

    header_idx = _find_factor_header(lines)

    dates: list[pd.Timestamp] = []
    risk_free: list[float] = []

    data_started = False

    for line in lines[header_idx + 1 :]:
        if not line.strip():
            if data_started:
                break
            continue

        cells = [
            cell.strip()
            for cell in line.split(",")
        ]

        if not cells:
            continue

        date_text = cells[0]

        if not (
            len(date_text) == 8
            and date_text.isdigit()
        ):
            if data_started:
                break
            continue

        if len(cells) < 5:
            raise ValueError(
                f"Malformed Fama/French factor row "
                f"for date {date_text}"
            )

        try:
            date = pd.to_datetime(
                date_text,
                format="%Y%m%d",
            )

            rf_percent = float(cells[4])

        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Could not parse Fama/French factor "
                f"row for {date_text}"
            ) from exc

        if rf_percent <= -99.0:
            raise ValueError(
                "Fama/French missing-value sentinel "
                f"detected for RF on {date_text}"
            )

        dates.append(date)

        # Official factor file reports returns in percent.
        risk_free.append(
            rf_percent / 100.0
        )

        data_started = True

    if not risk_free:
        raise ValueError(
            "No daily Fama/French factor observations found."
        )

    return RiskFreeSeries(
        dates=pd.DatetimeIndex(dates),
        returns=np.asarray(
            risk_free,
            dtype=np.float32,
        ),
    )


def read_ff3_daily_zip(
    path: str | Path,
) -> RiskFreeSeries:
    """Read the daily Fama/French factor ZIP archive."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"Not a valid ZIP archive: {path}"
        )

    with zipfile.ZipFile(path) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                "Expected exactly one CSV inside "
                f"factor archive, found {csv_files}"
            )

        raw = archive.read(csv_files[0])

    text = raw.decode("latin-1")

    return parse_ff3_daily_text(text)


def load_daily_risk_free(
    path: str | Path = DEFAULT_FF3_DAILY_CACHE,
    *,
    download_if_missing: bool = True,
) -> RiskFreeSeries:
    """Load the official daily Fama/French risk-free rate."""

    path = Path(path)

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(path)

        download_ff3_daily(path)

    return read_ff3_daily_zip(path)