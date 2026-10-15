"""Daily value-weighted FF49 and FF100 Size × Book-to-Market return panels.

Only the explicitly labelled daily value-weighted section is accepted. Source
percentages become arithmetic decimal returns; no risk-free adjustment occurs
here. Missing sentinels fail by default. ``missing_policy='drop'`` explicitly
excludes whole dates across all assets and reports the effect with a warning.
The validated KF12 parser remains separate and unchanged.
"""
from __future__ import annotations

import csv
from pathlib import Path
import warnings
import zipfile

import numpy as np
import pandas as pd

from diffusion_portfolio.data.core import ReturnTable
from diffusion_portfolio.data.sources.kenneth_french import _download_daily_zip


KF49_COLUMNS = (
    "Agric", "Food", "Soda", "Beer", "Smoke", "Toys", "Fun", "Books",
    "Hshld", "Clths", "Hlth", "MedEq", "Drugs", "Chems", "Rubbr", "Txtls",
    "BldMt", "Cnstr", "Steel", "FabPr", "Mach", "ElcEq", "Autos", "Aero",
    "Ships", "Guns", "Gold", "Mines", "Coal", "Oil", "Util", "Telcm",
    "PerSv", "BusSv", "Hardw", "Softw", "Chips", "LabEq", "Paper", "Boxes",
    "Trans", "Whlsl", "Rtail", "Meals", "Banks", "Insur", "RlEst", "Fin", "Other",
)
# Preserve French's labels and size-major, book-to-market-minor ordering.
KF100_SIZE_BM_COLUMNS = tuple(
    "SMALL LoBM" if (size, bm) == (1, 1) else
    "SMALL HiBM" if (size, bm) == (1, 10) else
    "BIG LoBM" if (size, bm) == (10, 1) else
    "BIG HiBM" if (size, bm) == (10, 10) else
    f"ME{size} BM{bm}"
    for size in range(1, 11) for bm in range(1, 11)
)

_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
KF49_DAILY_URL = _BASE_URL + "49_Industry_Portfolios_daily_CSV.zip"
KF100_SIZE_BM_DAILY_URL = _BASE_URL + "100_Portfolios_10x10_Daily_CSV.zip"
DEFAULT_KF49_CACHE = Path("data/raw/49_Industry_Portfolios_daily_CSV.zip")
DEFAULT_KF100_SIZE_BM_CACHE = Path("data/raw/100_Portfolios_10x10_Daily_CSV.zip")


def parse_french_daily_text(
    text: str, *, columns: tuple[str, ...], missing_policy: str = "error",
) -> ReturnTable:
    """Parse one expected daily VW table; reject reordered/unexpected labels.

    ``drop`` removes dates containing -99.99 or -999 in any portfolio, never
    individual assets. All original dates are validated before exclusion.
    Malformed rows, NaNs and infinities always fail, regardless of policy.
    """
    if missing_policy not in ("error", "drop"):
        raise ValueError("missing_policy must be 'error' or 'drop'")
    lines = text.splitlines()
    section = next((i for i, line in enumerate(lines)
                    if "average value weighted returns" in line.lower()
                    and "daily" in line.lower()), None)
    if section is None:
        raise ValueError("No explicitly labelled daily value-weighted return section")
    header = next((i for i in range(section + 1, len(lines)) if lines[i].strip()), None)
    if header is None:
        raise ValueError("Daily value-weighted return section has no column header")
    fields = tuple(cell.strip() for cell in next(csv.reader([lines[header]])))
    if fields != ("", *columns):
        raise ValueError("Unexpected daily portfolio columns or column ordering")

    dates, rows = [], []
    for line in lines[header + 1:]:
        if not line.strip():
            if rows:
                break
            continue
        # A following French table can start without a separating blank line.
        if line.strip().lower().startswith("average "):
            break
        cells = [cell.strip() for cell in next(csv.reader([line]))]
        date_text = cells[0]
        if len(cells) != len(columns) + 1:
            raise ValueError(f"Malformed daily row {date_text}: expected {len(columns)} returns")
        if len(date_text) != 8 or not date_text.isdigit():
            raise ValueError(f"Malformed daily date: {date_text!r}; expected YYYYMMDD")
        try:
            dates.append(pd.to_datetime(date_text, format="%Y%m%d"))
            rows.append([float(value) for value in cells[1:]])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Could not parse daily row for {date_text}") from exc
    if not rows:
        raise ValueError("No daily observations in value-weighted return section")
    dates = pd.DatetimeIndex(dates)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Daily dates must be chronological and unique")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Daily returns contain NaN or infinite values")
    missing = np.isin(values, (-99.99, -999.0)).any(axis=1)
    if missing.any():
        excluded = dates[missing]
        message = (
            f"French missing-value sentinels in {missing.sum()}/{len(dates)} daily rows "
            f"({excluded[0].date()} to {excluded[-1].date()})"
        )
        if missing_policy == "error":
            raise ValueError(message + "; use missing_policy='drop' to exclude whole dates")
        if missing.all():
            raise ValueError(message + "; no complete daily rows remain")
        warnings.warn(
            message + f"; missing_policy='drop' excludes these dates; retained {(~missing).sum()} rows",
            UserWarning, stacklevel=2,
        )
        dates, values = dates[~missing], values[~missing]
    return ReturnTable(dates, (values / 100.0).astype(np.float32), columns)


def read_french_daily_zip(path, *, columns, missing_policy="error") -> ReturnTable:
    """Read the sole CSV in a cached official archive."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one CSV in French archive, found {names}")
        text = archive.read(names[0]).decode("latin-1")
    return parse_french_daily_text(text, columns=columns, missing_policy=missing_policy)


def _load_daily(path, *, url, columns, download_if_missing, missing_policy):
    path = Path(path)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(path)
        _download_daily_zip(url, path)
    return read_french_daily_zip(path, columns=columns, missing_policy=missing_policy)


def load_kf49_daily(
    path=DEFAULT_KF49_CACHE, *, download_if_missing=True, missing_policy="error",
) -> ReturnTable:
    """Load/cache official daily value-weighted 49 Industry Portfolios."""
    return _load_daily(
        path, url=KF49_DAILY_URL, columns=KF49_COLUMNS,
        download_if_missing=download_if_missing, missing_policy=missing_policy,
    )


def load_kf100_size_bm_daily(
    path=DEFAULT_KF100_SIZE_BM_CACHE, *, download_if_missing=True, missing_policy="error",
) -> ReturnTable:
    """Load/cache official daily VW 100 Size × Book-to-Market Portfolios."""
    return _load_daily(
        path, url=KF100_SIZE_BM_DAILY_URL, columns=KF100_SIZE_BM_COLUMNS,
        download_if_missing=download_if_missing, missing_policy=missing_policy,
    )
