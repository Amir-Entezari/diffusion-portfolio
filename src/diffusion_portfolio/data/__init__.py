"""Data utilities for the diffusion-portfolio research MVP."""

from diffusion_portfolio.data.ken_french import (
    DEFAULT_KF12_CACHE,
    INDUSTRY_COLUMNS,
    KF12_DAILY_URL,
    ReturnTable,
    dataframe_to_return_table,
    download_kf12_daily,
    load_kf12_daily,
    parse_kf12_daily_text,
    read_kf12_daily_zip,
)
from diffusion_portfolio.data.windows import (
    WindowedReturns,
    make_windows,
)

__all__ = [
    "DEFAULT_KF12_CACHE",
    "INDUSTRY_COLUMNS",
    "KF12_DAILY_URL",
    "ReturnTable",
    "WindowedReturns",
    "dataframe_to_return_table",
    "download_kf12_daily",
    "load_kf12_daily",
    "make_windows",
    "parse_kf12_daily_text",
    "read_kf12_daily_zip",
]