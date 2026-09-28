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
    slice_return_table
)

from diffusion_portfolio.data.fama_french import (
    DEFAULT_FF3_DAILY_CACHE,
    FF3_DAILY_URL,
    RiskFreeSeries,
    download_ff3_daily,
    load_daily_risk_free,
    parse_ff3_daily_text,
    read_ff3_daily_zip,
)

from diffusion_portfolio.data.returns import (
    to_excess_returns,
)

from diffusion_portfolio.data.windows import (
    WindowedReturns,
    make_windows,
)

from diffusion_portfolio.data.splits import (
    WindowSplits,
    split_windows_by_date,
    split_windows_chronologically,
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
    "WindowSplits",
    "split_windows_chronologically",
    "slice_return_table",
    "split_windows_by_date",
    "DEFAULT_FF3_DAILY_CACHE",
    "FF3_DAILY_URL",
    "RiskFreeSeries",
    "download_ff3_daily",
    "load_daily_risk_free",
    "parse_ff3_daily_text",
    "read_ff3_daily_zip",
    "to_excess_returns",
]