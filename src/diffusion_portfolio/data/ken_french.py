"""Compatibility imports for archived KF12 notebooks. No implementation lives here."""
from diffusion_portfolio.data.core import ReturnTable, slice_return_table
from diffusion_portfolio.data.sources.kenneth_french import (
    DEFAULT_KF12_CACHE, INDUSTRY_COLUMNS, KF12_DAILY_URL,
    dataframe_to_return_table, download_kf12_daily, load_kf12_daily,
    parse_kf12_daily_text, read_kf12_daily_zip,
)
