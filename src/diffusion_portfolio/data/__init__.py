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
    slice_return_table,
)

from diffusion_portfolio.data.fama_french import (
    DEFAULT_FF3_DAILY_CACHE,
    FACTOR_COLUMNS,
    FF3_DAILY_URL,
    FactorTable,
    RiskFreeSeries,
    download_ff3_daily,
    load_daily_factors,
    load_daily_risk_free,
    parse_ff3_daily_text,
    parse_ff3_factor_text,
    read_ff3_daily_zip,
    read_ff3_factor_zip,
)

from diffusion_portfolio.data.dataset import (
    DatasetSplits,
    LoaderSplits,
    ReturnWindowDataset,
    build_window_datasets,
    collate_return_batch,
    make_dataloaders,
)

from diffusion_portfolio.data.asset_characteristics import (
    CHARACTERISTIC_COLUMNS,
    AssetCharacteristicTable,
    build_asset_characteristics,
)

from diffusion_portfolio.data.normalization import (
    TrainStandardizer,
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
from diffusion_portfolio.data.systematic_covariates import (
    SYSTEMATIC_COLUMNS,
    SystematicCovariateTable,
    align_systematic_covariates_to_dates,
    build_monthly_systematic_covariates,
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
    "TrainStandardizer",
    "DatasetSplits",
    "LoaderSplits",
    "ReturnWindowDataset",
    "build_window_datasets",
    "collate_return_batch",
    "make_dataloaders",
    "FACTOR_COLUMNS",
    "FactorTable",
    "load_daily_factors",
    "parse_ff3_factor_text",
    "read_ff3_factor_zip",
    "CHARACTERISTIC_COLUMNS",
    "AssetCharacteristicTable",
    "build_asset_characteristics",
    "SYSTEMATIC_COLUMNS",
    "SystematicCovariateTable",
    "align_systematic_covariates_to_dates",
    "build_monthly_systematic_covariates",
]