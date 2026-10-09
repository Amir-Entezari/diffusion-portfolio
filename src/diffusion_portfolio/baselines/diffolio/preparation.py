"""Shared Diffolio source alignment, preprocessing and window construction."""
from pathlib import Path

import pandas as pd

from diffusion_portfolio.data import (
    CovariateStandardizer, TrainStandardizer, align_systematic_covariates_to_dates,
    build_asset_characteristics, build_monthly_systematic_covariates,
    load_daily_factors, load_daily_risk_free, load_return_table,
    slice_return_table, to_excess_returns,
)
from diffusion_portfolio.baselines.diffolio.dataset import (
    make_diffolio_windows, split_diffolio_windows_by_date,
)


def prepare_diffolio_data(data_cfg, *, return_scaler=None, covariate_scaler=None):
    """Return window splits, raw returns and the fitted/supplied scalers.

    Characteristics retain their full pre-sample warmup. Fitting is performed
    only for training runs; evaluation must supply both saved scalers.
    """
    assets = load_return_table(data_cfg["dataset"])
    factors = load_daily_factors()
    rf = load_daily_risk_free()
    excess_full = to_excess_returns(assets, rf)
    raw_returns = slice_return_table(
        excess_full, data_cfg["sample_start"], data_cfg["sample_end"]
    )
    asset_covariates = build_asset_characteristics(
        excess_full, factors, start=data_cfg["sample_start"], end=data_cfg["sample_end"]
    )
    raw_goyal = pd.read_excel(Path(data_cfg["goyal_path"]), sheet_name="Monthly", engine="openpyxl")
    systematic_covariates = align_systematic_covariates_to_dates(
        build_monthly_systematic_covariates(raw_goyal), raw_returns.dates
    )
    if not (raw_returns.dates.equals(asset_covariates.dates)
            and raw_returns.dates.equals(systematic_covariates.dates)):
        raise RuntimeError("Data modalities are not exactly date-aligned")
    if (return_scaler is None) != (covariate_scaler is None):
        raise ValueError("Supply both saved Diffolio scalers or neither")
    if return_scaler is None:
        scaling = data_cfg.get("return_scaling", "train_zscore")
        if scaling == "train_zscore":
            return_scaler = TrainStandardizer.fit(raw_returns, train_end=data_cfg["train_end"])
        elif scaling == "none":
            return_scaler = TrainStandardizer.identity(raw_returns.columns)
        else:
            raise ValueError(f"Unsupported data.return_scaling: {scaling}")
        if data_cfg.get("covariate_scaling", "train_zscore") != "train_zscore":
            raise ValueError("Diffolio requires train_zscore covariate scaling")
        covariate_scaler = CovariateStandardizer.fit(
            asset_covariates, systematic_covariates, train_end=data_cfg["train_end"]
        )
    windows = make_diffolio_windows(
        return_scaler.transform(raw_returns), raw_returns,
        covariate_scaler.transform_asset(asset_covariates),
        covariate_scaler.transform_systematic(systematic_covariates),
        lookback=int(data_cfg["lookback"]),
    )
    splits = split_diffolio_windows_by_date(
        windows, train_end=data_cfg["train_end"], val_end=data_cfg["val_end"],
        test_end=data_cfg["sample_end"],
    )
    return splits, raw_returns, return_scaler, covariate_scaler
