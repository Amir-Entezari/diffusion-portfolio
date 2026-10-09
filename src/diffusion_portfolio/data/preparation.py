"""Canonical preparation of paired raw/model-space diffusion windows."""
from pathlib import Path

import numpy as np

from diffusion_portfolio.data.core import slice_return_table
from diffusion_portfolio.data.dataset import build_window_datasets
from diffusion_portfolio.data.normalization import TrainStandardizer
from diffusion_portfolio.data.registry import load_return_table
from diffusion_portfolio.data.returns import to_excess_returns
from diffusion_portfolio.data.sources.fama_french import load_daily_risk_free


def load_standardizer(path: str | Path) -> TrainStandardizer:
    with np.load(path, allow_pickle=False) as saved:
        return TrainStandardizer(
            mean=saved["mean"],
            std=saved["std"],
            columns=tuple(str(column) for column in saved["columns"]),
        )


def save_standardizer(path: str | Path, scaler: TrainStandardizer) -> None:
    np.savez(path, mean=scaler.mean, std=scaler.std, columns=np.asarray(scaler.columns))


def prepare_return_data(cfg, *, standardizer_path=None, scaler=None):
    """Return (datasets, scaler) without fitting any post-training observations.

    Existing diffusion experiments always fit train Z-scores when no saved
    scaler is supplied, including configs declaring return_scaling=none. Keep
    that historical convention here; Diffolio has its own explicit scaling.
    The source supplies decimal asset returns; RF subtraction stays separate.
    """
    data = cfg.data
    assets = slice_return_table(
        load_return_table(data.dataset), data.sample_start, data.sample_end
    )
    excess = to_excess_returns(assets, load_daily_risk_free())
    if scaler is not None and standardizer_path is not None:
        raise ValueError("Supply a scaler or its path, not both")
    if scaler is None:
        scaler = (
            TrainStandardizer.fit(excess, train_end=data.train_end)
            if standardizer_path is None
            else load_standardizer(standardizer_path)
        )
    datasets = build_window_datasets(
        scaler.transform(excess), excess,
        lookback=data.lookback, horizon=data.horizon,
        train_end=data.train_end, val_end=data.val_end, test_end=data.sample_end,
    )
    return datasets, scaler


def prepare_datasets(cfg, *, standardizer_path=None, scaler=None):
    """Prepare datasets with the saved-scaler contract used by ablations."""
    return prepare_return_data(
        cfg, standardizer_path=standardizer_path, scaler=scaler
    )[0]
