"""Observation-based provenance for controlled synthetic forecast comparisons."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from diffusion_portfolio.data.core import slice_return_table
from diffusion_portfolio.data.registry import load_return_table


COMPARISON_DATASETS = {"synthetic_gaussian", "synthetic_mean_reverting"}
ARITHMETIC_UNITS = "arithmetic decimal asset returns"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_data_identity(cfg, scaler):
    """Reconstruct observable train/validation data and verify saved train moments.

    The panel hash exactly follows Stage 4E-1. This never reads hidden states or
    test observations, refits the scaler, or mutates any random-number stream.
    """
    if cfg.data.dataset not in COMPARISON_DATASETS:
        raise ValueError("Synthetic comparison currently supports Gaussian and mean-reverting sources only")
    if cfg.data.subtract_risk_free or cfg.data.horizon != 1:
        raise ValueError("Synthetic comparison requires arithmetic asset returns and horizon=1")
    table = slice_return_table(load_return_table(cfg.data.dataset, **cfg.data.source_options),
                               cfg.data.sample_start, cfg.data.val_end)
    training = table.returns[table.dates <= pd.Timestamp(cfg.data.train_end)].astype(np.float64)
    if table.columns != scaler.columns:
        raise ValueError("Saved scaler asset order differs from the reconstructed source")
    if (not np.array_equal(training.mean(0), scaler.mean)
            or not np.array_equal(training.std(0), scaler.std)):
        raise ValueError("Saved scaler does not match train-only moments of the reconstructed source")
    panel_hash = hashlib.sha256(table.dates.values.astype("datetime64[ns]").tobytes()
                                + np.ascontiguousarray(table.returns).tobytes()).hexdigest()
    return {
        "data_config": asdict(cfg.data), "columns": list(table.columns), "return_units": ARITHMETIC_UNITS,
        "source_identity": {
            "dataset": cfg.data.dataset, "source_options": cfg.data.source_options,
            "train_validation_panel_sha256": panel_hash,
            "simulator_source_sha256": file_sha256(Path(__file__).parent / "sources/synthetic_market.py"),
        },
        "training": {"n_observations": len(training), "start": str(table.dates[0].date()),
                     "end": str(table.dates[len(training) - 1].date())},
    }
