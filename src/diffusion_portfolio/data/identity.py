"""Observation-based provenance for controlled synthetic forecast comparisons."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diffusion_portfolio.models.encoders.history_mask import history_mask_policy

from diffusion_portfolio.data.core import slice_return_table
from diffusion_portfolio.data.registry import load_return_table


COMPARISON_DATASETS = {"synthetic_gaussian", "synthetic_mean_reverting"}
ARITHMETIC_UNITS = "arithmetic decimal asset returns"


def synthetic_training_contract(cfg, *, history_mask=None):
    """Freeze the settings that define a Vanilla synthetic training experiment.

    Runtime device/resume/stop-after-epoch options are intentionally excluded:
    a two-epoch chunk must retain the original configured training budget.
    """
    contract = json.loads(json.dumps({
        "seed": cfg.seed, "data": asdict(cfg.data), "model": asdict(cfg.model),
        "optimizer": "torch.optim.AdamW", "training": asdict(cfg.training),
        "evaluation": asdict(cfg.evaluation),
    }))
    if history_mask is not None:
        if history_mask != history_mask_policy(cfg, history_mask.get("keep_last")):
            raise ValueError("Invalid history-mask policy")
        contract["history_mask"] = history_mask
    return contract


def require_training_contract(recorded, cfg):
    contract = recorded.get("training_contract")
    if contract is None:
        raise ValueError("Missing immutable training contract; legacy provenance cannot be retrospectively certified")
    if contract != synthetic_training_contract(cfg, history_mask=contract.get("history_mask")):
        raise ValueError("Immutable training contract differs from the requested/saved configuration; use the original configuration or a new run directory")
    return contract


def saved_history_mask(run_dir, cfg):
    """Restore the original mask; a small marker catches missing policy evidence.

    Only ablation runs create experiment.json with history_mask. Legacy runs
    with neither marker nor policy keep their original unmasked meaning.
    Neither evidence file is repaired or retrospectively manufactured here.
    """
    run_dir = Path(run_dir)
    identity_path, marker_path = run_dir / "data_identity.json", run_dir / "experiment.json"
    recorded = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
    marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
    contract = recorded.get("training_contract", {})
    policy = contract.get("history_mask")
    if policy is not None or "history_mask" in marker:
        require_training_contract(recorded, cfg)
        if policy is None or marker.get("history_mask") != policy:
            raise ValueError("History-mask policy differs or is missing from the original training contract/experiment marker")
    return policy


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
