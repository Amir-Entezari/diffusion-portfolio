"""Small artifact checks for frozen-backbone research experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data.preparation import load_standardizer


def source_identity(run_dir):
    """Identify the exact saved configuration, scaler and best checkpoint."""
    result = {}
    for name in ("config.yaml", "standardizer.npz", "best.pt"):
        digest = hashlib.sha256()
        with (Path(run_dir) / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def check_checkpoint_pair(run_dir):
    """Allow a new run; reject incomplete/inconsistent resumable checkpoints."""
    run_dir = Path(run_dir)
    paths = [run_dir / name for name in ("best.pt", "last.pt")]
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"Incomplete checkpoint pair: {run_dir}; best.pt and last.pt are required")
    best, last = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    if (best["epoch"] != last["best_epoch"]
            or not abs(best["val_loss"] - last["best_val_loss"]) < 1e-8):
        raise RuntimeError(f"Inconsistent best/last checkpoint metadata: {run_dir}")
    return int(last["epoch"])


def require_completed_run(run_dir, expected_config=None, *, assets=None, validation=False):
    """Check a completed diffusion reference without sampling or refitting data.

    Frozen legacy references may omit last.pt; when present it must be consistent
    and at the configured final epoch. Resumable outputs use check_checkpoint_pair.
    """
    run_dir = Path(run_dir)
    for name in ("config.yaml", "standardizer.npz", "best.pt", "summary.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Complete the prerequisite run first: {run_dir / name}")
    cfg = load_config(run_dir / "config.yaml")
    if expected_config is not None and cfg != load_config(expected_config):
        raise RuntimeError(f"Prerequisite configuration does not match {expected_config}: {run_dir}")
    scaler = load_standardizer(run_dir / "standardizer.npz")
    if assets is not None and len(scaler.columns) != assets:
        raise RuntimeError(f"Wrong asset universe in {run_dir}: {len(scaler.columns)} != {assets}")
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = json.loads((run_dir / "summary.json").read_text())
    if (best["epoch"] != summary["best_epoch"]
            or not abs(best["val_loss"] - summary["best_val_loss"]) < 1e-8):
        raise RuntimeError(f"Best checkpoint does not match completed summary: {run_dir}")
    if (run_dir / "last.pt").exists() and check_checkpoint_pair(run_dir) != cfg.training.epochs:
        raise RuntimeError(f"Prerequisite training is incomplete: {run_dir}")
    if (run_dir / "history.csv").is_file():
        import pandas as pd
        epochs = pd.read_csv(run_dir / "history.csv")["epoch"].tolist()
        if epochs != list(range(1, cfg.training.epochs + 1)):
            raise RuntimeError(f"Prerequisite history is incomplete: {run_dir}")
    if validation:
        metrics = json.loads((run_dir / "validation_probabilistic_metrics.json").read_text())
        if (metrics.get("split") != "validation" or metrics["checkpoint_epoch"] != best["epoch"]
                or not abs(metrics["checkpoint_val_loss"] - best["val_loss"]) < 1e-8
                or metrics["n_scenarios"] != cfg.evaluation.n_scenarios
                or metrics["sampling_seed"] != cfg.training.validation_seed):
            raise RuntimeError(f"Validation metrics do not match the completed best checkpoint: {run_dir}")
        values = [metrics["crps_mean"], metrics["energy_score"],
                  *[row["ace"] for row in metrics["calibration"]]]
        if not metrics["calibration"] or not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite validation metrics: {run_dir}")
    return cfg
