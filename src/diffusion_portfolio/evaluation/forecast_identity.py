"""Verify a saved Vanilla run for strict synthetic validation comparisons."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffusion_portfolio.data.identity import file_sha256, synthetic_data_identity
from diffusion_portfolio.utils.research_checks import check_checkpoint_pair


def verify_synthetic_checkpoint(run_dir, cfg, scaler, checkpoint):
    """Require training-time data evidence; distinguish partial/completed budgets."""
    run_dir = Path(run_dir)
    if cfg.model.history_encoder != "mlp" or cfg.model.type != "conditional_diffusion":
        raise ValueError("Synthetic comparison supports Vanilla MLP Gaussian diffusion only")
    path = run_dir / "data_identity.json"
    if not path.is_file():
        raise ValueError("Missing training-time data_identity.json; source provenance cannot be verified for this legacy run")
    identity = synthetic_data_identity(cfg, scaler)
    recorded = json.loads(path.read_text())
    checkpoint_hashes = recorded.pop("checkpoint_sha256", {})
    if recorded != identity:
        raise ValueError("Training source identity does not match the saved configuration/scaler and current source")
    for name in ("best.pt", "last.pt"):
        if (run_dir / name).exists() and checkpoint_hashes.get(name) != file_sha256(run_dir / name):
            raise ValueError(f"Checkpoint {name} is not bound to the recorded training source; resume a verified chunk or investigate artifact mismatch")
    best_epoch, best_loss = int(checkpoint["epoch"]), float(checkpoint["val_loss"])
    if not np.isfinite(best_loss) or not 1 <= best_epoch <= cfg.training.epochs:
        raise ValueError("Invalid best-checkpoint epoch or validation loss")
    if (run_dir / "last.pt").exists():
        last_epoch = check_checkpoint_pair(run_dir)
        last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        history = pd.DataFrame(last["history"])
    elif (run_dir / "history.csv").is_file():
        history = pd.read_csv(run_dir / "history.csv")
        last_epoch = int(history.epoch.iloc[-1])
    else:
        raise ValueError("Missing last.pt/history.csv; completed training budget cannot be verified")
    if (not best_epoch <= last_epoch <= cfg.training.epochs
            or history.epoch.tolist() != list(range(1, last_epoch + 1))
            or not np.isfinite(history[["train_loss", "val_loss"]].to_numpy()).all()):
        raise ValueError("Training history is incomplete or inconsistent with checkpoint epochs")
    best_row = history.loc[history.epoch == best_epoch, "val_loss"].iloc[0]
    if not np.isclose(best_row, best_loss, rtol=0, atol=1e-8) or not np.isclose(history.val_loss.min(), best_loss, rtol=0, atol=1e-8):
        raise ValueError("Best checkpoint does not match recorded validation history")
    summary_path = run_dir / "summary.json"
    completed = False
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if (summary["best_epoch"] != best_epoch
                or not np.isclose(summary["best_val_loss"], best_loss, rtol=0, atol=1e-8)
                or last_epoch != cfg.training.epochs):
            raise ValueError("Training summary conflicts with checkpoint/history or declared budget")
        completed = True
    training = {
        "status": "completed_declared_budget" if completed else "preliminary",
        "completed_epochs": last_epoch, "configured_epochs": cfg.training.epochs,
        "checkpoint_epoch": best_epoch, "checkpoint_val_loss": best_loss,
        "note": "Training budget completion is not evidence of forecast quality; validation selected this checkpoint.",
    }
    hashes = {name: file_sha256(run_dir / name) for name in
              ("config.yaml", "standardizer.npz", "best.pt", "data_identity.json", "last.pt", "summary.json", "history.csv")
              if (run_dir / name).is_file()}
    return {**identity, "training_run": training, "checkpoint_files_sha256": hashes}
