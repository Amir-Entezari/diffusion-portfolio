"""Small, explicit configuration surface for the scientific MVP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "ken_french_12"
    frequency: str = "daily"

    sample_start: str = "1958-01-01"
    sample_end: str = "2023-12-31"

    train_end: str = "1999-12-31"
    val_end: str = "2004-12-31"

    lookback: int = 60
    horizon: int = 1

    return_scaling: str = "train_zscore"


@dataclass(frozen=True)
class ModelConfig:
    type: str = "conditional_diffusion"

    condition_dim: int = 128
    history_hidden_dim: int = 256
    history_encoder: str = "mlp"

    cde_hidden_dim: int = 128
    cde_drift_hidden_dim: int = 256
    cde_sensitivity_hidden_dim: int = 256

    cde_solver: str = "dopri5"
    cde_rtol: float = 1e-4
    cde_atol: float = 1e-5
    cde_use_adjoint: bool = True
    diffusion_steps: int = 100
    schedule: str = "cosine"
    prediction_type: str = "v_prediction"

    time_embed_dim: int = 128
    channels: tuple[int, ...] = (
        32,
        64,
        128,
    )

    n_res_blocks: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5

    gradient_clip_norm: float = 1.0
    validation_seed: int = 12345


@dataclass(frozen=True)
class PortfolioConfig:
    optimizer: str = "mean_variance"
    allow_short: bool = False
    transaction_cost_bps: float = 10.0
    annualization_factor: int = 252


@dataclass(frozen=True)
class EvaluationConfig:
    n_scenarios: int = 256
    report_distribution_metrics: bool = True
    report_portfolio_metrics: bool = True


@dataclass(frozen=True)
class MVPConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    portfolio: PortfolioConfig
    evaluation: EvaluationConfig


def _section(cls: type, raw: dict[str, Any], key: str):
    values = raw.get(key, {})
    if not isinstance(values, dict):
        raise TypeError(f"Config section '{key}' must be a mapping")
    if cls is ModelConfig and "channels" in values:
        values = dict(values)
        values["channels"] = tuple(values["channels"])
    return cls(**values)


def load_config(path: str | Path) -> MVPConfig:
    """Load one experiment config and validate the small MVP contract."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = MVPConfig(
        seed=int(raw.get("seed", 42)),
        data=_section(DataConfig, raw, "data"),
        model=_section(ModelConfig, raw, "model"),
        training=_section(TrainingConfig, raw, "training"),
        portfolio=_section(PortfolioConfig, raw, "portfolio"),
        evaluation=_section(EvaluationConfig, raw, "evaluation"),
    )
    sample_start = pd.Timestamp(cfg.data.sample_start)
    sample_end = pd.Timestamp(cfg.data.sample_end)
    train_end = pd.Timestamp(cfg.data.train_end)
    val_end = pd.Timestamp(cfg.data.val_end)

    if not (
        sample_start
        < train_end
        < val_end
        < sample_end
    ):
        raise ValueError(
            "Expected chronological ordering: "
            "sample_start < train_end < val_end < sample_end"
        )
    if cfg.data.lookback <= 0 or cfg.data.horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    if cfg.data.return_scaling not in {
        "none",
        "train_zscore",
    }:
        raise ValueError(
            "data.return_scaling must be one of: "
            "'none', 'train_zscore'"
        )
    if cfg.model.diffusion_steps <= 1:
        raise ValueError("diffusion_steps must be > 1")
    if cfg.model.condition_dim <= 0:
        raise ValueError("condition_dim must be positive")
    if not cfg.model.channels:
        raise ValueError("model.channels cannot be empty")
    if cfg.model.history_hidden_dim <= 0:
        raise ValueError(
            "history_hidden_dim must be positive"
        )
    if cfg.model.history_encoder not in {
        "mlp",
        "cde",
    }:
        raise ValueError(
            "model.history_encoder must "
            "be 'mlp' or 'cde'"
        )

    if cfg.model.cde_hidden_dim <= 0:
        raise ValueError(
            "cde_hidden_dim must be positive"
        )

    if (
        cfg.model.cde_drift_hidden_dim
        <= 0
    ):
        raise ValueError(
            "cde_drift_hidden_dim "
            "must be positive"
        )

    if (
        cfg.model
        .cde_sensitivity_hidden_dim
        <= 0
    ):
        raise ValueError(
            "cde_sensitivity_hidden_dim "
            "must be positive"
        )

    if cfg.model.cde_rtol <= 0:
        raise ValueError(
            "cde_rtol must be positive"
        )

    if cfg.model.cde_atol <= 0:
        raise ValueError(
            "cde_atol must be positive"
        )
    if cfg.model.time_embed_dim <= 0:
        raise ValueError(
            "time_embed_dim must be positive"
        )

    if cfg.model.n_res_blocks <= 0:
        raise ValueError(
            "n_res_blocks must be positive"
        )

    if cfg.model.prediction_type not in {
        "epsilon",
        "v_prediction",
    }:
        raise ValueError(
            "prediction_type must be one of: "
            "'epsilon', 'v_prediction'"
        )
    if cfg.training.gradient_clip_norm <= 0:
        raise ValueError(
            "gradient_clip_norm must be positive"
        )

    if cfg.training.validation_seed < 0:
        raise ValueError(
            "validation_seed cannot be negative"
        )
    return cfg
