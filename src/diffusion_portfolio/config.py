"""Small, explicit configuration surface for the scientific MVP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "ken_french_12"
    frequency: str = "daily"
    lookback: int = 60
    horizon: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    type: str = "conditional_diffusion"
    condition_dim: int = 128
    diffusion_steps: int = 100
    schedule: str = "cosine"
    channels: tuple[int, ...] = (32, 64, 128)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5


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

    if cfg.data.lookback <= 0 or cfg.data.horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    if not (0 < cfg.data.train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 <= cfg.data.val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1)")
    if cfg.data.train_ratio + cfg.data.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")
    if cfg.model.diffusion_steps <= 1:
        raise ValueError("diffusion_steps must be > 1")
    if cfg.model.condition_dim <= 0:
        raise ValueError("condition_dim must be positive")
    if not cfg.model.channels:
        raise ValueError("model.channels cannot be empty")

    return cfg
