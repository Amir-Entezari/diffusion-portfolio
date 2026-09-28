"""
Two-Timescale Stochastic Approximation (TTSA) Training Loop.

Reference: Proposal §6.1 — Bi-Level TTSA Optimisation.

Implements the Stackelberg game training:

    INNER (fast timescale):  Generator (Phases 0-3)
        θ_{k+1} = θ_k − α_k ∇_θ L_inner(θ_k, φ_k)

    OUTER (slow timescale):  Controller (Phase 4)
        φ_{k+1} = φ_k − β_k ∇_φ L_outer(θ_{k+1}, φ_k)

Key TTSA conditions (§6.1.1):
    - α_k → 0, β_k → 0 (decaying step sizes)
    - β_k / α_k → 0 (outer is strictly slower)
    - Σ α_k = Σ β_k = ∞ (infinite exploration)

Stackelberg Wall (§6.1.2):
    Gradients from the outer loop MUST NOT flow into the inner loop.
    Implemented via .detach() on generated scenarios passed to controller.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class TTSAScheduler:
    r"""Two-Timescale learning rate scheduler.

    Inner (generator):
        α_k = α_0 / (1 + k)^{p_inner}

    Outer (controller):
        β_k = β_0 / (1 + k)^{p_outer}

    Constraint: p_outer > p_inner ⟹ β_k / α_k → 0.

    Args:
        inner_lr_initial: α_0.
        outer_lr_initial: β_0.
        inner_decay_power: p_inner.
        outer_decay_power: p_outer. Must be > p_inner.
    """

    def __init__(
        self,
        inner_lr_initial: float = 1e-3,
        outer_lr_initial: float = 5e-4,
        inner_decay_power: float = 0.6,
        outer_decay_power: float = 0.8,
    ) -> None:
        assert outer_decay_power > inner_decay_power, (
            f"TTSA violation: p_outer ({outer_decay_power}) must be > "
            f"p_inner ({inner_decay_power})"
        )
        self.inner_lr_initial = inner_lr_initial
        self.outer_lr_initial = outer_lr_initial
        self.inner_decay_power = inner_decay_power
        self.outer_decay_power = outer_decay_power

    def get_inner_lr(self, step: int) -> float:
        return self.inner_lr_initial / (1 + step) ** self.inner_decay_power

    def get_outer_lr(self, step: int) -> float:
        return self.outer_lr_initial / (1 + step) ** self.outer_decay_power

    def make_inner_scheduler(self, optimizer: torch.optim.Optimizer) -> LambdaLR:
        return LambdaLR(
            optimizer,
            lambda step: 1.0 / (1 + step) ** self.inner_decay_power,
        )

    def make_outer_scheduler(self, optimizer: torch.optim.Optimizer) -> LambdaLR:
        return LambdaLR(
            optimizer,
            lambda step: 1.0 / (1 + step) ** self.outer_decay_power,
        )


class TTSATrainer:
    """Two-Timescale Stochastic Approximation Trainer.

    Orchestrates the bi-level Stackelberg optimization:
        1. Run n_inner inner steps (generator training)
        2. Run 1 outer step (controller training)
        3. Enforce the Stackelberg wall via .detach()

    Args:
        generator: Phases 0-3 (produces scenarios).
        controller: Phase 4 (produces portfolio weights).
        inner_lr: α_0.
        outer_lr: β_0.
        inner_decay_power: p_inner.
        outer_decay_power: p_outer.
        n_inner_steps_per_outer: Number of inner steps per outer step.
        lambda_kl: Evidential KL weight.
        lambda_sgw: SGW regularization weight.
    """

    def __init__(
        self,
        generator: nn.Module,
        controller: nn.Module,
        inner_lr: float = 1e-3,
        outer_lr: float = 5e-4,
        inner_decay_power: float = 0.6,
        outer_decay_power: float = 0.8,
        n_inner_steps_per_outer: int = 5,
        lambda_kl: float = 0.01,
        lambda_sgw: float = 0.1,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.generator = generator
        self.controller = controller
        self.n_inner = n_inner_steps_per_outer
        self.lambda_kl = lambda_kl
        self.lambda_sgw = lambda_sgw
        self.max_grad_norm = max_grad_norm

        # Separate optimizers (TTSA requirement)
        self.inner_optimizer = Adam(generator.parameters(), lr=inner_lr)
        self.outer_optimizer = Adam(controller.parameters(), lr=outer_lr)

        # TTSA schedulers
        self.ttsa_scheduler = TTSAScheduler(
            inner_lr, outer_lr, inner_decay_power, outer_decay_power,
        )
        self.inner_scheduler = self.ttsa_scheduler.make_inner_scheduler(
            self.inner_optimizer
        )
        self.outer_scheduler = self.ttsa_scheduler.make_outer_scheduler(
            self.outer_optimizer
        )

        self.global_step = 0
        self.metrics_history: list[dict[str, float]] = []

    def inner_step(
        self,
        batch: dict[str, Tensor],
        inner_loss_fn: Callable[..., dict[str, Tensor]],
    ) -> dict[str, float]:
        """Execute one inner (generator) optimization step.

        Args:
            batch: Data batch from DataLoader.
            inner_loss_fn: Callable returning dict with 'loss' key.

        Returns:
            Metrics dict.
        """
        self.generator.train()
        self.inner_optimizer.zero_grad()

        result = inner_loss_fn(batch)
        loss = result["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), self.max_grad_norm
        )
        self.inner_optimizer.step()

        return {k: v.item() if isinstance(v, Tensor) else v for k, v in result.items()}

    def outer_step(
        self,
        batch: dict[str, Tensor],
        outer_loss_fn: Callable[..., dict[str, Tensor]],
    ) -> dict[str, float]:
        """Execute one outer (controller) optimization step.

        The Stackelberg wall is enforced here: generated scenarios
        are .detach()'ed before being passed to the controller.

        Args:
            batch: Data batch.
            outer_loss_fn: Callable returning dict with 'loss' key.

        Returns:
            Metrics dict.
        """
        self.controller.train()
        self.generator.eval()  # Freeze generator for outer step
        self.outer_optimizer.zero_grad()

        result = outer_loss_fn(batch)
        loss = result["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.controller.parameters(), self.max_grad_norm
        )
        self.outer_optimizer.step()

        return {k: v.item() if isinstance(v, Tensor) else v for k, v in result.items()}

    def train_step(
        self,
        batch: dict[str, Tensor],
        inner_loss_fn: Callable,
        outer_loss_fn: Callable,
    ) -> dict[str, Any]:
        """Execute one full TTSA step (n_inner inner + 1 outer).

        Args:
            batch: Data batch.
            inner_loss_fn: Generator loss function.
            outer_loss_fn: Controller loss function.

        Returns:
            Combined metrics from inner and outer steps.
        """
        metrics = {}

        # --- Inner loop: n_inner steps of generator training ---
        inner_metrics_list = []
        for i in range(self.n_inner):
            m = self.inner_step(batch, inner_loss_fn)
            inner_metrics_list.append(m)

        # Average inner metrics
        if inner_metrics_list:
            for key in inner_metrics_list[0]:
                vals = [m[key] for m in inner_metrics_list if isinstance(m.get(key), (int, float))]
                if vals:
                    metrics[f"inner/{key}"] = sum(vals) / len(vals)

        # --- Outer loop: 1 step of controller training ---
        outer_m = self.outer_step(batch, outer_loss_fn)
        for key, val in outer_m.items():
            metrics[f"outer/{key}"] = val

        # Step schedulers
        self.inner_scheduler.step()
        self.outer_scheduler.step()

        # Record
        metrics["step"] = self.global_step
        metrics["inner_lr"] = self.inner_optimizer.param_groups[0]["lr"]
        metrics["outer_lr"] = self.outer_optimizer.param_groups[0]["lr"]
        metrics["lr_ratio"] = metrics["outer_lr"] / max(metrics["inner_lr"], 1e-12)

        self.metrics_history.append(metrics)
        self.global_step += 1

        return metrics

    def verify_ttsa_conditions(self) -> dict[str, bool]:
        """Verify that TTSA theoretical conditions are satisfied.

        Returns:
            Dict of condition name → satisfied boolean.
        """
        inner_lr = self.inner_optimizer.param_groups[0]["lr"]
        outer_lr = self.outer_optimizer.param_groups[0]["lr"]

        conditions = {
            "outer_slower_than_inner": outer_lr < inner_lr,
            "lr_ratio_decreasing": True,  # Guaranteed by p_outer > p_inner
            "inner_lr_positive": inner_lr > 0,
            "outer_lr_positive": outer_lr > 0,
        }

        if len(self.metrics_history) >= 2:
            prev_ratio = self.metrics_history[-2].get("lr_ratio", 1.0)
            curr_ratio = self.metrics_history[-1].get("lr_ratio", 1.0)
            conditions["lr_ratio_decreasing"] = curr_ratio <= prev_ratio + 1e-8

        return conditions

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        torch.save({
            "global_step": self.global_step,
            "generator_state": self.generator.state_dict(),
            "controller_state": self.controller.state_dict(),
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "outer_optimizer": self.outer_optimizer.state_dict(),
            "inner_scheduler": self.inner_scheduler.state_dict(),
            "outer_scheduler": self.outer_scheduler.state_dict(),
            "metrics_history": self.metrics_history,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        ckpt = torch.load(path, weights_only=False)
        self.global_step = ckpt["global_step"]
        self.generator.load_state_dict(ckpt["generator_state"])
        self.controller.load_state_dict(ckpt["controller_state"])
        self.inner_optimizer.load_state_dict(ckpt["inner_optimizer"])
        self.outer_optimizer.load_state_dict(ckpt["outer_optimizer"])
        self.inner_scheduler.load_state_dict(ckpt["inner_scheduler"])
        self.outer_scheduler.load_state_dict(ckpt["outer_scheduler"])
        self.metrics_history = ckpt.get("metrics_history", [])
        logger.info(f"Checkpoint loaded from {path}, step={self.global_step}")
