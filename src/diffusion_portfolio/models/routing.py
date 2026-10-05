"""Residual mixture-of-experts routers for proposal Phase 0B2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_portfolio.models.evidential import (
    EvidentialRegimeHead,
)


@dataclass(frozen=True)
class RouterOutput:
    """Diagnostics from one routing operation."""

    condition: Tensor
    weights: Tensor
    base_probabilities: Tensor
    vacuity: Tensor | None


class ResidualExperts(nn.Module):
    """K zero-initialized residual condition experts."""

    def __init__(
        self,
        *,
        condition_dim: int,
        n_experts: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        if condition_dim <= 0:
            raise ValueError(
                "condition_dim must be positive"
            )

        if n_experts < 2:
            raise ValueError(
                "n_experts must be at least 2"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        self.condition_dim = condition_dim
        self.n_experts = n_experts

        self.experts = nn.ModuleList()

        for _ in range(
            n_experts
        ):
            expert = nn.Sequential(
                nn.Linear(
                    condition_dim,
                    hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(
                    hidden_dim,
                    condition_dim,
                ),
            )

            final_layer = expert[-1]

            nn.init.zeros_(
                final_layer.weight
            )

            nn.init.zeros_(
                final_layer.bias
            )

            self.experts.append(
                expert
            )

    def forward(
        self,
        condition: Tensor,
    ) -> Tensor:
        if condition.ndim != 2:
            raise ValueError(
                "condition must have shape "
                "[batch, condition_dim]"
            )

        if (
            condition.shape[1]
            != self.condition_dim
        ):
            raise ValueError(
                "condition dimension mismatch"
            )

        return torch.stack(
            [
                expert(
                    condition
                )
                for expert
                in self.experts
            ],
            dim=1,
        )


class SoftmaxResidualRouter(nn.Module):
    """Ordinary softmax MoE control for Phase 0B2."""

    def __init__(
        self,
        *,
        condition_dim: int,
        n_experts: int = 3,
        expert_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.condition_dim = condition_dim
        self.n_experts = n_experts

        self.routing_head = nn.Linear(
            condition_dim,
            n_experts,
        )

        self.experts = ResidualExperts(
            condition_dim=condition_dim,
            n_experts=n_experts,
            hidden_dim=expert_hidden_dim,
        )

    def route(
        self,
        condition: Tensor,
    ) -> RouterOutput:
        logits = self.routing_head(
            condition
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        expert_residuals = (
            self.experts(
                condition
            )
        )

        residual = (
            probabilities.unsqueeze(-1)
            * expert_residuals
        ).sum(
            dim=1
        )

        routed_condition = (
            condition
            + residual
        )

        return RouterOutput(
            condition=routed_condition,
            weights=probabilities,
            base_probabilities=(
                probabilities
            ),
            vacuity=None,
        )

    def forward(
        self,
        condition: Tensor,
    ) -> Tensor:
        return self.route(
            condition
        ).condition


class EvidentialResidualRouter(nn.Module):
    """Dirichlet uncertainty-aware residual MoE router."""

    def __init__(
        self,
        *,
        condition_dim: int,
        n_experts: int = 3,
        expert_hidden_dim: int = 128,
        uncertainty_threshold: float,
        transition_steepness: float = 10.0,
    ) -> None:
        super().__init__()

        if not (
            0.0
            < uncertainty_threshold
            < 1.0
        ):
            raise ValueError(
                "uncertainty_threshold must "
                "lie in (0, 1)"
            )

        if transition_steepness <= 0:
            raise ValueError(
                "transition_steepness "
                "must be positive"
            )

        self.condition_dim = condition_dim
        self.n_experts = n_experts

        self.uncertainty_threshold = float(
            uncertainty_threshold
        )

        self.transition_steepness = float(
            transition_steepness
        )

        self.evidential_head = (
            EvidentialRegimeHead(
                input_dim=condition_dim,
                n_classes=n_experts,
            )
        )

        self.experts = ResidualExperts(
            condition_dim=condition_dim,
            n_experts=n_experts,
            hidden_dim=expert_hidden_dim,
        )

    def route(
        self,
        condition: Tensor,
    ) -> RouterOutput:
        evidential = (
            self.evidential_head(
                condition
            )
        )

        probabilities = (
            evidential.probabilities
        )

        vacuity = (
            evidential.vacuity
        )

        blend = torch.sigmoid(
            self.transition_steepness
            * (
                vacuity
                - self.uncertainty_threshold
            )
        )

        uniform = torch.full_like(
            probabilities,
            1.0
            / self.n_experts,
        )

        weights = (
            (
                1.0
                - blend.unsqueeze(-1)
            )
            * probabilities
            + blend.unsqueeze(-1)
            * uniform
        )

        expert_residuals = (
            self.experts(
                condition
            )
        )

        residual = (
            weights.unsqueeze(-1)
            * expert_residuals
        ).sum(
            dim=1
        )

        routed_condition = (
            condition
            + residual
        )

        return RouterOutput(
            condition=routed_condition,
            weights=weights,
            base_probabilities=(
                probabilities
            ),
            vacuity=vacuity,
        )

    def forward(
        self,
        condition: Tensor,
    ) -> Tensor:
        return self.route(
            condition
        ).condition