from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_portfolio.models.diffusion.model import (
    ConditionalDiffusionModel,
    DiffusionTrainingOutput,
)


class PrecomputedConditionDiffusion(nn.Module):
    """Diffusion model driven by cached CDE-derived conditions."""

    def __init__(
        self,
        diffusion: ConditionalDiffusionModel,
        *,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()

        if extra_dim < 0:
            raise ValueError(
                "extra_dim cannot be negative"
            )

        self.diffusion = diffusion

        # The history encoder is unused with cached conditions.
        self.diffusion.history_encoder = (
            nn.Identity()
        )

        self.condition_dim = (
            diffusion.condition_dim
        )
        self.extra_dim = extra_dim

        if extra_dim == 0:
            self.residual_projection = None

        else:
            self.residual_projection = (
                nn.Linear(
                    extra_dim,
                    self.condition_dim,
                )
            )

            nn.init.zeros_(
                self.residual_projection.weight
            )

            nn.init.zeros_(
                self.residual_projection.bias
            )

    @property
    def diffusion_steps(self) -> int:
        return self.diffusion.diffusion_steps

    @property
    def n_assets(self) -> int:
        return self.diffusion.n_assets

    def make_condition(
        self,
        features: Tensor,
    ) -> Tensor:
        expected_dim = (
            self.condition_dim
            + self.extra_dim
        )

        if (
            features.ndim != 2
            or features.shape[1]
            != expected_dim
        ):
            raise ValueError(
                f"features must have shape "
                f"[batch, {expected_dim}]"
            )

        condition = features[
            :,
            :self.condition_dim,
        ]

        if self.residual_projection is not None:
            extra = features[
                :,
                self.condition_dim:,
            ]

            condition = (
                condition
                + self.residual_projection(
                    extra
                )
            )

        return condition

    def training_loss(
        self,
        history: Tensor,
        target: Tensor,
        *,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> DiffusionTrainingOutput:
        condition = self.make_condition(
            history
        )

        return (
            self.diffusion
            .training_loss_from_condition(
                condition,
                target,
                noise=noise,
                timesteps=timesteps,
            )
        )

    @torch.no_grad()
    def sample(
        self,
        history: Tensor,
        *,
        n_scenarios: int,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        condition = self.make_condition(
            history
        )

        return (
            self.diffusion
            .sample_from_condition(
                condition,
                n_scenarios=n_scenarios,
                initial_noise=initial_noise,
            )
        )