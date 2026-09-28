"""Vanilla conditional DDPM for next-day multivariate excess returns."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusion_portfolio.models.diffusion.conditioning import (
    HistoryEncoder,
)
from diffusion_portfolio.models.diffusion.schedule import (
    NoiseSchedule,
)
from diffusion_portfolio.models.diffusion.score_network import (
    ScoreNetwork,
)


@dataclass(frozen=True)
class DiffusionTrainingOutput:
    """Intermediate values from one DDPM training step."""

    loss: Tensor
    predicted_noise: Tensor
    target_noise: Tensor
    noisy_target: Tensor
    timesteps: Tensor


class ConditionalDiffusionModel(nn.Module):
    """Conditional epsilon-prediction DDPM.

    The model learns:

        epsilon_theta(x_t, t, history)

    where:

        history:
            previous L standardized excess returns

        x_0:
            standardized next-day multivariate excess return

        x_t:
            noisy version of x_0 at diffusion timestep t
    """

    def __init__(
        self,
        *,
        lookback: int,
        n_assets: int,
        condition_dim: int,
        history_hidden_dim: int,
        diffusion_steps: int,
        schedule_type: str,
        channels: list[int],
        time_embed_dim: int,
        n_res_blocks: int,
    ) -> None:
        super().__init__()

        if diffusion_steps <= 1:
            raise ValueError(
                "diffusion_steps must be > 1"
            )

        self.lookback = lookback
        self.n_assets = n_assets
        self.diffusion_steps = (
            diffusion_steps
        )

        self.history_encoder = (
            HistoryEncoder(
                lookback=lookback,
                n_assets=n_assets,
                condition_dim=condition_dim,
                hidden_dim=(
                    history_hidden_dim
                ),
            )
        )

        self.noise_schedule = (
            NoiseSchedule(
                n_steps=diffusion_steps,
                schedule_type=(
                    schedule_type
                ),
            )
        )

        self.score_network = (
            ScoreNetwork(
                data_dim=n_assets,
                channels=channels,
                time_embed_dim=(
                    time_embed_dim
                ),
                condition_dim=(
                    condition_dim
                ),
                n_res_blocks=(
                    n_res_blocks
                ),
            )
        )

    def _prepare_target(
        self,
        target: Tensor,
    ) -> Tensor:
        """Convert [B,1,N] or [B,N] target to [B,N]."""

        if target.ndim == 3:
            if target.shape[1] != 1:
                raise ValueError(
                    "Vanilla MVP currently supports "
                    "horizon=1 only"
                )

            target = target[
                :,
                0,
                :,
            ]

        elif target.ndim != 2:
            raise ValueError(
                "target must have shape "
                "[batch, assets] or "
                "[batch, 1, assets]"
            )

        if target.shape[1] != self.n_assets:
            raise ValueError(
                "target asset dimension "
                "does not match model"
            )

        return target

    def predict_noise(
        self,
        noisy_target: Tensor,
        timesteps: Tensor,
        history: Tensor,
    ) -> Tensor:
        """Predict epsilon from x_t, timestep, and historical context."""

        if noisy_target.ndim != 2:
            raise ValueError(
                "noisy_target must have "
                "shape [batch, assets]"
            )

        if noisy_target.shape[1] != self.n_assets:
            raise ValueError(
                "noisy_target asset dimension "
                "does not match model"
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [batch]"
            )

        if (
            timesteps.shape[0]
            != noisy_target.shape[0]
        ):
            raise ValueError(
                "timestep batch dimension mismatch"
            )

        condition = (
            self.history_encoder(
                history
            )
        )

        return self.score_network(
            noisy_target,
            timesteps,
            condition,
        )

    def training_loss(
        self,
        history: Tensor,
        target: Tensor,
        *,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> DiffusionTrainingOutput:
        """Calculate the standard DDPM epsilon-prediction objective."""

        x_0 = self._prepare_target(
            target
        )

        if history.shape[0] != x_0.shape[0]:
            raise ValueError(
                "history and target batch sizes differ"
            )

        batch_size = x_0.shape[0]

        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.diffusion_steps,
                size=(batch_size,),
                device=x_0.device,
            )

        else:
            timesteps = timesteps.to(
                device=x_0.device,
                dtype=torch.long,
            )

            if timesteps.shape != (
                batch_size,
            ):
                raise ValueError(
                    "timesteps must have shape [batch]"
                )

            if (
                torch.any(timesteps < 0)
                or torch.any(
                    timesteps
                    >= self.diffusion_steps
                )
            ):
                raise ValueError(
                    "timesteps are outside "
                    "the diffusion schedule"
                )

        if noise is None:
            noise = torch.randn_like(
                x_0
            )

        else:
            noise = noise.to(
                device=x_0.device,
                dtype=x_0.dtype,
            )

            if noise.shape != x_0.shape:
                raise ValueError(
                    "noise must match target shape"
                )

        x_t = self.noise_schedule.q_sample(
            x_0,
            timesteps,
            noise,
        )

        predicted_noise = (
            self.predict_noise(
                x_t,
                timesteps,
                history,
            )
        )

        loss = F.mse_loss(
            predicted_noise,
            noise,
        )

        return DiffusionTrainingOutput(
            loss=loss,
            predicted_noise=(
                predicted_noise
            ),
            target_noise=noise,
            noisy_target=x_t,
            timesteps=timesteps,
        )