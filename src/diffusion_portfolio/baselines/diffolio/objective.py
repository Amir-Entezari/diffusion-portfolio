"""Complete Diffolio diffusion training objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_portfolio.models.diffusion.schedule import (
    NoiseSchedule,
)
from diffusion_portfolio.baselines.diffolio.correlation_loss import (
    diffolio_correlation_regularizer,
)
from diffusion_portfolio.baselines.diffolio.correlation_target import (
    estimate_shrinkage_correlation,
)
from diffusion_portfolio.baselines.diffolio.denoiser import (
    DiffolioDenoiser,
)


@dataclass(frozen=True)
class DiffolioTrainingOutput:
    """Outputs from one Diffolio training objective evaluation."""

    loss: torch.Tensor
    ddpm_loss: torch.Tensor
    correlation_loss: torch.Tensor

    prediction: torch.Tensor
    noise: torch.Tensor
    noisy_target: torch.Tensor

    # Internal schedule indices are zero-based.
    schedule_timesteps: torch.Tensor

    # Paper notation tau is one-based: {1, ..., T}.
    diffusion_timesteps: torch.Tensor

    target_correlation: torch.Tensor
    shrinkage: torch.Tensor


class DiffolioObjective(nn.Module):
    """Diffolio DDPM + correlation-guided training objective."""

    def __init__(
        self,
        *,
        training_covariance: torch.Tensor,
        n_assets: int = 12,
        n_asset_characteristics: int = 10,
        n_systematic: int = 8,
        lookback: int = 63,
        hidden_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        time_embedding_dim: int = 32,
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lambda_corr: float = 0.05,
    ) -> None:
        super().__init__()

        if diffusion_steps <= 1:
            raise ValueError(
                "diffusion_steps must be > 1"
            )

        if lambda_corr < 0.0:
            raise ValueError(
                "lambda_corr cannot be negative"
            )

        if training_covariance.shape != (
            n_assets,
            n_assets,
        ):
            raise ValueError(
                "training_covariance must have shape "
                "[n_assets, n_assets]"
            )

        if not torch.isfinite(
            training_covariance
        ).all():
            raise ValueError(
                "training_covariance contains NaN "
                "or infinite values"
            )

        self.n_assets = n_assets
        self.lookback = lookback
        self.diffusion_steps = diffusion_steps
        self.lambda_corr = float(
            lambda_corr
        )

        self.denoiser = DiffolioDenoiser(
            n_assets=n_assets,
            n_asset_characteristics=(
                n_asset_characteristics
            ),
            n_systematic=n_systematic,
            lookback=lookback,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            time_embedding_dim=(
                time_embedding_dim
            ),
        )

        # Diffolio uses the original DDPM linear schedule.
        self.noise_schedule = NoiseSchedule(
            n_steps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type="linear",
        )

        # Fixed training-period target. It follows the model
        # across devices and is included in checkpoints but
        # receives no gradients.
        self.register_buffer(
            "training_covariance",
            training_covariance
            .detach()
            .clone()
            .float(),
        )

    def training_loss(
        self,
        target: torch.Tensor,
        return_history: torch.Tensor,
        return_history_raw: torch.Tensor,
        asset_covariates: torch.Tensor,
        systematic_covariates: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> DiffolioTrainingOutput:
        """Evaluate the full Diffolio objective for one minibatch.

        ``target`` and ``return_history`` are in model space.

        ``return_history_raw`` contains raw decimal excess
        returns and is used only for the shrinkage correlation
        target.
        """

        if target.ndim != 2:
            raise ValueError(
                "target must have shape [batch, assets]"
            )

        batch_size = target.shape[
            0
        ]

        if target.shape != (
            batch_size,
            self.n_assets,
        ):
            raise ValueError(
                "target asset dimension does not match model"
            )

        if return_history_raw.shape != (
            batch_size,
            self.lookback,
            self.n_assets,
        ):
            raise ValueError(
                "return_history_raw has unexpected shape"
            )

        if timesteps is None:
            schedule_timesteps = torch.randint(
                low=0,
                high=self.diffusion_steps,
                size=(
                    batch_size,
                ),
                device=target.device,
                dtype=torch.long,
            )

        else:
            schedule_timesteps = timesteps.to(
                device=target.device,
                dtype=torch.long,
            )

            if schedule_timesteps.shape != (
                batch_size,
            ):
                raise ValueError(
                    "timesteps must have shape [batch]"
                )

            if (
                torch.any(
                    schedule_timesteps < 0
                )
                or torch.any(
                    schedule_timesteps
                    >= self.diffusion_steps
                )
            ):
                raise ValueError(
                    "timesteps are outside "
                    "the diffusion schedule"
                )

        if noise is None:
            noise = torch.randn_like(
                target
            )

        else:
            noise = noise.to(
                device=target.device,
                dtype=target.dtype,
            )

            if noise.shape != target.shape:
                raise ValueError(
                    "noise must match target shape"
                )

        # Forward diffusion q(x_tau | x_0).
        noisy_target = (
            self.noise_schedule.q_sample(
                target,
                schedule_timesteps,
                noise,
            )
        )

        # Our NoiseSchedule arrays use Python/PyTorch indices
        # 0,...,T-1. The paper denotes diffusion steps as
        # tau = 1,...,T. The denoiser therefore receives the
        # explicitly one-based value.
        diffusion_timesteps = (
            schedule_timesteps
            + 1
        )

        denoiser_output = self.denoiser(
            noisy_target,
            diffusion_timesteps,
            return_history,
            asset_covariates,
            systematic_covariates,
        )

        ddpm_loss = F.mse_loss(
            denoiser_output.noise,
            noise,
        )

        correlation_target = (
            estimate_shrinkage_correlation(
                return_history_raw,
                self.training_covariance,
            )
        )

        correlation_loss = (
            diffolio_correlation_regularizer(
                denoiser_output.market_attention,
                correlation_target.correlation,
                n_assets=self.n_assets,
            )
        )

        loss = (
            ddpm_loss
            + self.lambda_corr
            * correlation_loss
        )

        return DiffolioTrainingOutput(
            loss=loss,
            ddpm_loss=ddpm_loss,
            correlation_loss=correlation_loss,
            prediction=denoiser_output.noise,
            noise=noise,
            noisy_target=noisy_target,
            schedule_timesteps=(
                schedule_timesteps
            ),
            diffusion_timesteps=(
                diffusion_timesteps
            ),
            target_correlation=(
                correlation_target.correlation
            ),
            shrinkage=(
                correlation_target.shrinkage
            ),
        )