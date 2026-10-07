"""Training-side conditional alpha-stable diffusion model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion_portfolio.models.diffusion.model import (
    ConditionalDiffusionModel,
    DiffusionTrainingOutput,
)
from diffusion_portfolio.models.levy.noise import (
    sample_ddpm_normalized_alpha_stable,
)
from diffusion_portfolio.models.levy.schedule import (
    LevyNoiseSchedule,
)


class ConditionalLevyDiffusionModel(
    ConditionalDiffusionModel
):
    """Conditional epsilon-prediction alpha-stable diffusion.

    This class currently implements the forward process and
    training objective only.

    Reverse sampling is deliberately not implemented yet.
    """

    def __init__(
        self,
        *,
        alpha: float,
        lookback: int,
        n_assets: int,
        condition_dim: int,
        history_hidden_dim: int,
        diffusion_steps: int,
        schedule_type: str,
        channels: list[int],
        time_embed_dim: int,
        n_res_blocks: int,
        history_encoder_type: str = "mlp",
        cde_hidden_dim: int = 128,
        cde_drift_hidden_dim: int = 256,
        cde_sensitivity_hidden_dim: int = 256,
        cde_solver: str = "dopri5",
        cde_rtol: float = 1e-4,
        cde_atol: float = 1e-5,
        cde_use_adjoint: bool = True,
        cde_fixed_steps_per_interval: int = 4,
    ) -> None:
        super().__init__(
            lookback=lookback,
            n_assets=n_assets,
            condition_dim=condition_dim,
            history_hidden_dim=history_hidden_dim,
            diffusion_steps=diffusion_steps,
            schedule_type=schedule_type,
            channels=channels,
            prediction_type="epsilon",
            time_embed_dim=time_embed_dim,
            n_res_blocks=n_res_blocks,
            history_encoder_type=(
                history_encoder_type
            ),
            cde_hidden_dim=cde_hidden_dim,
            cde_drift_hidden_dim=(
                cde_drift_hidden_dim
            ),
            cde_sensitivity_hidden_dim=(
                cde_sensitivity_hidden_dim
            ),
            cde_solver=cde_solver,
            cde_rtol=cde_rtol,
            cde_atol=cde_atol,
            cde_use_adjoint=(
                cde_use_adjoint
            ),
            cde_fixed_steps_per_interval=(
                cde_fixed_steps_per_interval
            ),
        )

        self.alpha = float(
            alpha
        )

        # Replace the Gaussian schedule created by the parent.
        self.noise_schedule = (
            LevyNoiseSchedule(
                alpha=self.alpha,
                n_steps=diffusion_steps,
                schedule_type=schedule_type,
            )
        )

    def training_loss_from_condition(
        self,
        condition: Tensor,
        target: Tensor,
        *,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> DiffusionTrainingOutput:
        """Compute the DLPM-style epsilon-prediction loss."""

        x_0 = self._prepare_target(
            target
        )

        if condition.shape != (
            x_0.shape[0],
            self.condition_dim,
        ):
            raise ValueError(
                "condition must have shape "
                "[batch, condition_dim]"
            )

        condition = condition.to(
            device=x_0.device,
            dtype=x_0.dtype,
        )

        batch_size = (
            x_0.shape[0]
        )

        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.diffusion_steps,
                size=(
                    batch_size,
                ),
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
                    "timesteps must have "
                    "shape [batch]"
                )

            if (
                torch.any(
                    timesteps < 0
                )
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
            noise = (
                sample_ddpm_normalized_alpha_stable(
                    alpha=self.alpha,
                    shape=tuple(
                        x_0.shape
                    ),
                    device=x_0.device,
                    dtype=x_0.dtype,
                )
            )

        else:
            noise = noise.to(
                device=x_0.device,
                dtype=x_0.dtype,
            )

            if noise.shape != (
                x_0.shape
            ):
                raise ValueError(
                    "noise must match "
                    "target shape"
                )

        x_t = (
            self.noise_schedule.q_sample(
                x_0,
                timesteps,
                noise,
            )
        )

        prediction = (
            self.score_network(
                x_t,
                timesteps,
                condition,
            )
        )

        # DLPM simple objective:
        # predict the alpha-stable perturbation itself.
        training_target = noise

        loss = F.mse_loss(
            prediction,
            training_target,
        )

        return DiffusionTrainingOutput(
            loss=loss,
            prediction=prediction,
            training_target=(
                training_target
            ),
            noisy_target=x_t,
            timesteps=timesteps,
        )

    @torch.no_grad()
    def sample_from_condition(
        self,
        condition: Tensor,
        *,
        n_scenarios: int,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError(
            "Levy reverse sampling has not "
            "been implemented yet"
        )