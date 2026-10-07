"""Scale-preserving schedule for Phase-3 alpha-stable diffusion."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from diffusion_portfolio.models.diffusion.schedule import (
    NoiseSchedule,
)


class LevyNoiseSchedule(nn.Module):
    """DLPM-style scale-preserving alpha-stable schedule.

    The schedule reuses the project's existing Gaussian beta
    discretization and transforms it into stable coefficients.

    Let

        a_t = 1 - beta_t.

    Then

        gamma_t = a_t^(1 / alpha)

        sigma_t =
            (1 - gamma_t^alpha)^(1 / alpha)

    and

        bar_gamma_t =
            product_{s <= t} gamma_s

        bar_sigma_t =
            (1 - bar_gamma_t^alpha)^(1 / alpha).

    With DDPM-normalized alpha-stable noise, alpha=2 exactly
    recovers the existing Gaussian DDPM forward marginal.
    """

    def __init__(
        self,
        *,
        alpha: float,
        n_steps: int,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()

        alpha = float(
            alpha
        )

        if not (
            1.0
            < alpha
            <= 2.0
        ):
            raise ValueError(
                "alpha must satisfy 1 < alpha <= 2"
            )

        if n_steps <= 1:
            raise ValueError(
                "n_steps must be > 1"
            )

        if schedule_type == "learned":
            raise ValueError(
                "learned beta schedules are not "
                "supported for LevyNoiseSchedule"
            )

        self.alpha = alpha
        self.n_steps = int(
            n_steps
        )

        self.schedule_type = (
            schedule_type
        )

        gaussian_schedule = (
            NoiseSchedule(
                n_steps=n_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                schedule_type=schedule_type,
            )
        )

        betas = (
            gaussian_schedule
            .effective_betas
            .detach()
            .clone()
        )

        gaussian_alphas = (
            1.0
            - betas
        )

        if alpha == 2.0:
            # Exact DDPM endpoint.
            #
            # Reuse the existing Gaussian schedule's stored
            # cumulative coefficients rather than recomputing
            # mathematically equivalent expressions in a
            # different floating-point order.
            gammas = torch.sqrt(
                gaussian_alphas
            )

            sigmas = torch.sqrt(
                betas
            )

            bar_gammas = (
                gaussian_schedule
                .sqrt_alphas_cumprod
                .detach()
                .clone()
            )

            bar_sigmas = (
                gaussian_schedule
                .sqrt_one_minus_alphas_cumprod
                .detach()
                .clone()
            )

        else:
            gammas = torch.pow(
                gaussian_alphas,
                1.0 / alpha,
            )

            bar_gammas = torch.cumprod(
                gammas,
                dim=0,
            )

            sigmas_alpha = (
                1.0
                - torch.pow(
                    gammas,
                    alpha,
                )
            ).clamp_min(
                0.0
            )

            bar_sigmas_alpha = (
                1.0
                - torch.pow(
                    bar_gammas,
                    alpha,
                )
            ).clamp_min(
                0.0
            )

            sigmas = torch.pow(
                sigmas_alpha,
                1.0 / alpha,
            )

            bar_sigmas = torch.pow(
                bar_sigmas_alpha,
                1.0 / alpha,
            )

        
        self.register_buffer(
            "betas",
            betas,
        )

        self.register_buffer(
            "gammas",
            gammas,
        )

        self.register_buffer(
            "bar_gammas",
            bar_gammas,
        )

        self.register_buffer(
            "sigmas",
            sigmas,
        )

        self.register_buffer(
            "bar_sigmas",
            bar_sigmas,
        )

    def get_coefficients(
        self,
        timesteps: Tensor,
    ) -> dict[str, Tensor]:
        """Return per-sample stable diffusion coefficients."""

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [batch]"
            )

        timesteps = timesteps.to(
            device=self.betas.device,
            dtype=torch.long,
        )

        if (
            torch.any(
                timesteps < 0
            )
            or torch.any(
                timesteps
                >= self.n_steps
            )
        ):
            raise ValueError(
                "timesteps are outside "
                "the diffusion schedule"
            )

        return {
            "gamma": self.gammas[
                timesteps
            ],
            "bar_gamma": (
                self.bar_gammas[
                    timesteps
                ]
            ),
            "sigma": self.sigmas[
                timesteps
            ],
            "bar_sigma": (
                self.bar_sigmas[
                    timesteps
                ]
            ),
            "beta": self.betas[
                timesteps
            ],
        }

    def q_sample(
        self,
        x_0: Tensor,
        timesteps: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Sample the closed-form forward marginal.

        Computes

            x_t =
                bar_gamma_t * x_0
                + bar_sigma_t * epsilon_alpha

        where epsilon_alpha follows the project's
        DDPM-normalized isotropic alpha-stable convention.
        """

        if noise.shape != x_0.shape:
            raise ValueError(
                "noise must match x_0 shape"
            )

        if timesteps.shape != (
            x_0.shape[0],
        ):
            raise ValueError(
                "timesteps must have shape [batch]"
            )

        coefficients = (
            self.get_coefficients(
                timesteps
            )
        )

        shape = (
            x_0.shape[0],
            *(
                [1]
                * (
                    x_0.ndim
                    - 1
                )
            ),
        )

        bar_gamma = (
            coefficients[
                "bar_gamma"
            ]
            .to(
                device=x_0.device,
                dtype=x_0.dtype,
            )
            .reshape(
                shape
            )
        )

        bar_sigma = (
            coefficients[
                "bar_sigma"
            ]
            .to(
                device=x_0.device,
                dtype=x_0.dtype,
            )
            .reshape(
                shape
            )
        )

        noise = noise.to(
            device=x_0.device,
            dtype=x_0.dtype,
        )

        return (
            bar_gamma
            * x_0
            + bar_sigma
            * noise
        )