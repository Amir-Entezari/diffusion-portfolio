"""
Diffusion Engine — Training and Sampling.

Reference: Proposal §5.6.3 — Denoising Score Matching Loss.

    L_DSM = E_{t,Y_0,ε}[||s_θ(Y_t, t, c) − ε||²]

Reference: Proposal §5.6.4 — Reverse SDE with Markov Superposition Bridge.
"""

from __future__ import annotations
import logging
import torch
import torch.nn as nn
from torch import Tensor

from models.phase3_diffusion.noise_schedule import NoiseSchedule
from models.phase3_diffusion.score_network import ScoreNetwork
from models.phase3_diffusion.jump_process import JumpProcess

logger = logging.getLogger(__name__)


class JumpDiffusionEngine(nn.Module):
    r"""Complete Phase 3: Jump-Diffusion Generative Engine.

    Training (§5.6.3):
        1. Sample t ~ Uniform({1,...,T}), ε ~ N(0,I)
        2. Y_t = √ᾱ_t Y_0 + √(1−ᾱ_t) ε + J·dN_t
        3. L = ||s_θ(Y_t, t, c_unified) − ε||²

    Sampling (§5.6.4):
        Reverse SDE with Euler-Maruyama + jump compensation:
        Y_{t−1} = (1/√α_t)(Y_t − (β_t/√(1−ᾱ_t))s_θ(Y_t,t,c)) + σ_t z + J_rev

    Args:
        data_dim: N — asset dimension.
        condition_dim: c_unified dimension.
        n_steps: T — total diffusion steps.
        schedule_type: Noise schedule type.
        jump_intensity: λ — Poisson jump rate.
    """

    def __init__(
        self,
        data_dim: int = 12,
        condition_dim: int = 256,
        n_steps: int = 1000,
        schedule_type: str = "cosine",
        jump_intensity: float = 0.05,
        channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.n_steps = n_steps

        self.noise_schedule = NoiseSchedule(
            n_steps=n_steps, schedule_type=schedule_type,
        )
        self.score_net = ScoreNetwork(
            data_dim=data_dim, condition_dim=condition_dim,
            channels=channels,
        )
        self.jump_process = JumpProcess(jump_intensity=jump_intensity)

    def training_loss(
        self, x_0: Tensor, c_unified: Tensor,
    ) -> dict[str, Tensor]:
        r"""Compute denoising score matching loss.

        Args:
            x_0: Clean data Y_0, shape [B, D].
            c_unified: Conditioning, shape [B, C].

        Returns:
            Dict with 'loss' (scalar) and 'predicted_noise' [B, D].
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.noise_schedule.q_sample(x_0, t, noise)

        # Add jump noise
        jump_noise = self.jump_process(x_t, t[0].item())
        x_t = x_t + jump_noise

        # Predict noise
        predicted = self.score_net(x_t, t, c_unified)

        # MSE loss (ε-prediction formulation)
        loss = nn.functional.mse_loss(predicted, noise)

        return {"loss": loss, "predicted_noise": predicted}

    @torch.no_grad()
    def sample(
        self, c_unified: Tensor, n_samples: int | None = None,
    ) -> Tensor:
        """Generate samples via reverse SDE.

        Args:
            c_unified: Conditioning, shape [B, C].
            n_samples: Override batch size.

        Returns:
            Generated samples Y_0, shape [B, D].
        """
        B = n_samples or c_unified.shape[0]
        device = c_unified.device

        # Start from pure noise
        x = torch.randn(B, self.data_dim, device=device)

        for t_idx in reversed(range(self.n_steps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            coeff = self.noise_schedule.get_coefficients(t)

            beta = coeff["beta"]
            sqrt_ac = coeff["sqrt_alpha_cumprod"]
            sqrt_omc = coeff["sqrt_one_minus_alpha_cumprod"]
            alpha = 1.0 - beta

            # Predict noise
            pred_noise = self.score_net(x, t, c_unified)

            # Reverse step mean
            shape = [B] + [1] * (x.dim() - 1)
            mean = (1.0 / alpha.reshape(shape).sqrt()) * (
                x - beta.reshape(shape) / sqrt_omc.reshape(shape) * pred_noise
            )

            # Add noise (except at t=0)
            if t_idx > 0:
                z = torch.randn_like(x)
                sigma = beta.reshape(shape).sqrt()
                x = mean + sigma * z
            else:
                x = mean

        return x
