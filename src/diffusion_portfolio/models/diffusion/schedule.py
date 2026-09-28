"""
Noise Schedule for Diffusion Process.

Implements β(t) variance schedules: linear, cosine, and learned.
Precomputes cumulative products ᾱ_t for efficient training.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor


class NoiseSchedule(nn.Module):
    """Variance schedule β(t) with precomputed diffusion coefficients.

    Precomputes:
        α_t = 1 − β_t
        ᾱ_t = Π_{s=1}^{t} α_s
        √ᾱ_t, √(1−ᾱ_t) for q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I)

    Args:
        n_steps: T — total diffusion steps.
        beta_start: β(0).
        beta_end: β(T).
        schedule_type: 'linear', 'cosine', or 'learned'.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ) -> None:
        super().__init__()
        self.n_steps = n_steps

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule_type == "cosine":
            betas = self._cosine_schedule(n_steps)
        elif schedule_type == "learned":
            self.log_betas = nn.Parameter(
                torch.linspace(math.log(beta_start), math.log(beta_end), n_steps)
            )
            betas = None
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        self.schedule_type = schedule_type
        if betas is not None:
            self._register_schedule(betas)

    @staticmethod
    def _cosine_schedule(n_steps: int, s: float = 0.008) -> Tensor:
        """Cosine schedule from Nichol & Dhariwal (2021)."""
        t = torch.linspace(0, 1, n_steps + 1)
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(max=0.999)

    def _register_schedule(self, betas: Tensor) -> None:
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    @property
    def effective_betas(self) -> Tensor:
        if self.schedule_type == "learned":
            betas = torch.sigmoid(self.log_betas) * 0.02
            return betas
        return self.betas

    def get_coefficients(self, t: Tensor) -> dict[str, Tensor]:
        """Get diffusion coefficients at timesteps t.

        Args:
            t: Timestep indices, shape [B]. Integer in [0, T-1].

        Returns:
            Dict with sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, beta.
        """
        if self.schedule_type == "learned":
            betas = self.effective_betas
            alphas = 1.0 - betas
            ac = torch.cumprod(alphas, dim=0)
            return {
                "sqrt_alpha_cumprod": ac[t].sqrt(),
                "sqrt_one_minus_alpha_cumprod": (1 - ac[t]).sqrt(),
                "beta": betas[t],
            }
        return {
            "sqrt_alpha_cumprod": self.sqrt_alphas_cumprod[t],
            "sqrt_one_minus_alpha_cumprod": self.sqrt_one_minus_alphas_cumprod[t],
            "beta": self.betas[t],
        }

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        r"""Forward diffusion: q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I).

        Args:
            x_0: Clean data, shape [B, ...].
            t: Timesteps, shape [B].
            noise: Gaussian noise ε ~ N(0,I), same shape as x_0.

        Returns:
            Noised data x_t, same shape as x_0.
        """
        coeff = self.get_coefficients(t)
        # Reshape for broadcasting
        shape = [x_0.shape[0]] + [1] * (x_0.dim() - 1)
        sqrt_ac = coeff["sqrt_alpha_cumprod"].reshape(shape)
        sqrt_omc = coeff["sqrt_one_minus_alpha_cumprod"].reshape(shape)
        return sqrt_ac * x_0 + sqrt_omc * noise
