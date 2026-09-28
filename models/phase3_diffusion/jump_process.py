"""
Jump Process for Compound Poisson Augmentation.

Reference: Proposal §5.6.1 — Forward Jump-Diffusion SDE.

    dN_t ~ Poisson(λ(t))
    J(Y_{t⁻}) ~ NIG or Variance-Gamma distribution

Augments the standard diffusion with realistic heavy-tailed jumps
modeling flash crashes and Black Swan events.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class JumpProcess(nn.Module):
    """Compound Poisson jump process.

    Args:
        jump_intensity: λ — base Poisson rate per step.
        jump_distribution: Distribution family for magnitudes.
        jump_scale: Scale of jump magnitudes.
    """

    def __init__(
        self,
        jump_intensity: float = 0.05,
        jump_distribution: str = "gaussian",
        jump_scale: float = 0.1,
        jump_mean: float = -0.03,
    ) -> None:
        super().__init__()
        self.jump_intensity = jump_intensity
        self.jump_distribution = jump_distribution
        self.jump_scale = jump_scale
        self.jump_mean = jump_mean

    def sample_jumps(
        self, shape: tuple, device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Sample jump events and magnitudes.

        Args:
            shape: Output shape [B, D].
            device: Target device.

        Returns:
            Tuple of (jump_mask, jump_magnitudes), each shape [B, D].
        """
        # Poisson arrivals
        jump_counts = torch.poisson(
            torch.full(shape, self.jump_intensity, device=device)
        )
        jump_mask = (jump_counts > 0).float()

        # Jump magnitudes
        if self.jump_distribution == "gaussian":
            magnitudes = torch.normal(
                self.jump_mean, self.jump_scale, size=shape, device=device
            )
        elif self.jump_distribution == "normal_inverse_gaussian":
            # Approximate NIG via location-scale mixture
            mixing = torch.distributions.InverseGamma(
                torch.tensor(1.0, device=device),
                torch.tensor(1.0, device=device),
            ).sample(shape)
            magnitudes = torch.normal(
                self.jump_mean, self.jump_scale * mixing.sqrt()
            )
        else:
            magnitudes = torch.normal(
                self.jump_mean, self.jump_scale, size=shape, device=device
            )

        return jump_mask, jump_mask * magnitudes

    def forward(
        self, x: Tensor, t_idx: int,
    ) -> Tensor:
        """Add jump noise to diffusion state.

        Args:
            x: Current state, shape [B, D].
            t_idx: Current time index.

        Returns:
            Jump contribution J·dN, shape [B, D].
        """
        mask, magnitudes = self.sample_jumps(x.shape, x.device)
        return magnitudes
