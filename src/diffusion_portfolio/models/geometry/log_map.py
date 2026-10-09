"""
Riemannian Logarithmic Map on SPD Manifolds.

Reference: Proposal §5.4.3 — Logarithmic Map to Tangent Space.

Projects SPD density matrices to the Euclidean tangent space:

    v_t = ρ_ref^{1/2} log(ρ_ref^{-1/2} ρ_t ρ_ref^{-1/2}) ρ_ref^{1/2}

The tangent vector v_t ∈ T_{ρ_ref}M is a symmetric matrix that lives
in a flat (Euclidean) space, making it compatible with standard neural
network operations (linear layers, attention, etc.).
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch import Tensor

from diffusion_portfolio.utils.matrix_ops import (
    stable_matrix_sqrt,
    stable_matrix_inv_sqrt,
    stable_matrix_log,
    symmetrise,
    upper_triangular_to_vector,
)

logger = logging.getLogger(__name__)


class RiemannianLogMap(nn.Module):
    r"""Riemannian logarithmic map on the SPD manifold.

    .. math::
        v_t = \rho_{ref}^{1/2} \log\!\left(\rho_{ref}^{-1/2}\,\rho_t\,\rho_{ref}^{-1/2}\right) \rho_{ref}^{1/2}

    Args:
        n_assets: N — matrix dimension.
        reference_point: Type of base point ρ_ref.
        ema_decay: Decay for EMA reference.
        flatten_output: If True, output is the upper-tri vector ∈ ℝ^{N(N+1)/2}.
        eigenvalue_clamp_min: Minimum eigenvalue for stability.
    """

    def __init__(
        self,
        n_assets: int,
        reference_point: str = "identity",
        ema_decay: float = 0.99,
        flatten_output: bool = True,
        eigenvalue_clamp_min: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.reference_type = reference_point
        self.ema_decay = ema_decay
        self.flatten_output = flatten_output
        self.eigenvalue_clamp_min = eigenvalue_clamp_min

        # Initialise reference as scaled identity: ρ_ref = I/N
        ref = torch.eye(n_assets) / n_assets
        self.register_buffer("reference", ref)
        self.register_buffer("_ema_ref", ref.clone())

    def update_reference_ema(self, rho: Tensor) -> None:
        """Update the EMA reference point (call during training).

        Args:
            rho: Current density matrix batch, shape [B, N, N].
        """
        if self.reference_type == "ema":
            batch_mean = rho.mean(dim=0)
            self._ema_ref.mul_(self.ema_decay).add_(
                (1 - self.ema_decay) * batch_mean
            )
            self.reference.copy_(self._ema_ref)

    def forward(self, rho: Tensor) -> Tensor:
        r"""Compute the logarithmic map.

        Args:
            rho: Density matrices, shape [..., N, N].

        Returns:
            If flatten_output: tangent vectors, shape [..., N(N+1)/2].
            Else: tangent matrices, shape [..., N, N].
        """
        ref = self.reference  # [N, N]
        eps = self.eigenvalue_clamp_min

        # ρ_ref^{1/2} and ρ_ref^{-1/2}
        sqrt_ref = stable_matrix_sqrt(ref.unsqueeze(0), eps).squeeze(0)
        inv_sqrt_ref = stable_matrix_inv_sqrt(ref.unsqueeze(0), eps).squeeze(0)

        # ρ_ref^{-1/2} · ρ_t · ρ_ref^{-1/2}
        inner = inv_sqrt_ref @ rho @ inv_sqrt_ref
        inner = symmetrise(inner)

        # log(inner)
        log_inner = stable_matrix_log(inner, eps)

        # v_t = ρ_ref^{1/2} · log(·) · ρ_ref^{1/2}
        v = sqrt_ref @ log_inner @ sqrt_ref
        v = symmetrise(v)

        if self.flatten_output:
            return upper_triangular_to_vector(v)
        return v
