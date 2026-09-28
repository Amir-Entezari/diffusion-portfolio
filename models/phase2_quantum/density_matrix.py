"""
Density Matrix Construction from Covariance.

Reference: Proposal §5.4.1 — Covariance to Density Matrix Mapping.

Maps the empirical covariance matrix to a quantum density matrix:

    ρ_t = Σ_t / Tr(Σ_t)

Guarantees:
    - ρ_t ⪰ 0 (positive semi-definite)
    - Tr(ρ_t) = 1 (unit trace, like a quantum probability)
    - ρ_t ∈ S⁺⁺_N (SPD manifold when regularised)
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch import Tensor

from utils.matrix_ops import symmetrise, regularise_matrix, batch_trace

logger = logging.getLogger(__name__)


class DensityMatrixBuilder(nn.Module):
    r"""Constructs quantum density matrices from rolling covariance.

    .. math::
        \rho_t = \frac{\Sigma_t + \epsilon I}{\text{Tr}(\Sigma_t + \epsilon I)}

    Args:
        n_assets: N — asset universe size.
        window_size: Rolling window for empirical covariance.
        regularization_eps: ε — Tikhonov regularisation for strict PD.
    """

    def __init__(
        self,
        n_assets: int = 12,
        window_size: int = 60,
        regularization_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.window_size = window_size
        self.eps = regularization_eps

    def compute_rolling_covariance(self, returns: Tensor) -> Tensor:
        """Compute rolling covariance matrices.

        Args:
            returns: Asset returns, shape [B, T, N].

        Returns:
            Rolling covariance, shape [B, T_valid, N, N].
            T_valid = T - window_size + 1.
        """
        B, T, N = returns.shape
        W = self.window_size
        T_valid = T - W + 1

        covs = []
        for t in range(T_valid):
            window = returns[:, t:t + W, :]  # [B, W, N]
            # Demean within window
            mean = window.mean(dim=1, keepdim=True)  # [B, 1, N]
            centered = window - mean  # [B, W, N]
            # Covariance: (1/(W-1)) X^T X
            cov = torch.bmm(
                centered.transpose(1, 2), centered
            ) / (W - 1)  # [B, N, N]
            covs.append(cov)

        return torch.stack(covs, dim=1)  # [B, T_valid, N, N]

    def covariance_to_density(self, cov: Tensor) -> Tensor:
        r"""Convert covariance matrix to density matrix.

        .. math::
            \rho = \frac{\Sigma + \epsilon I}{\text{Tr}(\Sigma + \epsilon I)}

        Args:
            cov: Covariance matrix, shape [..., N, N].

        Returns:
            Density matrix ρ, shape [..., N, N].
            Satisfies: ρ ⪰ 0, Tr(ρ) = 1.
        """
        # Symmetrise and regularise
        cov_reg = regularise_matrix(symmetrise(cov), eps=self.eps)

        # Normalise by trace
        tr = batch_trace(cov_reg)  # [...]
        rho = cov_reg / tr.unsqueeze(-1).unsqueeze(-1)

        return rho

    def forward(self, returns: Tensor) -> Tensor:
        """Full pipeline: returns → rolling covariance → density matrices.

        Args:
            returns: shape [B, T, N].

        Returns:
            Density matrices, shape [B, T_valid, N, N].
        """
        covs = self.compute_rolling_covariance(returns)
        B, T_v, N, _ = covs.shape
        covs_flat = covs.reshape(B * T_v, N, N)
        rhos_flat = self.covariance_to_density(covs_flat)
        return rhos_flat.reshape(B, T_v, N, N)
