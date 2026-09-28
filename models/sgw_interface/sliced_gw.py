"""
Sliced Gromov-Wasserstein Distance.

Reference: Proposal §5.5.2 — Sliced Gromov-Wasserstein (SGW).

    SGW(T_t, v_t) = (1/L) Σ_l GW_1D(⟨T_t,θ_l⟩, ⟨v_t,θ_l⟩)

where θ_l are random directions on S^{d-1} and GW_1D is solved
via sorting in O(N log N).
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


def _gw_1d(x: Tensor, y: Tensor) -> Tensor:
    """Solve 1D Gromov-Wasserstein via sorting.

    For 1D distributions, GW reduces to comparing sorted pairwise
    distance matrices, solvable in O(N log N).

    Args:
        x: 1D projected points, shape [N].
        y: 1D projected points, shape [M].

    Returns:
        1D GW distance (scalar).
    """
    x_sorted = torch.sort(x)[0]
    y_sorted = torch.sort(y)[0]

    N = len(x_sorted)
    M = len(y_sorted)

    # Pairwise distances within each set
    dx = (x_sorted.unsqueeze(1) - x_sorted.unsqueeze(0)).abs()  # [N, N]
    dy = (y_sorted.unsqueeze(1) - y_sorted.unsqueeze(0)).abs()  # [M, M]

    # For equal-size case, optimal coupling is identity (sorted matching)
    if N == M:
        cost = ((dx - dy) ** 2).mean()
    else:
        # Interpolate to common size
        K = min(N, M)
        idx_x = torch.linspace(0, N - 1, K, device=x.device).long()
        idx_y = torch.linspace(0, M - 1, K, device=y.device).long()
        dx_sub = dx[idx_x][:, idx_x]
        dy_sub = dy[idx_y][:, idx_y]
        cost = ((dx_sub - dy_sub) ** 2).mean()

    return cost


class SlicedGromovWasserstein(nn.Module):
    r"""Sliced Gromov-Wasserstein distance.

    .. math::
        SGW(\mathcal{T}_t, v_t) \approx \frac{1}{L}\sum_{l=1}^{L}
        GW_{1D}(\langle \mathcal{T}_t, \theta_l \rangle,
                 \langle v_t, \theta_l \rangle)

    Args:
        n_projections: L — number of random projection directions.
        projection_dim: Shared dimension for projecting both inputs.
    """

    def __init__(
        self,
        n_projections: int = 50,
        projection_dim: int = 64,
        topo_input_dim: int = 128,
        quantum_input_dim: int = 78,
    ) -> None:
        super().__init__()
        self.n_projections = n_projections
        self.projection_dim = projection_dim

        # Learnable projections to shared dimension
        self.topo_proj = nn.Linear(topo_input_dim, projection_dim)
        self.quantum_proj = nn.Linear(quantum_input_dim, projection_dim)

    def _sample_projections(self, dim: int, device: torch.device) -> Tensor:
        """Sample random unit vectors on S^{d-1}.

        Args:
            dim: Ambient dimension.
            device: Target device.

        Returns:
            Random directions, shape [L, dim].
        """
        theta = torch.randn(self.n_projections, dim, device=device)
        theta = theta / theta.norm(dim=1, keepdim=True)
        return theta

    def forward(
        self,
        topo_features: Tensor,
        quantum_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute SGW distance and transport plan.

        Args:
            topo_features: T_t from Phase 1, shape [B, D_topo].
            quantum_features: v_t from Phase 2, shape [B, D_quantum].

        Returns:
            Tuple of:
                - sgw_distance: SGW distance, shape [B].
                - transport_weights: Soft coupling, shape [B, D_proj, D_proj].
        """
        # Project to shared dimension
        T_proj = self.topo_proj(topo_features)    # [B, D_proj]
        v_proj = self.quantum_proj(quantum_features)  # [B, D_proj]

        B = T_proj.shape[0]
        device = T_proj.device

        # Sample random projections
        theta = self._sample_projections(self.projection_dim, device)  # [L, D_proj]

        # Project onto each direction: [B, L]
        T_sliced = T_proj @ theta.T  # [B, L]
        v_sliced = v_proj @ theta.T  # [B, L]

        # Compute 1D GW for each batch and average over projections
        sgw_dists = torch.zeros(B, device=device)
        for b in range(B):
            total = torch.tensor(0.0, device=device)
            for l in range(self.n_projections):
                total = total + _gw_1d(
                    T_sliced[b:b+1, l].expand(self.projection_dim),
                    v_sliced[b:b+1, l].expand(self.projection_dim),
                )
            sgw_dists[b] = total / self.n_projections

        # Soft transport plan via cosine similarity as proxy
        # True optimal coupling from SGW is expensive; this soft version
        # captures the alignment structure for cross-attention masking
        T_norm = T_proj / (T_proj.norm(dim=-1, keepdim=True) + 1e-8)
        v_norm = v_proj / (v_proj.norm(dim=-1, keepdim=True) + 1e-8)
        transport = torch.bmm(
            T_norm.unsqueeze(-1), v_norm.unsqueeze(-2)
        )  # [B, D_proj, D_proj] — but these are 1D, so it's outer product
        transport = torch.softmax(transport.reshape(B, -1), dim=-1)
        transport = transport.reshape(B, self.projection_dim, self.projection_dim)

        return sgw_dists, transport
