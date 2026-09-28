"""
SGW-Guided Cross-Attention for Manifold Fusion.

Reference: Proposal §5.5.3 — SGW Cross-Attention.

    c_unified = Σ_k (Softmax(Q_T · K_v^T / √d_k) ⊙ π*) V_v

where:
    - Q_T: queries from topology (Phase 1)
    - K_v, V_v: keys/values from quantum geometry (Phase 2)
    - π*: SGW transport plan (geometric truth mask)
    - ⊙: Hadamard product
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class SGWCrossAttention(nn.Module):
    r"""SGW-guided cross-attention for topology-quantum fusion.

    .. math::
        c_{unified} = \text{Softmax}\!\left(\frac{Q_\mathcal{T} K_v^\top}{\sqrt{d_k}}\right) V_v

    Args:
        topo_dim: Input dimension of topological features.
        quantum_dim: Input dimension of quantum features.
        n_heads: Number of attention heads.
        head_dim: Per-head dimension d_k.
        output_dim: Dimension of c_unified.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        topo_dim: int = 128,
        quantum_dim: int = 78,
        n_heads: int = 8,
        head_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)

        inner_dim = n_heads * head_dim

        # Query from topology
        self.W_q = nn.Linear(topo_dim, inner_dim, bias=False)
        # Key, Value from quantum
        self.W_k = nn.Linear(quantum_dim, inner_dim, bias=False)
        self.W_v = nn.Linear(quantum_dim, inner_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(inner_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        topo_features: Tensor,
        quantum_features: Tensor,
        transport_plan: Tensor | None = None,
    ) -> Tensor:
        """Compute SGW-guided cross-attention.

        Args:
            topo_features: T_t, shape [B, D_topo].
            quantum_features: v_t, shape [B, D_quantum].
            transport_plan: π*, shape [B, H_t, H_q] (optional mask).

        Returns:
            c_unified: Unified conditioning tensor, shape [B, output_dim].
        """
        B = topo_features.shape[0]

        # Expand to sequence dim for attention: [B, 1, D]
        Q = self.W_q(topo_features).unsqueeze(1)   # [B, 1, inner]
        K = self.W_k(quantum_features).unsqueeze(1)  # [B, 1, inner]
        V = self.W_v(quantum_features).unsqueeze(1)  # [B, 1, inner]

        # Reshape for multi-head: [B, n_heads, 1, head_dim]
        Q = Q.reshape(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, n_heads, 1, 1]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted values: [B, n_heads, 1, head_dim]
        out = torch.matmul(attn, V)

        # Reshape: [B, 1, inner_dim] → [B, inner_dim]
        out = out.transpose(1, 2).reshape(B, -1)

        # Output projection + residual-like layer norm
        c_unified = self.layer_norm(self.W_o(out))

        return c_unified


class SGWFusionModule(nn.Module):
    """Complete SGW Interface: SGW distance + cross-attention fusion.

    Combines SlicedGromovWasserstein and SGWCrossAttention into
    a single module that produces c_unified from Phase 1 and Phase 2 outputs.

    Args:
        topo_dim: Topological feature dimension (from PersistenceLandscapeLayer).
        quantum_dim: Quantum feature dimension (N(N+1)/2 from log map).
        n_projections: SGW projection count.
        n_heads: Cross-attention heads.
        output_dim: c_unified dimension.
    """

    def __init__(
        self,
        topo_dim: int = 128,
        quantum_dim: int = 78,
        n_projections: int = 50,
        projection_dim: int = 64,
        n_heads: int = 8,
        head_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        from models.sgw_interface.sliced_gw import SlicedGromovWasserstein

        self.sgw = SlicedGromovWasserstein(
            n_projections=n_projections,
            projection_dim=projection_dim,
            topo_input_dim=topo_dim,
            quantum_input_dim=quantum_dim,
        )
        self.cross_attention = SGWCrossAttention(
            topo_dim=topo_dim,
            quantum_dim=quantum_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        topo_features: Tensor,
        quantum_features: Tensor,
    ) -> dict[str, Tensor]:
        """Compute SGW distance and fused conditioning tensor.

        Args:
            topo_features: T_t, shape [B, D_topo].
            quantum_features: v_t, shape [B, D_quantum].

        Returns:
            Dict with:
                - c_unified: shape [B, output_dim]
                - sgw_distance: shape [B]
                - transport_plan: shape [B, D_proj, D_proj]
        """
        sgw_dist, transport = self.sgw(topo_features, quantum_features)
        c_unified = self.cross_attention(
            topo_features, quantum_features, transport
        )
        return {
            "c_unified": c_unified,
            "sgw_distance": sgw_dist,
            "transport_plan": transport,
        }
