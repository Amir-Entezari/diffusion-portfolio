"""
Bures Metric on SPD Manifolds.

Reference: Proposal §5.4.2 — Bures Distance.

    D_B(ρ₁, ρ₂)² = 2(1 − Tr(√(√ρ₁ · ρ₂ · √ρ₁)))

Key properties:
    - D_B ∈ [0, √2]: bounded distance
    - No matrix inversion required → immune to singularity
    - Metrises the topology of quantum states
"""

from __future__ import annotations

import torch
from torch import Tensor

from utils.matrix_ops import stable_matrix_sqrt, batch_trace, symmetrise


def bures_distance_squared(
    rho1: Tensor,
    rho2: Tensor,
    eigenvalue_clamp_min: float = 1e-8,
) -> Tensor:
    r"""Compute the squared Bures distance.

    .. math::
        D_B(\rho_1, \rho_2)^2 = 2\left(1 - \text{Tr}\sqrt{\sqrt{\rho_1}\,\rho_2\,\sqrt{\rho_1}}\right)

    Args:
        rho1: First density matrix, shape [..., N, N].
        rho2: Second density matrix, shape [..., N, N].
        eigenvalue_clamp_min: Minimum eigenvalue for numerical stability.

    Returns:
        Squared Bures distance, shape [...].
    """
    sqrt_rho1 = stable_matrix_sqrt(rho1, eigenvalue_clamp_min)
    inner = sqrt_rho1 @ rho2 @ sqrt_rho1
    inner = symmetrise(inner)
    sqrt_inner = stable_matrix_sqrt(inner, eigenvalue_clamp_min)
    fidelity = batch_trace(sqrt_inner)
    d_sq = 2.0 * (1.0 - fidelity)
    return torch.clamp(d_sq, min=0.0)


def bures_distance(
    rho1: Tensor,
    rho2: Tensor,
    eigenvalue_clamp_min: float = 1e-8,
) -> Tensor:
    """Compute the Bures distance D_B(ρ₁, ρ₂).

    Args:
        rho1, rho2: Density matrices, shape [..., N, N].

    Returns:
        Bures distance, shape [...]. In [0, √2].
    """
    return torch.sqrt(bures_distance_squared(rho1, rho2, eigenvalue_clamp_min))


def quantum_fidelity(
    rho1: Tensor,
    rho2: Tensor,
    eigenvalue_clamp_min: float = 1e-8,
) -> Tensor:
    r"""Compute the quantum fidelity F(ρ₁, ρ₂).

    .. math::
        F(\rho_1, \rho_2) = \left(\text{Tr}\sqrt{\sqrt{\rho_1}\,\rho_2\,\sqrt{\rho_1}}\right)^2

    F = 1 iff ρ₁ = ρ₂.  F = 0 iff ρ₁ ⊥ ρ₂.

    Args:
        rho1, rho2: Density matrices, shape [..., N, N].

    Returns:
        Fidelity, shape [...]. In [0, 1].
    """
    sqrt_rho1 = stable_matrix_sqrt(rho1, eigenvalue_clamp_min)
    inner = sqrt_rho1 @ rho2 @ sqrt_rho1
    inner = symmetrise(inner)
    sqrt_inner = stable_matrix_sqrt(inner, eigenvalue_clamp_min)
    tr = batch_trace(sqrt_inner)
    return torch.clamp(tr ** 2, 0.0, 1.0)
