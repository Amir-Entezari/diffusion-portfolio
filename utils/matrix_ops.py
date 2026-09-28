"""
Numerically Stable Matrix Operations for Riemannian Geometry.

This module provides GPU-accelerated, numerically stable implementations
of matrix square root, inverse square root, logarithm, and exponential
operations on Symmetric Positive-Definite (SPD) matrices.  These operations
are critical for:
    - Phase 2 (§5.4): Bures metric computation via matrix sqrt.
    - Phase 2 (§5.4.3): Riemannian logarithmic map via matrix log.
    - Training stability: Preventing NaN gradients from near-singular
      covariance matrices during Black Swan events.

All operations use eigendecomposition-based implementations to guarantee:
    1. Positive eigenvalues (clamped below by ε).
    2. Gradient flow through ``torch.linalg.eigh``.
    3. Batch support: all functions accept [..., N, N] shaped tensors.

Mathematical Reference:
    Given a symmetric matrix A = Q Λ Q^T (eigendecomposition),
        A^{1/2}   = Q Λ^{1/2} Q^T
        A^{-1/2}  = Q Λ^{-1/2} Q^T
        log(A)    = Q log(Λ) Q^T
        exp(A)    = Q exp(Λ) Q^T
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _safe_eigh(
    A: Tensor,
    eigenvalue_clamp_min: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Compute eigendecomposition with numerical safeguards.

    Performs A = Q Λ Q^T via ``torch.linalg.eigh`` (guaranteed real for
    symmetric matrices) and clamps eigenvalues from below.

    Args:
        A: Symmetric matrix tensor of shape [..., N, N].
        eigenvalue_clamp_min: ε — minimum eigenvalue.  Prevents:
            - Division by zero in A^{-1/2}.
            - log(0) = -∞ in log(A).
            - Imaginary numbers from negative eigenvalues caused by
              floating-point rounding in near-singular matrices.

    Returns:
        eigenvalues: Clamped eigenvalues, shape [..., N].
        eigenvectors: Orthonormal eigenvectors Q, shape [..., N, N].

    Note:
        ``torch.linalg.eigh`` is used instead of ``torch.linalg.eig``
        because it:
            1. Guarantees real-valued outputs for symmetric inputs.
            2. Returns orthogonal Q (Q^T Q = I).
            3. Is ~2× faster than the general eigendecomposition.
    """
    # Ensure exact symmetry (eliminate floating-point asymmetry)
    A_sym = symmetrise(A)

    # Eigendecomposition: A = Q diag(λ) Q^T
    eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)

    # Clamp eigenvalues for numerical stability
    eigenvalues = eigenvalues.clamp(min=eigenvalue_clamp_min)

    return eigenvalues, eigenvectors


def stable_matrix_sqrt(
    A: Tensor,
    eigenvalue_clamp_min: float = 1e-8,
) -> Tensor:
    r"""Compute the principal matrix square root A^{1/2}.

    Given the spectral decomposition A = Q Λ Q^T, computes:

    .. math::
        A^{1/2} = Q \Lambda^{1/2} Q^T

    where Λ^{1/2} = diag(√λ₁, √λ₂, ..., √λ_N).

    This is used extensively in the Bures metric (§5.4.2):
        D_B(ρ₁, ρ₂)² = 2(1 − Tr(√(√ρ₁ · ρ₂ · √ρ₁)))

    Args:
        A: SPD matrix tensor, shape [..., N, N].
        eigenvalue_clamp_min: Minimum eigenvalue for stability.

    Returns:
        A^{1/2}: Matrix square root, shape [..., N, N].
            Guaranteed SPD if input is PSD.

    Example:
        >>> A = torch.eye(3) * 4.0
        >>> sqrt_A = stable_matrix_sqrt(A)
        >>> torch.allclose(sqrt_A, torch.eye(3) * 2.0)
        True
    """
    eigenvalues, Q = _safe_eigh(A, eigenvalue_clamp_min)

    # Λ^{1/2}
    sqrt_eigenvalues = eigenvalues.sqrt()

    # Q Λ^{1/2} Q^T
    return Q @ torch.diag_embed(sqrt_eigenvalues) @ Q.transpose(-2, -1)


def stable_matrix_inv_sqrt(
    A: Tensor,
    eigenvalue_clamp_min: float = 1e-8,
) -> Tensor:
    r"""Compute the inverse matrix square root A^{-1/2}.

    .. math::
        A^{-1/2} = Q \Lambda^{-1/2} Q^T

    where Λ^{-1/2} = diag(1/√λ₁, 1/√λ₂, ..., 1/√λ_N).

    Used in the Riemannian logarithmic map (§5.4.3):
        v_t = ρ_ref^{1/2} log(ρ_ref^{-1/2} ρ_t ρ_ref^{-1/2}) ρ_ref^{1/2}

    Args:
        A: SPD matrix tensor, shape [..., N, N].
        eigenvalue_clamp_min: Minimum eigenvalue (prevents division by zero).

    Returns:
        A^{-1/2}: Inverse square root, shape [..., N, N].
    """
    eigenvalues, Q = _safe_eigh(A, eigenvalue_clamp_min)

    # Λ^{-1/2}
    inv_sqrt_eigenvalues = eigenvalues.rsqrt()  # 1/√λ

    return Q @ torch.diag_embed(inv_sqrt_eigenvalues) @ Q.transpose(-2, -1)


def stable_matrix_log(
    A: Tensor,
    eigenvalue_clamp_min: float = 1e-8,
) -> Tensor:
    r"""Compute the principal matrix logarithm log(A).

    .. math::
        \log(A) = Q \log(\Lambda) Q^T

    where log(Λ) = diag(log λ₁, log λ₂, ..., log λ_N).

    Used in the logarithmic map on the SPD manifold (§5.4.3):
        log_{ρ_ref}(ρ_t) = ρ_ref^{1/2} log(ρ_ref^{-1/2} ρ_t ρ_ref^{-1/2}) ρ_ref^{1/2}

    Args:
        A: SPD matrix tensor, shape [..., N, N].
            Must be positive-definite (all eigenvalues > 0).
        eigenvalue_clamp_min: Minimum eigenvalue (prevents log(0) = -∞).

    Returns:
        log(A): Matrix logarithm, shape [..., N, N].
            Note: log(A) is symmetric but NOT necessarily PD.
    """
    eigenvalues, Q = _safe_eigh(A, eigenvalue_clamp_min)

    # log(Λ)
    log_eigenvalues = eigenvalues.log()

    return Q @ torch.diag_embed(log_eigenvalues) @ Q.transpose(-2, -1)


def stable_matrix_exp(
    A: Tensor,
) -> Tensor:
    r"""Compute the matrix exponential exp(A).

    .. math::
        \exp(A) = Q \exp(\Lambda) Q^T

    This is the inverse of the matrix logarithm and maps from the
    tangent space back to the SPD manifold (the exponential map).

    Args:
        A: Symmetric matrix tensor, shape [..., N, N].
            Does NOT need to be positive-definite.

    Returns:
        exp(A): Matrix exponential, shape [..., N, N].
            Guaranteed SPD for symmetric inputs.
    """
    A_sym = symmetrise(A)
    eigenvalues, Q = torch.linalg.eigh(A_sym)

    # exp(Λ)
    exp_eigenvalues = eigenvalues.exp()

    return Q @ torch.diag_embed(exp_eigenvalues) @ Q.transpose(-2, -1)


def symmetrise(A: Tensor) -> Tensor:
    """Force a matrix to be exactly symmetric: A ← (A + A^T) / 2.

    Floating-point arithmetic can introduce tiny asymmetries in matrices
    that are theoretically symmetric (e.g., covariance matrices, density
    matrices).  This function eliminates them to prevent ``eigh`` failures.

    Args:
        A: Matrix tensor, shape [..., N, N].

    Returns:
        Symmetrised matrix, shape [..., N, N].
    """
    return 0.5 * (A + A.transpose(-2, -1))


def upper_triangular_to_vector(A: Tensor) -> Tensor:
    """Flatten the upper-triangular part (incl. diagonal) of a symmetric matrix.

    For a symmetric matrix A ∈ ℝ^{N×N}, the unique information is stored
    in N(N+1)/2 elements.  This function extracts them as a 1D vector,
    suitable for feeding into standard neural network layers.

    Used after the Riemannian log map (§5.4.3) to convert the tangent
    space matrix v_t ∈ T_{ρ_ref}M to a flat vector ∈ ℝ^{N(N+1)/2}.

    Args:
        A: Symmetric matrix tensor, shape [..., N, N].

    Returns:
        Vector of upper-triangular elements, shape [..., N(N+1)/2].
    """
    N = A.shape[-1]
    # Get upper triangular indices
    row_idx, col_idx = torch.triu_indices(N, N, device=A.device)
    return A[..., row_idx, col_idx]


def vector_to_upper_triangular(v: Tensor, N: int) -> Tensor:
    """Reconstruct a symmetric matrix from its upper-triangular vector.

    Inverse of ``upper_triangular_to_vector``.

    Args:
        v: Flat vector, shape [..., N(N+1)/2].
        N: Matrix dimension.

    Returns:
        Symmetric matrix, shape [..., N, N].
    """
    batch_shape = v.shape[:-1]
    row_idx, col_idx = torch.triu_indices(N, N, device=v.device)

    A = torch.zeros(*batch_shape, N, N, device=v.device, dtype=v.dtype)
    A[..., row_idx, col_idx] = v
    A[..., col_idx, row_idx] = v  # Mirror for symmetry
    return A


def batch_trace(A: Tensor) -> Tensor:
    """Compute the trace of a batch of matrices.

    Tr(A) = Σ_i A_{ii}

    Used in:
        - Density matrix normalisation (§5.4.1): ρ = Σ / Tr(Σ)
        - Bures metric (§5.4.2): Tr(√(√ρ₁ · ρ₂ · √ρ₁))
        - Quantum fidelity computation

    Args:
        A: Matrix tensor, shape [..., N, N].

    Returns:
        Trace values, shape [...].
    """
    return torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)


def regularise_matrix(
    A: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Add Tikhonov regularisation to ensure strict positive-definiteness.

    A ← A + ε · I

    This is critical during Black Swan events when correlation matrices
    converge toward singularity (§2.2, §5.4.1).

    Args:
        A: Matrix tensor, shape [..., N, N].
        eps: Regularisation scalar.

    Returns:
        Regularised matrix, shape [..., N, N].
    """
    N = A.shape[-1]
    eye = torch.eye(N, device=A.device, dtype=A.dtype)
    # Expand identity to match batch dimensions
    for _ in range(len(A.shape) - 2):
        eye = eye.unsqueeze(0)
    eye = eye.expand_as(A)
    return A + eps * eye
