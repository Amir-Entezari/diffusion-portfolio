"""SPD covariance representations for market-return histories."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def regularized_covariance(
    history: Tensor,
    *,
    relative_eigenvalue_floor: float = 1e-6,
) -> Tensor:
    """Estimate a strictly SPD covariance matrix per history window.

    Parameters
    ----------
    history:
        Raw return histories with shape
        [batch, time, assets].

    relative_eigenvalue_floor:
        Minimum eigenvalue as a fraction of the
        mean covariance eigenvalue for each sample.

    Returns
    -------
    Tensor
        Covariance matrices with shape
        [batch, assets, assets].
    """

    if history.ndim != 3:
        raise ValueError(
            "history must have shape "
            "[batch, time, assets]"
        )

    if history.shape[1] < 2:
        raise ValueError(
            "history must contain at least "
            "two time points"
        )

    if history.shape[2] < 1:
        raise ValueError(
            "history must contain at least "
            "one asset"
        )

    if relative_eigenvalue_floor <= 0:
        raise ValueError(
            "relative_eigenvalue_floor "
            "must be positive"
        )

    if not torch.isfinite(
        history
    ).all():
        raise ValueError(
            "history contains non-finite values"
        )

    centered = (
        history
        - history.mean(
            dim=1,
            keepdim=True,
        )
    )

    covariance = (
        centered.transpose(
            1,
            2,
        )
        @ centered
    ) / (
        history.shape[1]
        - 1
    )

    covariance = 0.5 * (
        covariance
        + covariance.transpose(
            1,
            2,
        )
    )

    eigenvalues, eigenvectors = (
        torch.linalg.eigh(
            covariance
        )
    )

    mean_eigenvalue = (
        eigenvalues
        .mean(
            dim=-1,
            keepdim=True,
        )
        .clamp_min(
            torch.finfo(
                history.dtype
            ).eps
        )
    )

    floor = (
        relative_eigenvalue_floor
        * mean_eigenvalue
    )

    eigenvalues = torch.maximum(
        eigenvalues,
        floor,
    )

    covariance = (
        eigenvectors
        @ torch.diag_embed(
            eigenvalues
        )
        @ eigenvectors.transpose(
            1,
            2,
        )
    )

    return 0.5 * (
        covariance
        + covariance.transpose(
            1,
            2,
        )
    )


def symmetric_vectorize(
    matrix: Tensor,
) -> Tensor:
    """Isometric vectorization of symmetric matrices.

    Off-diagonal entries are multiplied by sqrt(2), so
    Euclidean vector distance equals Frobenius matrix distance.
    """

    if matrix.ndim != 3:
        raise ValueError(
            "matrix must have shape "
            "[batch, n, n]"
        )

    if (
        matrix.shape[1]
        != matrix.shape[2]
    ):
        raise ValueError(
            "matrix must be square"
        )

    n = matrix.shape[-1]

    row, col = torch.triu_indices(
        n,
        n,
        device=matrix.device,
    )

    values = matrix[
        :,
        row,
        col,
    ].clone()

    off_diagonal = (
        row != col
    )

    values[
        :,
        off_diagonal,
    ] *= math.sqrt(
        2.0
    )

    return values


def covariance_features(
    history: Tensor,
    *,
    relative_eigenvalue_floor: float = 1e-6,
) -> Tensor:
    """Euclidean covariance control features."""

    covariance = regularized_covariance(
        history,
        relative_eigenvalue_floor=(
            relative_eigenvalue_floor
        ),
    )

    return symmetric_vectorize(
        covariance
    )


def log_euclidean_spd_features(
    history: Tensor,
    *,
    relative_eigenvalue_floor: float = 1e-6,
) -> Tensor:
    """Log-Euclidean SPD covariance features."""

    covariance = regularized_covariance(
        history,
        relative_eigenvalue_floor=(
            relative_eigenvalue_floor
        ),
    )

    eigenvalues, eigenvectors = (
        torch.linalg.eigh(
            covariance
        )
    )

    log_covariance = (
        eigenvectors
        @ torch.diag_embed(
            torch.log(
                eigenvalues
            )
        )
        @ eigenvectors.transpose(
            1,
            2,
        )
    )

    log_covariance = 0.5 * (
        log_covariance
        + log_covariance.transpose(
            1,
            2,
        )
    )

    return symmetric_vectorize(
        log_covariance
    )