"""Correlation targets for Diffolio's structural regularizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShrinkageCorrelationTarget:
    """Batch of shrinkage covariance/correlation estimates.

    covariance:
        [batch, assets, assets]

    correlation:
        [batch, assets, assets]

    shrinkage:
        Ledoit-Wolf shrinkage intensity for each sample:
        [batch]
    """

    covariance: torch.Tensor
    correlation: torch.Tensor
    shrinkage: torch.Tensor


def compute_training_covariance(
    returns: torch.Tensor,
) -> torch.Tensor:
    """Compute the fixed covariance target from training returns.

    Parameters
    ----------
    returns:
        Raw excess returns from the TRAINING period only,
        shape [time, assets].

    Notes
    -----
    We follow the Ledoit-Wolf convention of using the
    maximum-likelihood covariance denominator T after
    demeaning, rather than T-1.
    """

    if returns.ndim != 2:
        raise ValueError(
            "returns must have shape [time, assets]"
        )

    if returns.shape[0] < 2:
        raise ValueError(
            "At least two training observations are required"
        )

    if not torch.isfinite(
        returns
    ).all():
        raise ValueError(
            "returns contain NaN or infinite values"
        )

    original_dtype = returns.dtype

    work = returns.to(
        dtype=torch.float64
    )

    centered = (
        work
        - work.mean(
            dim=0,
            keepdim=True,
        )
    )

    n_observations = (
        centered.shape[0]
    )

    covariance = (
        centered.transpose(
            0,
            1
        )
        @ centered
    ) / n_observations

    return covariance.to(
        dtype=original_dtype
    )


def covariance_to_correlation(
    covariance: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert covariance matrices to correlation matrices."""

    if covariance.ndim < 2:
        raise ValueError(
            "covariance must have at least two dimensions"
        )

    if (
        covariance.shape[-1]
        != covariance.shape[-2]
    ):
        raise ValueError(
            "covariance matrices must be square"
        )

    if not torch.isfinite(
        covariance
    ).all():
        raise ValueError(
            "covariance contains NaN or infinite values"
        )

    variances = torch.diagonal(
        covariance,
        dim1=-2,
        dim2=-1,
    )

    if torch.any(
        variances <= 0.0
    ):
        raise ValueError(
            "covariance diagonal must be positive"
        )

    standard_deviations = torch.sqrt(
        torch.clamp(
            variances,
            min=eps,
        )
    )

    denominator = (
        standard_deviations.unsqueeze(
            -1
        )
        * standard_deviations.unsqueeze(
            -2
        )
    )

    correlation = (
        covariance
        / denominator
    )

    # Numerical roundoff can produce values slightly
    # beyond the valid correlation interval.
    correlation = torch.clamp(
        correlation,
        min=-1.0,
        max=1.0,
    )

    # Enforce an exact unit diagonal.
    n_assets = correlation.shape[
        -1
    ]

    identity = torch.eye(
        n_assets,
        dtype=correlation.dtype,
        device=correlation.device,
    )

    correlation = (
        correlation
        * (
            1.0
            - identity
        )
        + identity
    )

    return correlation


def estimate_shrinkage_correlation(
    recent_returns: torch.Tensor,
    training_covariance: torch.Tensor,
    *,
    eps: float = 1e-18,
) -> ShrinkageCorrelationTarget:
    """Estimate Diffolio's time-varying target correlation.

    Parameters
    ----------
    recent_returns:
        Raw excess-return histories with shape

            [batch, lookback, assets]

        For Diffolio, lookback=63.

    training_covariance:
        Fixed covariance matrix estimated from the entire
        training period only:

            [assets, assets]

    Method
    ------
    Let S_t be the covariance from the recent window and
    F = Sigma_train the fixed training covariance.

    We estimate

        Sigma_hat_t
          = delta_t F
            + (1 - delta_t) S_t

    where delta_t is the fixed-target Ledoit-Wolf
    shrinkage intensity.

    The resulting covariance is converted to a correlation
    matrix for use in Diffolio's correlation-guided loss.
    """

    if recent_returns.ndim != 3:
        raise ValueError(
            "recent_returns must have shape "
            "[batch, lookback, assets]"
        )

    batch_size, lookback, n_assets = (
        recent_returns.shape
    )

    if lookback < 2:
        raise ValueError(
            "At least two observations are required"
        )

    if training_covariance.shape != (
        n_assets,
        n_assets,
    ):
        raise ValueError(
            "training_covariance shape does not match "
            "asset dimension"
        )

    if not torch.isfinite(
        recent_returns
    ).all():
        raise ValueError(
            "recent_returns contain NaN or infinite values"
        )

    if not torch.isfinite(
        training_covariance
    ).all():
        raise ValueError(
            "training_covariance contains NaN or infinite values"
        )

    original_dtype = recent_returns.dtype

    # Correlation targets are supervision, not trainable
    # quantities. Float64 is inexpensive here and improves
    # numerical stability of the covariance calculations.
    x = recent_returns.detach().to(
        dtype=torch.float64
    )

    target = (
        training_covariance
        .detach()
        .to(
            device=x.device,
            dtype=torch.float64,
        )
    )

    centered = (
        x
        - x.mean(
            dim=1,
            keepdim=True,
        )
    )

    # Ledoit-Wolf uses the ML covariance convention,
    # denominator M after demeaning.
    sample_covariance = torch.einsum(
        "bti,btj->bij",
        centered,
        centered,
    ) / lookback

    # Estimate total sampling variance pi.
    #
    # (x_i x_j)^2 = x_i^2 x_j^2, so we avoid materializing
    # [B, M, N, N].
    squared = centered.square()

    second_moment = torch.einsum(
        "bti,btj->bij",
        squared,
        squared,
    ) / lookback

    pi_matrix = (
        second_moment
        - sample_covariance.square()
    )

    pi_hat = (
        pi_matrix.sum(
            dim=(-2, -1)
        )
        .clamp_min(
            0.0
        )
    )

    target_batch = (
        target.unsqueeze(
            0
        )
        .expand(
            batch_size,
            -1,
            -1,
        )
    )

    gamma_hat = (
        (
            sample_covariance
            - target_batch
        )
        .square()
        .sum(
            dim=(-2, -1)
        )
    )

    # Fixed-target Ledoit-Wolf form.
    #
    # Since the shrinkage target is precomputed and fixed
    # rather than estimated from the same 63-day window,
    # rho_hat = 0.
    raw_shrinkage = (
        pi_hat
        / (
            gamma_hat.clamp_min(
                eps
            )
            * lookback
        )
    )

    shrinkage = torch.clamp(
        raw_shrinkage,
        min=0.0,
        max=1.0,
    )

    # If S and F are effectively identical, gamma -> 0.
    # In that limit either matrix is equivalent; choose the
    # stable target explicitly.
    shrinkage = torch.where(
        gamma_hat <= eps,
        torch.ones_like(
            shrinkage
        ),
        shrinkage,
    )

    covariance = (
        shrinkage[
            :,
            None,
            None,
        ]
        * target_batch
        + (
            1.0
            - shrinkage[
                :,
                None,
                None,
            ]
        )
        * sample_covariance
    )

    correlation = (
        covariance_to_correlation(
            covariance,
            eps=eps,
        )
    )

    return ShrinkageCorrelationTarget(
        covariance=covariance.to(
            dtype=original_dtype
        ),
        correlation=correlation.to(
            dtype=original_dtype
        ),
        shrinkage=shrinkage.to(
            dtype=original_dtype
        ),
    )