"""Correlation-guided attention regularization for Diffolio."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_asset_attention(
    market_attention: torch.Tensor,
    *,
    n_assets: int,
) -> torch.Tensor:
    """Extract Diffolio's asset-to-asset attention matrix.

    Parameters
    ----------
    market_attention:
        Per-head market-level attention probabilities:

            [batch, heads, tokens, tokens]

        where the first ``n_assets`` tokens correspond to
        assets and the remaining tokens correspond to
        systematic covariates.

    n_assets:
        Number of asset tokens.

    Returns
    -------
    torch.Tensor
        Head-averaged asset-to-asset attention:

            [batch, n_assets, n_assets]

    Notes
    -----
    The Diffolio paper defines a single N x N attention
    matrix A but does not specify how multiple attention
    heads are aggregated.

    Our reproduction convention is to average the
    probability matrices across heads before slicing out
    the asset-to-asset block.

    We intentionally do NOT renormalize the asset block,
    because attention allocated from asset queries to
    systematic tokens should remain excluded rather than
    redistributed over the assets.
    """

    if market_attention.ndim != 4:
        raise ValueError(
            "market_attention must have shape "
            "[batch, heads, tokens, tokens]"
        )

    if n_assets <= 0:
        raise ValueError(
            "n_assets must be positive"
        )

    if (
        market_attention.shape[-1]
        != market_attention.shape[-2]
    ):
        raise ValueError(
            "market attention matrices must be square"
        )

    n_tokens = market_attention.shape[
        -1
    ]

    if n_assets > n_tokens:
        raise ValueError(
            "n_assets cannot exceed token count"
        )

    if not torch.isfinite(
        market_attention
    ).all():
        raise ValueError(
            "market_attention contains NaN or infinite values"
        )

    # [B, H, T, T] -> [B, T, T]
    attention = market_attention.mean(
        dim=1
    )

    # Assets are the first N tokens in Diffolio's
    # market-level hierarchy.
    return attention[
        :,
        :n_assets,
        :n_assets,
    ]


def correlation_guided_loss(
    asset_attention: torch.Tensor,
    target_correlation: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Diffolio's correlation-guided regularizer.

    Implements Eq. (4.4):

        L_corr
          = -(1/N) sum_i
              cos(A_i, Sigma_target_i)

    and averages the result across the batch.

    Parameters
    ----------
    asset_attention:
        Asset-to-asset attention matrix:

            [batch, assets, assets]

    target_correlation:
        Shrinkage target correlation matrix:

            [batch, assets, assets]

    Returns
    -------
    torch.Tensor
        Scalar correlation-guided loss.
    """

    if asset_attention.ndim != 3:
        raise ValueError(
            "asset_attention must have shape "
            "[batch, assets, assets]"
        )

    if target_correlation.ndim != 3:
        raise ValueError(
            "target_correlation must have shape "
            "[batch, assets, assets]"
        )

    if (
        asset_attention.shape
        != target_correlation.shape
    ):
        raise ValueError(
            "attention and target correlation shapes must match"
        )

    if (
        asset_attention.shape[-1]
        != asset_attention.shape[-2]
    ):
        raise ValueError(
            "asset attention matrices must be square"
        )

    if not torch.isfinite(
        asset_attention
    ).all():
        raise ValueError(
            "asset_attention contains NaN or infinite values"
        )

    if not torch.isfinite(
        target_correlation
    ).all():
        raise ValueError(
            "target_correlation contains NaN or infinite values"
        )

    # Target correlations are supervision only.
    target = target_correlation.detach().to(
        device=asset_attention.device,
        dtype=asset_attention.dtype,
    )

    # Cosine similarity along each matrix row.
    #
    # Shape:
    #   [B, N, N]
    #      ↓ dim=-1
    #   [B, N]
    row_similarity = F.cosine_similarity(
        asset_attention,
        target,
        dim=-1,
        eps=eps,
    )

    # Equation 4.4 averages over rows.
    #
    # We additionally average across minibatch samples.
    return -row_similarity.mean()


def diffolio_correlation_regularizer(
    market_attention: torch.Tensor,
    target_correlation: torch.Tensor,
    *,
    n_assets: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convenience wrapper from raw market attention to L_corr."""

    asset_attention = extract_asset_attention(
        market_attention,
        n_assets=n_assets,
    )

    return correlation_guided_loss(
        asset_attention,
        target_correlation,
        eps=eps,
    )