"""Complete Diffolio hierarchical noise-prediction network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from diffusion_portfolio.baselines.diffolio.hierarchy import (
    DiffolioHierarchy,
)


@dataclass(frozen=True)
class DiffolioDenoiserOutput:
    """Outputs of the complete Diffolio denoiser.

    noise:
        Predicted Gaussian noise:
        [batch, assets]

    asset_attention:
        Asset-level cross-attention:
        [batch, assets, heads, 1, lookback]

    market_attention:
        Market-level self-attention:
        [batch, heads, assets + systematic,
         assets + systematic]
    """

    noise: torch.Tensor
    asset_attention: torch.Tensor
    market_attention: torch.Tensor


class DiffolioDenoiser(nn.Module):
    """Hierarchical Diffolio epsilon-prediction network."""

    def __init__(
        self,
        *,
        n_assets: int = 12,
        n_asset_characteristics: int = 10,
        n_systematic: int = 8,
        lookback: int = 63,
        hidden_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        time_embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        self.n_assets = n_assets

        self.hierarchy = DiffolioHierarchy(
            n_assets=n_assets,
            n_asset_characteristics=(
                n_asset_characteristics
            ),
            n_systematic=n_systematic,
            lookback=lookback,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            time_embedding_dim=(
                time_embedding_dim
            ),
        )

        # One decoder shared across all assets.
        self.decoder = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        return_history: torch.Tensor,
        asset_covariates: torch.Tensor,
        systematic_covariates: torch.Tensor,
    ) -> DiffolioDenoiserOutput:
        hierarchy = self.hierarchy(
            noisy_target,
            timesteps,
            return_history,
            asset_covariates,
            systematic_covariates,
        )

        # [B, N, D]
        asset_latents = (
            hierarchy.asset_latents
        )

        # Shared Linear(D -> 1) applied independently
        # to every asset token.
        noise = self.decoder(
            asset_latents
        ).squeeze(
            -1
        )
        # [B, N]

        if noise.shape != (
            noisy_target.shape[0],
            self.n_assets,
        ):
            raise RuntimeError(
                "Decoded noise has unexpected shape"
            )

        return DiffolioDenoiserOutput(
            noise=noise,
            asset_attention=(
                hierarchy.asset_attention
            ),
            market_attention=(
                hierarchy.market_attention
            ),
        )