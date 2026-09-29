"""Embedding and hierarchical conditioning network for Diffolio."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from diffusion_portfolio.models.diffolio.attention import (
    CrossAttentionBlock,
    SelfAttentionBlock,
)


@dataclass(frozen=True)
class DiffolioHierarchyOutput:
    """Outputs of the two-stage Diffolio hierarchy.

    asset_latents:
        Final market-integrated asset representations:
        [batch, assets, hidden_dim]

    systematic_latents:
        Final market-integrated systematic representations:
        [batch, systematic_features, hidden_dim]

    asset_attention:
        Asset-level cross-attention probabilities:
        [batch, assets, heads, 1, lookback]

    market_attention:
        Market-level self-attention probabilities:
        [batch, heads, assets + systematic_features,
         assets + systematic_features]
    """

    asset_latents: torch.Tensor
    systematic_latents: torch.Tensor
    asset_attention: torch.Tensor
    market_attention: torch.Tensor


class SinusoidalDiffusionEmbedding(nn.Module):
    """Fixed sinusoidal embedding for diffusion step tau."""

    def __init__(
        self,
        *,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if embedding_dim % 2 != 0:
            raise ValueError(
                "embedding_dim must be even"
            )

        self.embedding_dim = embedding_dim

    def forward(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [batch]"
            )

        half_dim = (
            self.embedding_dim
            // 2
        )

        device = timesteps.device

        frequencies = torch.exp(
            -math.log(
                10000.0
            )
            * torch.arange(
                half_dim,
                device=device,
                dtype=torch.float32,
            )
            / max(
                half_dim - 1,
                1,
            )
        )

        arguments = (
            timesteps.float()[
                :,
                None,
            ]
            * frequencies[
                None,
                :,
            ]
        )

        return torch.cat(
            (
                torch.sin(
                    arguments
                ),
                torch.cos(
                    arguments
                ),
            ),
            dim=-1,
        )


class DiffolioHierarchy(nn.Module):
    """Two-stage hierarchical conditioning network from Diffolio.

    Stage 1
    -------
    For every asset independently:

        query:
            noisy future return
            + diffusion-step embedding

        context:
            historical return
            + asset characteristics

        -> shared CrossAttentionBlock

    Stage 2
    -------
    Concatenate:

        N asset latents
        +
        N_y systematic-history embeddings

    then apply a shared market-level SelfAttentionBlock.
    """

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

        if n_assets <= 0:
            raise ValueError(
                "n_assets must be positive"
            )

        if n_asset_characteristics <= 0:
            raise ValueError(
                "n_asset_characteristics must be positive"
            )

        if n_systematic <= 0:
            raise ValueError(
                "n_systematic must be positive"
            )

        if lookback <= 0:
            raise ValueError(
                "lookback must be positive"
            )

        self.n_assets = n_assets
        self.n_asset_characteristics = (
            n_asset_characteristics
        )
        self.n_systematic = n_systematic
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = (
            time_embedding_dim
        )

        # -----------------------------------------------------
        # Diffusion-step embedding.
        # -----------------------------------------------------
        self.time_embedding = (
            SinusoidalDiffusionEmbedding(
                embedding_dim=time_embedding_dim
            )
        )

        # -----------------------------------------------------
        # Stage 1 shared embeddings.
        #
        # Query input:
        #     one noisy return
        #     +
        #     diffusion-step embedding
        # -----------------------------------------------------
        self.query_embedding = nn.Linear(
            1 + time_embedding_dim,
            hidden_dim,
        )

        # Historical context input per asset/day:
        #
        #     return
        #     +
        #     asset characteristics
        self.asset_context_embedding = (
            nn.Linear(
                1
                + n_asset_characteristics,
                hidden_dim,
            )
        )

        self.asset_attention = (
            CrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
            )
        )

        # -----------------------------------------------------
        # Stage 2 systematic embedding.
        #
        # Each systematic variable has one M-day sequence:
        #
        #     [lookback]
        #
        # which becomes one D-dimensional token.
        #
        # The same linear layer is shared over all systematic
        # variables.
        # -----------------------------------------------------
        self.systematic_embedding = nn.Linear(
            lookback,
            hidden_dim,
        )

        self.market_attention = (
            SelfAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
            )
        )

    def _validate_inputs(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        return_history: torch.Tensor,
        asset_covariates: torch.Tensor,
        systematic_covariates: torch.Tensor,
    ) -> None:
        if noisy_target.ndim != 2:
            raise ValueError(
                "noisy_target must have shape "
                "[batch, assets]"
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [batch]"
            )

        if return_history.ndim != 3:
            raise ValueError(
                "return_history must have shape "
                "[batch, lookback, assets]"
            )

        if asset_covariates.ndim != 4:
            raise ValueError(
                "asset_covariates must have shape "
                "[batch, lookback, assets, characteristics]"
            )

        if systematic_covariates.ndim != 3:
            raise ValueError(
                "systematic_covariates must have shape "
                "[batch, lookback, systematic_features]"
            )

        batch_size = noisy_target.shape[
            0
        ]

        if noisy_target.shape != (
            batch_size,
            self.n_assets,
        ):
            raise ValueError(
                "noisy_target shape does not match "
                "configured asset count"
            )

        if timesteps.shape[0] != batch_size:
            raise ValueError(
                "timestep batch size differs"
            )

        if return_history.shape != (
            batch_size,
            self.lookback,
            self.n_assets,
        ):
            raise ValueError(
                "return_history shape does not match "
                "configured dimensions"
            )

        if asset_covariates.shape != (
            batch_size,
            self.lookback,
            self.n_assets,
            self.n_asset_characteristics,
        ):
            raise ValueError(
                "asset_covariates shape does not match "
                "configured dimensions"
            )

        if systematic_covariates.shape != (
            batch_size,
            self.lookback,
            self.n_systematic,
        ):
            raise ValueError(
                "systematic_covariates shape does not match "
                "configured dimensions"
            )

    def forward(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        return_history: torch.Tensor,
        asset_covariates: torch.Tensor,
        systematic_covariates: torch.Tensor,
    ) -> DiffolioHierarchyOutput:
        self._validate_inputs(
            noisy_target,
            timesteps,
            return_history,
            asset_covariates,
            systematic_covariates,
        )

        batch_size = noisy_target.shape[
            0
        ]

        # =====================================================
        # Stage 1: asset-level hierarchy
        # =====================================================

        time_embedding = self.time_embedding(
            timesteps
        )

        # Same diffusion timestep applies to every asset.
        expanded_time = (
            time_embedding[
                :,
                None,
                :,
            ]
            .expand(
                -1,
                self.n_assets,
                -1,
            )
        )

        noisy_feature = noisy_target[
            :,
            :,
            None,
        ]

        query_input = torch.cat(
            (
                noisy_feature,
                expanded_time,
            ),
            dim=-1,
        )

        # Shared linear layer over every asset.
        queries = self.query_embedding(
            query_input
        )
        # [B, N, D]

        historical_returns = (
            return_history[
                :,
                :,
                :,
                None,
            ]
        )

        context_input = torch.cat(
            (
                historical_returns,
                asset_covariates,
            ),
            dim=-1,
        )
        # [B, M, N, 1+C]

        contexts = (
            self.asset_context_embedding(
                context_input
            )
        )
        # [B, M, N, D]

        # -----------------------------------------------------
        # The paper processes each asset independently with
        # shared parameters.
        #
        # Flatten B and N so one CrossAttentionBlock call
        # performs exactly those independent operations.
        # -----------------------------------------------------
        flat_queries = (
            queries.reshape(
                batch_size
                * self.n_assets,
                self.hidden_dim,
            )
        )

        flat_context = (
            contexts.permute(
                0,
                2,
                1,
                3,
            )
            .reshape(
                batch_size
                * self.n_assets,
                self.lookback,
                self.hidden_dim,
            )
        )

        asset_output = self.asset_attention(
            flat_queries,
            flat_context,
            flat_context,
        )

        asset_latents = (
            asset_output.hidden.reshape(
                batch_size,
                self.n_assets,
                self.hidden_dim,
            )
        )

        asset_attention = (
            asset_output.attention.reshape(
                batch_size,
                self.n_assets,
                -1,
                1,
                self.lookback,
            )
        )

        # =====================================================
        # Stage 2: market-level hierarchy
        # =====================================================

        # Each systematic variable becomes one market token.
        #
        # [B, M, Ny]
        # ->
        # [B, Ny, M]
        systematic_sequences = (
            systematic_covariates.transpose(
                1,
                2,
            )
        )

        systematic_latents = (
            self.systematic_embedding(
                systematic_sequences
            )
        )
        # [B, Ny, D]

        market_tokens = torch.cat(
            (
                asset_latents,
                systematic_latents,
            ),
            dim=1,
        )
        # [B, N + Ny, D]

        market_output = (
            self.market_attention(
                market_tokens
            )
        )

        final_asset_latents = (
            market_output.hidden[
                :,
                : self.n_assets,
                :,
            ]
        )

        final_systematic_latents = (
            market_output.hidden[
                :,
                self.n_assets :,
                :,
            ]
        )

        return DiffolioHierarchyOutput(
            asset_latents=final_asset_latents,
            systematic_latents=(
                final_systematic_latents
            ),
            asset_attention=asset_attention,
            market_attention=(
                market_output.attention
            ),
        )