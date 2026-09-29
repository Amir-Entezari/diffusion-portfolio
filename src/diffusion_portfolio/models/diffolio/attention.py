"""Attention building blocks for the Diffolio denoising network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AttentionBlockOutput:
    """Output of an attention block.

    hidden:
        Refined latent representation.

    attention:
        Per-head attention probabilities with shape
        [batch, heads, query_tokens, key_tokens].
    """

    hidden: torch.Tensor
    attention: torch.Tensor


class FeedForward(nn.Module):
    """Two-layer GELU MLP used inside Diffolio attention blocks."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        mlp_dim: int,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        if mlp_dim <= 0:
            raise ValueError(
                "mlp_dim must be positive"
            )

        self.net = nn.Sequential(
            nn.Linear(
                hidden_dim,
                mlp_dim,
            ),
            nn.GELU(),
            nn.Linear(
                mlp_dim,
                hidden_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            x
        )


class CrossAttentionBlock(nn.Module):
    """Diffolio asset-level cross-attention block.

    Query
    -----
    One latent query per asset:

        [batch, hidden_dim]

    Keys / values
    -------------
    Historical asset context:

        [batch, lookback, hidden_dim]

    Output
    ------
    Asset latent:

        [batch, hidden_dim]
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible "
                "by num_heads"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.mlp = FeedForward(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> AttentionBlockOutput:
        if query.ndim != 2:
            raise ValueError(
                "query must have shape "
                "[batch, hidden_dim]"
            )

        if key.ndim != 3:
            raise ValueError(
                "key must have shape "
                "[batch, lookback, hidden_dim]"
            )

        if value.ndim != 3:
            raise ValueError(
                "value must have shape "
                "[batch, lookback, hidden_dim]"
            )

        if key.shape != value.shape:
            raise ValueError(
                "key and value shapes must match"
            )

        if query.shape[0] != key.shape[0]:
            raise ValueError(
                "query/key batch dimensions differ"
            )

        if query.shape[-1] != self.hidden_dim:
            raise ValueError(
                "query hidden dimension differs"
            )

        if key.shape[-1] != self.hidden_dim:
            raise ValueError(
                "key hidden dimension differs"
            )

        # MultiheadAttention expects token dimension.
        query_tokens = query.unsqueeze(
            1
        )

        attended, weights = self.attention(
            query=query_tokens,
            key=key,
            value=value,
            need_weights=True,
            average_attn_weights=False,
        )

        # Equation (4.1):
        #
        # h = CA + MLP(LN(CA))
        hidden = (
            attended
            + self.mlp(
                self.norm(
                    attended
                )
            )
        )

        return AttentionBlockOutput(
            hidden=hidden.squeeze(
                1
            ),
            attention=weights,
        )


class SelfAttentionBlock(nn.Module):
    """Diffolio market-level self-attention block.

    Input
    -----
    Asset and systematic tokens:

        [batch, tokens, hidden_dim]

    Output
    ------
    Refined tokens with identical shape.

    The per-head attention matrix is retained because the
    asset-to-asset portion will later be used by Diffolio's
    correlation-guided regularizer.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible "
                "by num_heads"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.mlp = FeedForward(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> AttentionBlockOutput:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape "
                "[batch, tokens, hidden_dim]"
            )

        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                "x hidden dimension differs"
            )

        attended, weights = self.attention(
            query=x,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,
        )

        # Equation (4.2):
        #
        # h' = SA + MLP(LN(SA))
        hidden = (
            attended
            + self.mlp(
                self.norm(
                    attended
                )
            )
        )

        return AttentionBlockOutput(
            hidden=hidden,
            attention=weights,
        )