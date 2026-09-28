"""
Score Network (Conditional UNet) for Denoising Score Matching.

Reference: Proposal §5.6.2 — Riemannian Score Network.

Predicts the Stein score ∇_Y log p_t(Y_t) conditioned on condition
via Adaptive Group Normalisation (AdaGN).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Args: t shape [B]. Returns: shape [B, dim]."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaGN(nn.Module):
    """Adaptive Group Normalisation — condition injection via scale/shift.

    GroupNorm(x) → γ(c) · GN(x) + β(c)

    Args:
        n_channels: Number of channels.
        condition_dim: Dimension of conditioning vector.
        n_groups: Number of groups for GroupNorm.
    """

    def __init__(self, n_channels: int, condition_dim: int, n_groups: int = 8) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(min(n_groups, n_channels), n_channels)
        self.proj = nn.Linear(condition_dim, 2 * n_channels)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Args: x [B,C,L], cond [B,D]. Returns: [B,C,L]."""
        h = self.gn(x)
        scale_shift = self.proj(cond).unsqueeze(-1)  # [B, 2C, 1]
        scale, shift = scale_shift.chunk(2, dim=1)
        return h * (1.0 + scale) + shift


class ResBlock(nn.Module):
    """Residual block with AdaGN condition injection."""

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = AdaGN(channels, condition_dim)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = AdaGN(channels, condition_dim)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.act(self.norm1(x, cond))
        h = self.conv1(h)
        h = self.act(self.norm2(h, cond))
        h = self.conv2(h)
        return x + h


class ScoreNetwork(nn.Module):
    r"""Conditional 1D UNet for score prediction.

    Predicts s_θ(Y_t, t, condition) ≈ ∇_Y log p_t(Y_t).

    Architecture:
        - Encoder: strided convolutions with ResBlocks
        - Bottleneck: ResBlock with self-attention
        - Decoder: transposed convolutions with skip connections
        - Condition injection: AdaGN at every ResBlock

    Args:
        data_dim: D — dimension of the data Y_t (N assets).
        channels: Channel dims per resolution level.
        time_embed_dim: Sinusoidal time embedding dimension.
        condition_dim: condition dimension.
        n_res_blocks: ResBlocks per level.
    """

    def __init__(
        self,
        data_dim: int = 12,
        channels: list[int] | None = None,
        time_embed_dim: int = 128,
        condition_dim: int = 256,
        n_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        cond_dim = time_embed_dim + condition_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(1, channels[0], 1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, ch in enumerate(channels):
            blocks = nn.ModuleList(
                [ResBlock(ch, cond_dim) for _ in range(n_res_blocks)]
            )
            self.encoders.append(blocks)
            if i < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv1d(ch, channels[i + 1], 3, stride=2, padding=1)
                )

        # Bottleneck
        self.bottleneck = ResBlock(channels[-1], cond_dim)

        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            self.upsamples.append(
                nn.ConvTranspose1d(rev_channels[i], rev_channels[i + 1], 4, stride=2, padding=1)
            )
            blocks = nn.ModuleList(
                [ResBlock(rev_channels[i + 1], cond_dim) for _ in range(n_res_blocks)]
            )
            self.decoders.append(blocks)

        # Output projection
        out_ch = channels[0] if len(channels) > 0 else 64
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, 1, 1),
        )

        self.data_dim = data_dim

    def forward(
        self, x: Tensor, t: Tensor, condition: Tensor,
    ) -> Tensor:
        """Predict the score (noise prediction).

        Args:
            x: Noisy data Y_t, shape [B, D].
            t: Diffusion timesteps, shape [B].
            condition: Conditioning tensor, shape [B, C].

        Returns:
            Predicted score/noise, shape [B, D].
        """
        # Embed time and concatenate with condition
        t_emb = self.time_embed(t)  # [B, time_dim]
        cond = torch.cat([t_emb, condition], dim=-1)  # [B, cond_dim]

        # Reshape data: [B, D] → [B, 1, D]
        h = x.unsqueeze(1)
        h = self.input_proj(h)  # [B, C0, D]

        # Encoder with skip connections
        skips = []
        for i, blocks in enumerate(self.encoders):
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        # Bottleneck
        h = self.bottleneck(h, cond)

        # Decoder
        for i, (upsample, blocks) in enumerate(zip(self.upsamples, self.decoders)):
            h = upsample(h)
            skip = skips[-(i + 2)]
            # Handle size mismatch from strided conv
            if h.shape[-1] != skip.shape[-1]:
                h = h[..., :skip.shape[-1]]
            h = h + skip
            for block in blocks:
                h = block(h, cond)

        # Output
        h = self.output_proj(h)  # [B, 1, D]
        return h.squeeze(1)  # [B, D]
