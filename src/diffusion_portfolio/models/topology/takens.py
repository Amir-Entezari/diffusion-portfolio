"""
Takens Delay Embedding for Attractor Reconstruction.

Reference: Proposal §5.3.1 — Takens Embedding Theorem Application.

Constructs delay-coordinate embeddings from the CDE hidden state:

    Y(t) = [H(t)^T, H(t−τ)^T, ..., H(t−(m−1)τ)^T]^T ∈ ℝ^{m × h}

where:
    - m: embedding dimension (must satisfy m > 2d, Whitney bound)
    - τ: time delay (auto-selected via mutual information minimum)
    - d: fractal dimension of the underlying market attractor

Takens' Theorem guarantees that Y(t) is diffeomorphic to the
original attractor when m and τ are chosen correctly, preserving
the topological structure that topology will analyse.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class TakensEmbedding(nn.Module):
    r"""Takens delay-coordinate embedding.

    .. math::
        \mathbf{Y}(t) = [H(t)^T, H(t-\tau)^T, \ldots, H(t-(m-1)\tau)^T]^T
                       \in \mathbb{R}^{m \times h}

    Args:
        embedding_dim: m — number of delay copies.
        time_delay: τ — delay in time-step units. If None, auto-selected.
        time_delay_max_lag: Maximum lag to search for auto-τ.

    Example:
        >>> emb = TakensEmbedding(embedding_dim=10, time_delay=3)
        >>> H = torch.randn(4, 50, 32)  # [B, T, h]
        >>> Y = emb(H)
        >>> Y.shape  # [4, 23, 10, 32]  — [B, T_valid, m, h]
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        time_delay: Optional[int] = None,
        time_delay_max_lag: int = 50,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.time_delay_max_lag = time_delay_max_lag
        self._auto_delay: Optional[int] = None

    def _estimate_delay_mutual_information(
        self,
        signal: np.ndarray,
        max_lag: int,
        n_bins: int = 64,
    ) -> int:
        r"""Estimate optimal τ via first minimum of mutual information.

        The mutual information between X(t) and X(t+τ) quantifies
        the non-linear dependence. The first local minimum corresponds
        to the delay where the time-lagged copies are maximally
        "independent" — optimal for attractor reconstruction.

        Uses histogram-based MI estimation.

        Args:
            signal: 1D time series, shape [T].
            max_lag: Maximum lag to evaluate.
            n_bins: Number of histogram bins for MI estimation.

        Returns:
            Optimal delay τ (integer time steps).
        """
        T = len(signal)
        max_lag = min(max_lag, T // 3)

        mi_values = np.zeros(max_lag)

        for lag in range(1, max_lag):
            x = signal[:T - lag]
            y = signal[lag:]

            # 2D histogram for joint distribution
            hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
            pxy = hist_2d / hist_2d.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)

            # MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
            mask = pxy > 0
            mi = np.sum(
                pxy[mask] * np.log(
                    pxy[mask] / (px[:, None] * py[None, :])[mask]
                )
            )
            mi_values[lag] = mi

        # Find first local minimum
        for lag in range(2, max_lag - 1):
            if mi_values[lag] < mi_values[lag - 1] and mi_values[lag] < mi_values[lag + 1]:
                return lag

        # Fallback: first inflection or default
        return max(1, max_lag // 4)

    def estimate_delay(self, hidden_states: Tensor) -> int:
        """Estimate optimal time delay from hidden state trajectory.

        Args:
            hidden_states: H(t), shape [B, T, h] or [T, h].

        Returns:
            Estimated optimal delay τ.
        """
        if hidden_states.dim() == 3:
            # Use first batch, first hidden dimension
            signal = hidden_states[0, :, 0].detach().cpu().numpy()
        else:
            signal = hidden_states[:, 0].detach().cpu().numpy()

        tau = self._estimate_delay_mutual_information(
            signal, self.time_delay_max_lag
        )
        self._auto_delay = tau
        logger.info(f"Auto-estimated Takens delay: τ = {tau}")
        return tau

    def forward(
        self,
        hidden_states: Tensor,
        time_delay: Optional[int] = None,
    ) -> Tensor:
        r"""Construct Takens delay embedding from hidden state trajectory.

        Args:
            hidden_states: H(t), shape [B, T, h].
            time_delay: Override delay. If None, uses self.time_delay
                or auto-estimated value.

        Returns:
            Delay embedding Y(t), shape [B, T_valid, m, h].
            T_valid = T − (m−1)·τ
        """
        tau = time_delay or self.time_delay or self._auto_delay
        if tau is None:
            tau = self.estimate_delay(hidden_states)

        B, T, h = hidden_states.shape
        m = self.embedding_dim

        # Required history: (m-1)*tau time steps
        required = (m - 1) * tau
        if T <= required:
            raise ValueError(
                f"Insufficient time steps: T={T}, but need > {required} "
                f"for m={m}, τ={tau}"
            )

        T_valid = T - required

        # Construct delay vectors: Y(t) = [H(t), H(t-τ), ..., H(t-(m-1)τ)]
        # Each column is a delayed copy of the hidden state
        delays = []
        for k in range(m):
            offset = required - k * tau
            delays.append(hidden_states[:, offset:offset + T_valid, :])

        # Stack along new dimension: [B, T_valid, m, h]
        Y = torch.stack(delays, dim=2)

        return Y
