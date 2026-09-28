"""
Causally-Strict Rolling Normalisation.

Reference: Proposal §7.4.3 — Causally-Strict Calibration.

Applies a rolling causal Z-score normalisation:

    F̃(t) = (F(t) − μ(t−W, t)) / σ(t−W, t)

where μ and σ are computed ONLY from the causal window [t−W, t].

Critical Design Constraint:
    Applying a global Z-score (μ_global, σ_global) over the entire dataset
    introduces a CATASTROPHIC look-ahead bias, as the normalisation
    parameters would contain information from the future (the test set).
    This module guarantees strict causal information flow.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class CausalNormalizer:
    r"""Rolling causal Z-score normaliser.

    Normalises features using statistics computed from a strictly causal
    rolling window, ensuring zero look-ahead bias.

    .. math::
        \tilde{\mathbf{F}}(t) = \frac{\mathbf{F}(t) - \mu_{(t-W, t)}}{\sigma_{(t-W, t)} + \epsilon}

    where:
        - W is the rolling window size (e.g., 120 ticks ≈ 2 hours of HFT data)
        - μ and σ are the sample mean and standard deviation
        - ε prevents division by zero during periods of zero variance

    The normaliser operates strictly in a causal manner:
        - At time t, ONLY data from [max(0, t−W+1), t] is used.
        - No future information leaks into the normalisation.
        - The first W−1 samples use a growing window (warm-up period).

    Args:
        window_size: W — rolling window size (number of ticks).
        eps: ε — small constant for numerical stability.
        min_periods: Minimum number of observations before normalising.
            If fewer observations are available, output is zero-filled.

    Example:
        >>> normalizer = CausalNormalizer(window_size=120)
        >>> raw_features = np.random.randn(1000, 12, 3)  # [T, N, F]
        >>> normed = normalizer.normalize_numpy(raw_features)
        >>> normed.shape  # [1000, 12, 3]
    """

    def __init__(
        self,
        window_size: int = 120,
        eps: float = 1e-8,
        min_periods: int = 10,
    ) -> None:
        self.window_size = window_size
        self.eps = eps
        self.min_periods = min_periods

    def normalize_numpy(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        r"""Apply causal rolling normalisation to a NumPy array.

        .. math::
            \tilde{F}_{t,i,j} = \frac{F_{t,i,j} - \mu_{i,j}^{(t-W:t)}}
                                     {\sigma_{i,j}^{(t-W:t)} + \epsilon}

        Args:
            features: Raw feature tensor, shape [T, ...].
                Typically [T, N, F] for multi-asset multi-feature data.
                The rolling window operates along axis 0 (time).

        Returns:
            Normalised features, same shape as input.
        """
        T = features.shape[0]
        rest_shape = features.shape[1:]
        flat = features.reshape(T, -1)  # [T, D]
        D = flat.shape[1]

        result = np.zeros_like(flat)

        # Compute rolling statistics causally
        for t in range(T):
            start = max(0, t - self.window_size + 1)
            window = flat[start:t + 1]  # [window_len, D]

            if len(window) < self.min_periods:
                # Not enough data: output zero (uninformative)
                result[t] = 0.0
                continue

            mu = window.mean(axis=0)  # [D]
            sigma = window.std(axis=0)  # [D]

            result[t] = (flat[t] - mu) / (sigma + self.eps)

        return result.reshape(T, *rest_shape)

    def normalize_numpy_vectorised(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Vectorised (fast) causal normalisation using cumulative sums.

        Uses the mathematical identity for rolling statistics:
            Σ_{i=t-W+1}^{t} x_i = cumsum[t] − cumsum[t−W]

        This avoids the O(T·W) loop, achieving O(T) complexity.

        Args:
            features: Raw features, shape [T, ...].

        Returns:
            Normalised features, same shape.
        """
        T = features.shape[0]
        rest_shape = features.shape[1:]
        flat = features.reshape(T, -1).astype(np.float64)
        W = self.window_size

        # Cumulative sums for mean
        cumsum = np.cumsum(flat, axis=0)        # [T, D]
        cumsum_sq = np.cumsum(flat ** 2, axis=0)  # [T, D]

        # Prepend zeros for clean subtraction
        cumsum = np.vstack([np.zeros((1, flat.shape[1])), cumsum])
        cumsum_sq = np.vstack([np.zeros((1, flat.shape[1])), cumsum_sq])

        result = np.zeros_like(flat)

        for t in range(T):
            start = max(0, t - W + 1)
            n = t - start + 1

            if n < self.min_periods:
                result[t] = 0.0
                continue

            # Rolling mean: (cumsum[t+1] − cumsum[start]) / n
            roll_sum = cumsum[t + 1] - cumsum[start]
            mu = roll_sum / n

            # Rolling variance: E[X²] − (E[X])²
            roll_sum_sq = cumsum_sq[t + 1] - cumsum_sq[start]
            var = roll_sum_sq / n - mu ** 2
            var = np.maximum(var, 0.0)  # Clamp numerical negatives
            sigma = np.sqrt(var)

            result[t] = (flat[t] - mu) / (sigma + self.eps)

        return result.reshape(T, *rest_shape).astype(np.float32)

    def normalize_torch(
        self,
        features: Tensor,
    ) -> Tensor:
        r"""Apply causal rolling normalisation to a PyTorch tensor.

        GPU-accelerated version using ``torch.cumsum`` for O(T) complexity.

        .. math::
            \mu_t = \frac{1}{n_t}\sum_{i=s_t}^{t} F_i, \quad
            \sigma_t = \sqrt{\frac{1}{n_t}\sum_{i=s_t}^{t} F_i^2 - \mu_t^2}

        where s_t = max(0, t − W + 1) and n_t = t − s_t + 1.

        Args:
            features: Raw features, shape [T, ...].

        Returns:
            Normalised features, same shape.
        """
        T = features.shape[0]
        rest_shape = features.shape[1:]
        flat = features.reshape(T, -1).double()  # [T, D]
        W = self.window_size
        D = flat.shape[1]

        # Cumulative sums
        cumsum = torch.cumsum(flat, dim=0)
        cumsum_sq = torch.cumsum(flat ** 2, dim=0)

        # Prepend zeros
        zero_row = torch.zeros(1, D, dtype=flat.dtype, device=flat.device)
        cumsum = torch.cat([zero_row, cumsum], dim=0)      # [T+1, D]
        cumsum_sq = torch.cat([zero_row, cumsum_sq], dim=0)  # [T+1, D]

        result = torch.zeros_like(flat)

        # Vectorise over time using index arithmetic
        t_indices = torch.arange(T, device=flat.device)
        start_indices = torch.clamp(t_indices - W + 1, min=0)
        n_samples = (t_indices - start_indices + 1).float().unsqueeze(1)  # [T, 1]

        # Rolling sums via cumsum differences
        roll_sum = cumsum[t_indices + 1] - cumsum[start_indices]  # [T, D]
        roll_sum_sq = cumsum_sq[t_indices + 1] - cumsum_sq[start_indices]

        mu = roll_sum / n_samples
        var = (roll_sum_sq / n_samples) - mu ** 2
        var = torch.clamp(var, min=0.0)
        sigma = torch.sqrt(var)

        result = (flat - mu) / (sigma + self.eps)

        # Zero out warm-up period
        mask = n_samples.squeeze(1) < self.min_periods
        result[mask] = 0.0

        return result.float().reshape(T, *rest_shape)

    def fit_statistics(
        self,
        train_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute global train-set statistics for optional global normalisation.

        This provides a fallback global Z-score computed ONLY from training
        data (no look-ahead bias).  Useful for features that don't benefit
        from rolling normalisation (e.g., macro indicators).

        Args:
            train_features: Training data, shape [T_train, ...].

        Returns:
            Tuple of (mean, std), each with shape matching trailing dims.
        """
        mu = train_features.mean(axis=0)
        sigma = train_features.std(axis=0) + self.eps
        return mu, sigma
