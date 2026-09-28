"""
Microstructural Feature Engineering from LOB Data.

Reference: Proposal §7.4.2 — Microstructural Invariant Feature Extraction.

Extracts three stationary microstructural invariants from raw LOB data:

1. Volume-Weighted Micro-Price (§7.4.2, Eq. 1):
    P_micro(t) = (V¹_b · P¹_a + V¹_a · P¹_b) / (V¹_b + V¹_a)

2. Order Flow Imbalance — OFI (§7.4.2, Eq. 2):
    OFI(t) = ΔW^b(t) − ΔW^a(t)
    where ΔW^b(t) = 𝕀(P¹_b(t) ≥ P¹_b(t−1))·V¹_b(t) − 𝕀(P¹_b(t) ≤ P¹_b(t−1))·V¹_b(t−1)

3. Multi-Level Liquidity Skew (§7.4.2, Eq. 3):
    S_skew(t) = Σ_d exp(−κ·d) · (V^d_b − V^d_a) / (V^d_b + V^d_a)

These features are stationary by construction, making them mathematically
suitable for the Neural CDE integration in Phase 0.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Microstructural feature extractor from Level-2 LOB data.

    Transforms raw, non-stationary LOB state vectors into stationary
    invariant features that can be safely fed into the Neural CDE.

    Financial Rationale:
        Raw prices P and volumes V are non-stationary (macroeconomic drift)
        and therefore "mathematically toxic" for deep learning initialisation
        (§7.4.2).  These engineered features extract the stationary physics
        of order flow.

    Args:
        depth: D — LOB depth per side.
        kappa: κ — exponential decay for liquidity skew.
            Higher κ → more weight on Level 1 (immediate impact).
        eps: Small constant to prevent division by zero.

    Example:
        >>> fe = FeatureEngineer(depth=10, kappa=0.5)
        >>> features = fe.compute_all(bid_prices, bid_volumes, ask_prices, ask_volumes)
        >>> features.shape  # [T, N, 3]  (micro_price, ofi, skew)
    """

    def __init__(
        self,
        depth: int = 10,
        kappa: float = 0.5,
        eps: float = 1e-10,
    ) -> None:
        self.depth = depth
        self.kappa = kappa
        self.eps = eps

        # Precompute exponential decay weights for liquidity skew
        # exp(-κ·d) for d = 0, 1, ..., D-1
        self.decay_weights = np.exp(-kappa * np.arange(depth))

    def compute_micro_price(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> np.ndarray:
        r"""Compute the Volume-Weighted Micro-Price.

        .. math::
            P_{micro}(t) = \frac{V_1^b(t) \cdot P_1^a(t) + V_1^a(t) \cdot P_1^b(t)}
                                {V_1^b(t) + V_1^a(t)}

        Financial Interpretation:
            If a massive bid wall appears (V¹_b >> V¹_a), the micro-price
            shifts closer to the ask price, anticipating an upward breakout
            BEFORE any trade occurs.

        Args:
            bid_prices: Best bid prices, shape [T, ...] or [T, N].
                Level 1 = column 0 (closest to mid).
            bid_volumes: Best bid volumes, shape [T, ...].
            ask_prices: Best ask prices, shape [T, ...].
            ask_volumes: Best ask volumes, shape [T, ...].

        Returns:
            Micro-prices, shape [T, ...].
        """
        # Use only Level 1 (best bid/ask)
        p_bid = bid_prices[..., 0]  # P¹_b
        v_bid = bid_volumes[..., 0]  # V¹_b
        p_ask = ask_prices[..., 0]  # P¹_a
        v_ask = ask_volumes[..., 0]  # V¹_a

        denominator = v_bid + v_ask + self.eps
        micro_price = (v_bid * p_ask + v_ask * p_bid) / denominator

        return micro_price

    def compute_ofi(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> np.ndarray:
        r"""Compute Order Flow Imbalance (OFI).

        .. math::
            OFI(t) = \Delta W^b(t) - \Delta W^a(t)

        where:
        .. math::
            \Delta W^b(t) = \mathbb{I}(P_1^b(t) \ge P_1^b(t{-}1)) \cdot V_1^b(t)
                          - \mathbb{I}(P_1^b(t) \le P_1^b(t{-}1)) \cdot V_1^b(t{-}1)

        Financial Interpretation:
            Positive OFI → aggressive buying pressure (or aggressive
            cancellation of sell orders).  OFI is a bounded, stationary
            derivative of the LOB, making it ideal fuel for the Neural CDE.

        Args:
            bid_prices: Bid prices, shape [T, ..., D].
            bid_volumes: Bid volumes, shape [T, ..., D].
            ask_prices: Ask prices, shape [T, ..., D].
            ask_volumes: Ask volumes, shape [T, ..., D].

        Returns:
            OFI values, shape [T, ...].
            First element is 0 (no previous tick for differencing).
        """
        # Level 1 only
        p_bid = bid_prices[..., 0]
        v_bid = bid_volumes[..., 0]
        p_ask = ask_prices[..., 0]
        v_ask = ask_volumes[..., 0]

        T = p_bid.shape[0]
        ofi = np.zeros_like(p_bid)

        # Compute ΔW^b(t)
        # 𝕀(P¹_b(t) ≥ P¹_b(t-1)) · V¹_b(t) - 𝕀(P¹_b(t) ≤ P¹_b(t-1)) · V¹_b(t-1)
        bid_price_up = (p_bid[1:] >= p_bid[:-1]).astype(np.float64)
        bid_price_down = (p_bid[1:] <= p_bid[:-1]).astype(np.float64)
        delta_w_bid = bid_price_up * v_bid[1:] - bid_price_down * v_bid[:-1]

        # Compute ΔW^a(t) (analogous for ask side)
        ask_price_up = (p_ask[1:] >= p_ask[:-1]).astype(np.float64)
        ask_price_down = (p_ask[1:] <= p_ask[:-1]).astype(np.float64)
        delta_w_ask = ask_price_up * v_ask[1:] - ask_price_down * v_ask[:-1]

        ofi[1:] = delta_w_bid - delta_w_ask

        return ofi

    def compute_liquidity_skew(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> np.ndarray:
        r"""Compute Multi-Level Liquidity Skew.

        .. math::
            S_{skew}(t) = \sum_{d=1}^{D} e^{-\kappa d}
                          \cdot \frac{V_d^b(t) - V_d^a(t)}{V_d^b(t) + V_d^a(t)}

        Financial Interpretation:
            Captures the deeper structural topology of the order book
            that Phase 1 persistent homology will later analyse for voids.
            κ acknowledges that Level 1 liquidity has vastly higher
            immediate market impact than Level 10.

        Args:
            bid_volumes: Bid volumes, shape [T, ..., D].
            ask_volumes: Ask volumes, shape [T, ..., D].

        Returns:
            Liquidity skew, shape [T, ...].
        """
        D = min(self.depth, bid_volumes.shape[-1])
        weights = self.decay_weights[:D]  # [D]

        # Normalised imbalance per level: (V^b - V^a) / (V^b + V^a)
        imbalance = (bid_volumes[..., :D] - ask_volumes[..., :D]) / (
            bid_volumes[..., :D] + ask_volumes[..., :D] + self.eps
        )

        # Weighted sum: Σ exp(-κd) · imbalance_d
        # weights shape: [D], imbalance shape: [T, ..., D]
        skew = np.sum(imbalance * weights, axis=-1)

        return skew

    def compute_all(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> np.ndarray:
        """Compute all three microstructural features.

        Args:
            bid_prices: shape [T, N, D] or [T, D] for single asset.
            bid_volumes: shape [T, N, D] or [T, D].
            ask_prices: shape [T, N, D] or [T, D].
            ask_volumes: shape [T, N, D] or [T, D].

        Returns:
            Feature tensor F(t) = [P_micro, OFI, S_skew],
            shape [T, N, 3] or [T, 3] for single asset.
        """
        micro_price = self.compute_micro_price(
            bid_prices, bid_volumes, ask_prices, ask_volumes
        )
        ofi = self.compute_ofi(
            bid_prices, bid_volumes, ask_prices, ask_volumes
        )
        skew = self.compute_liquidity_skew(bid_volumes, ask_volumes)

        # Stack along the last axis: [T, ..., 3]
        features = np.stack([micro_price, ofi, skew], axis=-1)

        logger.info(
            f"Feature extraction complete: shape={features.shape}, "
            f"features=[micro_price, OFI, liquidity_skew]"
        )

        return features

    def compute_all_torch(
        self,
        bid_prices: Tensor,
        bid_volumes: Tensor,
        ask_prices: Tensor,
        ask_volumes: Tensor,
    ) -> Tensor:
        """Torch-native version of ``compute_all`` for GPU acceleration.

        All operations are differentiable, enabling end-to-end gradient
        flow from raw LOB data through feature extraction.

        Args:
            bid_prices: shape [T, N, D].
            bid_volumes: shape [T, N, D].
            ask_prices: shape [T, N, D].
            ask_volumes: shape [T, N, D].

        Returns:
            Feature tensor, shape [T, N, 3].
        """
        eps = self.eps

        # --- Micro-Price ---
        p_bid = bid_prices[..., 0]
        v_bid = bid_volumes[..., 0]
        p_ask = ask_prices[..., 0]
        v_ask = ask_volumes[..., 0]
        micro_price = (v_bid * p_ask + v_ask * p_bid) / (v_bid + v_ask + eps)

        # --- OFI ---
        bid_up = (p_bid[1:] >= p_bid[:-1]).float()
        bid_down = (p_bid[1:] <= p_bid[:-1]).float()
        dw_bid = bid_up * v_bid[1:] - bid_down * v_bid[:-1]

        ask_up = (p_ask[1:] >= p_ask[:-1]).float()
        ask_down = (p_ask[1:] <= p_ask[:-1]).float()
        dw_ask = ask_up * v_ask[1:] - ask_down * v_ask[:-1]

        ofi = torch.zeros_like(p_bid)
        ofi[1:] = dw_bid - dw_ask

        # --- Liquidity Skew ---
        D = min(self.depth, bid_volumes.shape[-1])
        decay = torch.tensor(
            self.decay_weights[:D],
            dtype=bid_volumes.dtype,
            device=bid_volumes.device,
        )
        imbalance = (bid_volumes[..., :D] - ask_volumes[..., :D]) / (
            bid_volumes[..., :D] + ask_volumes[..., :D] + eps
        )
        skew = (imbalance * decay).sum(dim=-1)

        return torch.stack([micro_price, ofi, skew], dim=-1)
