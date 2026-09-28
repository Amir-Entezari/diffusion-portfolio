"""
Level-2 Limit Order Book (LOB) Parser.

Reference: Proposal §7.4.1 — The Raw Limit Order Book (LOB) Tensor Formulation.

Constructs the raw spatial state vector for each asset at event tick t:

    X_LOB(t) = [P¹_b(t), V¹_b(t), ..., P^D_b(t), V^D_b(t),
                P¹_a(t), V¹_a(t), ..., P^D_a(t), V^D_a(t)]^T

    X_LOB(t) ∈ ℝ^{4D × 1}  (D levels × 2 sides × 2 fields [price, volume])

For N assets, the full market state is:
    X_market(t) ∈ ℝ^{N × 4D}

Supports:
    - LOBSTER format (https://lobsterdata.com/)
    - NASDAQ ITCH binary protocol (parsed to CSV)
    - Generic CSV with configurable column mappings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class LOBParser:
    """Parser for Level-2 Limit Order Book data.

    Converts raw LOB files into structured tensors suitable for the
    Neural CDE (Phase 0) and feature engineering pipeline.

    The parser handles the fundamental challenge of LOB data: asynchronous,
    event-driven timestamps where Δt = t_n - t_{n-1} is highly stochastic.
    During flash crashes, hundreds of ticks per millisecond; during lunch
    hours, seconds of silence.

    Attributes:
        n_assets: N — number of assets in the cross-section.
        depth: D — number of price levels per side.
        raw_dim: 4D — raw state dimension per asset.

    Example:
        >>> parser = LOBParser(n_assets=12, depth=10)
        >>> tensors = parser.parse_directory("data/raw/lobster/")
        >>> tensors["prices"].shape  # [T_total, N, 2*D]  (bid+ask prices)
        >>> tensors["volumes"].shape  # [T_total, N, 2*D]  (bid+ask volumes)
    """

    def __init__(
        self,
        n_assets: int = 12,
        depth: int = 10,
    ) -> None:
        """Initialise the LOB parser.

        Args:
            n_assets: N — number of assets.
            depth: D — number of LOB levels per side (bid/ask).
        """
        self.n_assets = n_assets
        self.depth = depth
        self.raw_dim = 4 * depth  # P_bid, V_bid, P_ask, V_ask × D levels

    def parse_lobster_file(
        self,
        orderbook_file: Union[str, Path],
        message_file: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """Parse a single-asset LOBSTER orderbook + message file pair.

        LOBSTER format (https://lobsterdata.com/):
            - Orderbook file: columns are
              [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...]
              with 4 × D columns total.
            - Message file: columns are
              [time, type, order_id, size, price, direction]

        Args:
            orderbook_file: Path to the LOBSTER orderbook CSV.
            message_file: Path to the LOBSTER message CSV.

        Returns:
            Dictionary with keys:
                - ``timestamps``: Event timestamps, shape [T].
                - ``bid_prices``: Bid prices per level, shape [T, D].
                - ``bid_volumes``: Bid volumes per level, shape [T, D].
                - ``ask_prices``: Ask prices per level, shape [T, D].
                - ``ask_volumes``: Ask volumes per level, shape [T, D].
                - ``lob_state``: Full state vector X_LOB(t), shape [T, 4D].
        """
        orderbook_file = Path(orderbook_file)
        message_file = Path(message_file)

        logger.info(f"Parsing LOBSTER: {orderbook_file.name}")

        # Parse message file for timestamps
        messages = pd.read_csv(
            message_file,
            header=None,
            names=["time", "type", "order_id", "size", "price", "direction"],
        )
        timestamps = messages["time"].values

        # Parse orderbook file
        # LOBSTER format: ask_p1, ask_s1, bid_p1, bid_s1, ask_p2, ask_s2, ...
        ob = pd.read_csv(orderbook_file, header=None).values

        D = self.depth
        n_cols = ob.shape[1]
        expected_cols = 4 * D

        if n_cols < expected_cols:
            raise ValueError(
                f"Orderbook file has {n_cols} columns but depth={D} "
                f"requires {expected_cols} columns."
            )

        # Extract price/volume arrays
        # LOBSTER interleaves: ask_p, ask_s, bid_p, bid_s for each level
        ask_prices = ob[:, 0::4][:, :D]   # Every 4th starting from 0
        ask_volumes = ob[:, 1::4][:, :D]   # Every 4th starting from 1
        bid_prices = ob[:, 2::4][:, :D]    # Every 4th starting from 2
        bid_volumes = ob[:, 3::4][:, :D]   # Every 4th starting from 3

        # Convert prices from integer ticks to decimal (LOBSTER uses cents×10000)
        ask_prices = ask_prices.astype(np.float64) / 10000.0
        bid_prices = bid_prices.astype(np.float64) / 10000.0
        ask_volumes = ask_volumes.astype(np.float64)
        bid_volumes = bid_volumes.astype(np.float64)

        # Construct the full state vector X_LOB(t) per §7.4.1:
        # [P¹_b, V¹_b, ..., P^D_b, V^D_b, P¹_a, V¹_a, ..., P^D_a, V^D_a]
        lob_state = np.zeros((len(timestamps), 4 * D), dtype=np.float64)
        for d in range(D):
            lob_state[:, 2 * d] = bid_prices[:, d]
            lob_state[:, 2 * d + 1] = bid_volumes[:, d]
        for d in range(D):
            lob_state[:, 2 * D + 2 * d] = ask_prices[:, d]
            lob_state[:, 2 * D + 2 * d + 1] = ask_volumes[:, d]

        return {
            "timestamps": timestamps,
            "bid_prices": bid_prices,
            "bid_volumes": bid_volumes,
            "ask_prices": ask_prices,
            "ask_volumes": ask_volumes,
            "lob_state": lob_state,
        }

    def parse_generic_csv(
        self,
        filepath: Union[str, Path],
        timestamp_col: str = "timestamp",
        bid_price_cols: Optional[List[str]] = None,
        bid_volume_cols: Optional[List[str]] = None,
        ask_price_cols: Optional[List[str]] = None,
        ask_volume_cols: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Parse a generic CSV with configurable column names.

        Args:
            filepath: Path to the CSV file.
            timestamp_col: Column name for timestamps.
            bid_price_cols: Column names for bid prices [level 1..D].
            bid_volume_cols: Column names for bid volumes [level 1..D].
            ask_price_cols: Column names for ask prices [level 1..D].
            ask_volume_cols: Column names for ask volumes [level 1..D].

        Returns:
            Same dictionary structure as ``parse_lobster_file``.
        """
        filepath = Path(filepath)
        logger.info(f"Parsing generic CSV: {filepath.name}")

        df = pd.read_csv(filepath)
        D = self.depth

        timestamps = df[timestamp_col].values

        # Auto-generate column names if not provided
        if bid_price_cols is None:
            bid_price_cols = [f"bid_price_{d+1}" for d in range(D)]
        if bid_volume_cols is None:
            bid_volume_cols = [f"bid_volume_{d+1}" for d in range(D)]
        if ask_price_cols is None:
            ask_price_cols = [f"ask_price_{d+1}" for d in range(D)]
        if ask_volume_cols is None:
            ask_volume_cols = [f"ask_volume_{d+1}" for d in range(D)]

        bid_prices = df[bid_price_cols].values.astype(np.float64)
        bid_volumes = df[bid_volume_cols].values.astype(np.float64)
        ask_prices = df[ask_price_cols].values.astype(np.float64)
        ask_volumes = df[ask_volume_cols].values.astype(np.float64)

        # Construct full LOB state
        lob_state = np.zeros((len(timestamps), 4 * D), dtype=np.float64)
        for d in range(D):
            lob_state[:, 2 * d] = bid_prices[:, d]
            lob_state[:, 2 * d + 1] = bid_volumes[:, d]
        for d in range(D):
            lob_state[:, 2 * D + 2 * d] = ask_prices[:, d]
            lob_state[:, 2 * D + 2 * d + 1] = ask_volumes[:, d]

        return {
            "timestamps": timestamps,
            "bid_prices": bid_prices,
            "bid_volumes": bid_volumes,
            "ask_prices": ask_prices,
            "ask_volumes": ask_volumes,
            "lob_state": lob_state,
        }

    def align_multi_asset(
        self,
        asset_data: Dict[str, Dict[str, np.ndarray]],
        method: str = "forward_fill",
    ) -> Dict[str, np.ndarray]:
        """Align multiple assets to a common asynchronous timeline.

        LOB data arrives asynchronously — each asset has its own event
        stream.  This method creates a unified timeline and forward-fills
        stale quotes (the standard approach in market microstructure).

        Args:
            asset_data: Mapping from asset name → parsed LOB dict.
            method: Alignment method.
                ``"forward_fill"``: Carry last known state forward (no look-ahead).
                ``"interpolate"``: Linear interpolation between ticks.

        Returns:
            Dictionary with:
                - ``timestamps``: Unified timeline, shape [T_unified].
                - ``market_state``: Full market tensor, shape [T_unified, N, 4D].
        """
        # Collect all unique timestamps across all assets
        all_timestamps = set()
        asset_names = sorted(asset_data.keys())

        for name in asset_names:
            all_timestamps.update(asset_data[name]["timestamps"].tolist())

        unified_times = np.array(sorted(all_timestamps))
        T = len(unified_times)
        N = len(asset_names)
        dim = 4 * self.depth

        market_state = np.zeros((T, N, dim), dtype=np.float64)

        for i, name in enumerate(asset_names):
            data = asset_data[name]
            asset_times = data["timestamps"]
            asset_lob = data["lob_state"]

            if method == "forward_fill":
                # For each unified timestamp, find the most recent asset event
                indices = np.searchsorted(asset_times, unified_times, side="right") - 1
                indices = np.clip(indices, 0, len(asset_times) - 1)
                market_state[:, i, :] = asset_lob[indices]
            elif method == "interpolate":
                # Linear interpolation
                for d in range(dim):
                    market_state[:, i, d] = np.interp(
                        unified_times, asset_times, asset_lob[:, d]
                    )
            else:
                raise ValueError(f"Unknown alignment method: {method}")

        logger.info(
            f"Multi-asset alignment complete: "
            f"T={T}, N={N}, dim_per_asset={dim}"
        )

        return {
            "timestamps": unified_times,
            "market_state": market_state,
        }

    def to_tensor(
        self,
        market_data: Dict[str, np.ndarray],
        device: str = "cpu",
    ) -> Dict[str, Tensor]:
        """Convert numpy arrays to PyTorch tensors with proper dtypes.

        Args:
            market_data: Output from ``parse_*`` or ``align_multi_asset``.
            device: Target device string.

        Returns:
            Dictionary with the same keys but as ``torch.Tensor`` objects.
        """
        result = {}
        for key, val in market_data.items():
            if isinstance(val, np.ndarray):
                if val.dtype in (np.float32, np.float64):
                    result[key] = torch.tensor(val, dtype=torch.float32, device=device)
                else:
                    result[key] = torch.tensor(val, device=device)
            else:
                result[key] = val
        return result
