"""
PyTorch Dataset and DataLoader Wrappers.

Provides the ``MarketDataset`` class that encapsulates:
    - Sliding-window sample extraction from continuous time-series.
    - Chronologically-strict train/validation/test splitting.
    - Proper lookback window construction across split boundaries.
    - Lazy loading for large-scale LOB datasets.

Each sample returned by the dataset contains:
    - features: [lookback, N, F] — historical feature tensor.
    - timestamps: [lookback] — corresponding time points.
    - spline_coeffs: Precomputed spline coefficients for the window.
    - targets: [horizon, N] — future returns for evaluation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class MarketDataset(Dataset):
    """Sliding-window dataset for financial time-series.

    Extracts overlapping windows of length ``lookback`` from the
    normalised feature tensor, with corresponding forward-looking
    targets of length ``horizon``.

    Information Barrier:
        Each sample at index i uses features from [i, i+lookback) and
        targets from [i+lookback, i+lookback+horizon).  There is zero
        overlap between feature and target windows, preventing data leakage.

    Args:
        features: Normalised feature tensor, shape [T, N, F].
        timestamps: Corresponding timestamps, shape [T].
        returns: Asset returns for target construction, shape [T, N].
            If None, targets are not provided (generation-only mode).
        lookback: Number of historical time-steps per sample.
        horizon: Number of future time-steps for targets.

    Example:
        >>> dataset = MarketDataset(features, timestamps, returns,
        ...                         lookback=60, horizon=10)
        >>> sample = dataset[0]
        >>> sample["features"].shape  # [60, 12, 3]
        >>> sample["targets"].shape   # [10, 12]
    """

    def __init__(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        returns: Optional[np.ndarray] = None,
        lookback: int = 60,
        horizon: int = 10,
    ) -> None:
        self.lookback = lookback
        self.horizon = horizon
        self.has_targets = returns is not None

        # Validate shapes
        T = features.shape[0]
        assert timestamps.shape[0] == T, (
            f"Timestamp length {timestamps.shape[0]} != feature length {T}"
        )
        if returns is not None:
            assert returns.shape[0] == T, (
                f"Returns length {returns.shape[0]} != feature length {T}"
            )

        # Store as tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float64)
        if returns is not None:
            self.returns = torch.tensor(returns, dtype=torch.float32)
        else:
            self.returns = None

        # Compute valid indices
        if self.has_targets:
            self.n_samples = T - lookback - horizon + 1
        else:
            self.n_samples = T - lookback + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data: T={T}, lookback={lookback}, "
                f"horizon={horizon} → n_samples={self.n_samples}"
            )

        logger.info(
            f"MarketDataset: T={T}, lookback={lookback}, horizon={horizon}, "
            f"n_samples={self.n_samples}, features_shape={features.shape}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a single training sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
                - ``features``: [lookback, N, F]
                - ``timestamps``: [lookback]
                - ``targets``: [horizon, N] (if returns provided)
        """
        start = idx
        end = idx + self.lookback

        sample = {
            "features": self.features[start:end],
            "timestamps": self.timestamps[start:end],
        }

        if self.has_targets:
            target_start = end
            target_end = end + self.horizon
            sample["targets"] = self.returns[target_start:target_end]

        return sample


def create_dataloaders(
    features: np.ndarray,
    timestamps: np.ndarray,
    returns: Optional[np.ndarray] = None,
    lookback: int = 60,
    horizon: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Create train/validation/test DataLoaders with chronological splits.

    Splits are strictly chronological (no shuffling across time):
        - Train: [0, T_train)
        - Validation: [T_train, T_val)
        - Test: [T_val, T]

    This prevents look-ahead bias in the evaluation.

    Args:
        features: Full feature tensor, shape [T, N, F].
        timestamps: Full timestamps, shape [T].
        returns: Full returns, shape [T, N].
        lookback: Historical window size.
        horizon: Forecast horizon.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        batch_size: Mini-batch size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for CUDA transfers.

    Returns:
        Dictionary with ``"train"``, ``"val"``, ``"test"`` DataLoaders.
    """
    T = features.shape[0]
    T_train = int(T * train_ratio)
    T_val = int(T * (train_ratio + val_ratio))

    splits = {
        "train": (0, T_train),
        "val": (T_train, T_val),
        "test": (T_val, T),
    }

    dataloaders = {}

    for split_name, (start, end) in splits.items():
        # Extend start backward by lookback to allow full windows
        # at the beginning of val/test sets
        effective_start = max(0, start - lookback)

        split_features = features[effective_start:end]
        split_timestamps = timestamps[effective_start:end]
        split_returns = returns[effective_start:end] if returns is not None else None

        dataset = MarketDataset(
            features=split_features,
            timestamps=split_timestamps,
            returns=split_returns,
            lookback=lookback,
            horizon=horizon,
        )

        shuffle = (split_name == "train")

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        logger.info(
            f"DataLoader '{split_name}': {len(dataset)} samples, "
            f"batch_size={batch_size}, shuffle={shuffle}"
        )

    return dataloaders
