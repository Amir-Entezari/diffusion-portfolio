"""PyTorch datasets and loaders for windowed excess returns.

The neural model consumes standardized returns, while raw decimal excess
returns are retained alongside them for evaluation and portfolio construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from diffusion_portfolio.data.core import ReturnTable
from diffusion_portfolio.data.splits import (
    WindowSplits,
    split_windows_by_date,
)
from diffusion_portfolio.data.windows import (
    WindowedReturns,
    make_windows,
)


class ReturnWindowDataset(Dataset):
    """Paired standardized/raw return windows.

    Parameters
    ----------
    model_windows:
        Windows in model space, normally standardized returns.

    raw_windows:
        Matching windows in raw decimal excess-return space.

    Notes
    -----
    The two inputs must describe exactly the same samples and dates.
    """

    def __init__(
        self,
        model_windows: WindowedReturns,
        raw_windows: WindowedReturns,
    ) -> None:
        if model_windows.history.shape != raw_windows.history.shape:
            raise ValueError(
                "Model/raw history shapes do not match"
            )

        if model_windows.target.shape != raw_windows.target.shape:
            raise ValueError(
                "Model/raw target shapes do not match"
            )

        if not model_windows.target_dates.equals(
            raw_windows.target_dates
        ):
            raise ValueError(
                "Model/raw target dates do not match"
            )

        self.model_windows = model_windows
        self.raw_windows = raw_windows

    def __len__(self) -> int:
        return self.model_windows.history.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        return {
            "history": torch.from_numpy(
                self.model_windows.history[index]
            ).float(),
            "target": torch.from_numpy(
                self.model_windows.target[index]
            ).float(),
            "history_raw": torch.from_numpy(
                self.raw_windows.history[index]
            ).float(),
            "target_raw": torch.from_numpy(
                self.raw_windows.target[index]
            ).float(),
            "target_date": self.model_windows.target_dates[
                index
            ],
        }


def collate_return_batch(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate tensors while preserving calendar dates."""

    return {
        "history": torch.stack(
            [sample["history"] for sample in batch]
        ),
        "target": torch.stack(
            [sample["target"] for sample in batch]
        ),
        "history_raw": torch.stack(
            [sample["history_raw"] for sample in batch]
        ),
        "target_raw": torch.stack(
            [sample["target_raw"] for sample in batch]
        ),
        "target_date": pd.DatetimeIndex(
            [
                sample["target_date"]
                for sample in batch
            ]
        ),
    }


@dataclass(frozen=True)
class DatasetSplits:
    """Train/validation/test PyTorch datasets."""

    train: ReturnWindowDataset
    val: ReturnWindowDataset
    test: ReturnWindowDataset


@dataclass(frozen=True)
class LoaderSplits:
    """Train/validation/test PyTorch dataloaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def _pair_window_splits(
    model_splits: WindowSplits,
    raw_splits: WindowSplits,
) -> DatasetSplits:
    """Pair model-space and raw-space windows."""

    return DatasetSplits(
        train=ReturnWindowDataset(
            model_splits.train,
            raw_splits.train,
        ),
        val=ReturnWindowDataset(
            model_splits.val,
            raw_splits.val,
        ),
        test=ReturnWindowDataset(
            model_splits.test,
            raw_splits.test,
        ),
    )


def build_window_datasets(
    model_table: ReturnTable,
    raw_table: ReturnTable,
    *,
    lookback: int,
    horizon: int,
    train_end: str,
    val_end: str,
    test_end: str | None = None,
) -> DatasetSplits:
    """Build aligned model/raw PyTorch datasets.

    ``model_table`` is normally standardized excess returns.

    ``raw_table`` is the corresponding unstandardized decimal
    excess-return table.
    """

    if not model_table.dates.equals(
        raw_table.dates
    ):
        raise ValueError(
            "Model/raw return tables must have identical dates"
        )

    if model_table.columns != raw_table.columns:
        raise ValueError(
            "Model/raw return tables must have identical columns"
        )

    model_windows = make_windows(
        model_table,
        lookback=lookback,
        horizon=horizon,
    )

    raw_windows = make_windows(
        raw_table,
        lookback=lookback,
        horizon=horizon,
    )

    model_splits = split_windows_by_date(
        model_windows,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )

    raw_splits = split_windows_by_date(
        raw_windows,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )

    return _pair_window_splits(
        model_splits,
        raw_splits,
    )


def make_dataloaders(
    datasets: DatasetSplits,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> LoaderSplits:
    """Create deterministic train/validation/test loaders.

    Training samples are shuffled because each sample already contains
    its complete chronological history window.

    Validation and test samples remain chronological.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        datasets.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_return_batch,
        generator=generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        datasets.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_return_batch,
        drop_last=False,
    )

    test_loader = DataLoader(
        datasets.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_return_batch,
        drop_last=False,
    )

    # Explicit semantic assertions. These are cheap and document intent.
    if not isinstance(
        train_loader.sampler,
        RandomSampler,
    ):
        raise RuntimeError(
            "Training loader must shuffle samples"
        )

    if not isinstance(
        val_loader.sampler,
        SequentialSampler,
    ):
        raise RuntimeError(
            "Validation loader must remain chronological"
        )

    if not isinstance(
        test_loader.sampler,
        SequentialSampler,
    ):
        raise RuntimeError(
            "Test loader must remain chronological"
        )

    return LoaderSplits(
        train=train_loader,
        val=val_loader,
        test=test_loader,
    )