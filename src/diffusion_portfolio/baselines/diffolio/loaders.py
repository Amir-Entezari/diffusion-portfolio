"""PyTorch datasets and loaders for Diffolio windows."""

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

from diffusion_portfolio.baselines.diffolio.dataset import (
    DiffolioWindows,
    DiffolioWindowSplits,
)


class DiffolioDataset(Dataset):
    """PyTorch wrapper around frozen Diffolio windows."""

    def __init__(
        self,
        windows: DiffolioWindows,
    ) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return self.windows.target.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        return {
            "return_history": torch.from_numpy(
                self.windows.return_history[index]
            ).float(),

            "return_history_raw": torch.from_numpy(
                self.windows.return_history_raw[index]
            ).float(),

            "asset_covariates": torch.from_numpy(
                self.windows.asset_covariates[index]
            ).float(),

            "systematic_covariates": torch.from_numpy(
                self.windows.systematic_covariates[index]
            ).float(),

            "target": torch.from_numpy(
                self.windows.target[index]
            ).float(),

            "target_raw": torch.from_numpy(
                self.windows.target_raw[index]
            ).float(),

            "target_date": self.windows.target_dates[
                index
            ],
        }


def collate_diffolio_batch(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack tensors while preserving calendar dates."""

    return {
        "return_history": torch.stack(
            [
                sample["return_history"]
                for sample in batch
            ]
        ),

        "return_history_raw": torch.stack(
            [
                sample["return_history_raw"]
                for sample in batch
            ]
        ),

        "asset_covariates": torch.stack(
            [
                sample["asset_covariates"]
                for sample in batch
            ]
        ),

        "systematic_covariates": torch.stack(
            [
                sample["systematic_covariates"]
                for sample in batch
            ]
        ),

        "target": torch.stack(
            [
                sample["target"]
                for sample in batch
            ]
        ),

        "target_raw": torch.stack(
            [
                sample["target_raw"]
                for sample in batch
            ]
        ),

        "target_date": pd.DatetimeIndex(
            [
                sample["target_date"]
                for sample in batch
            ]
        ),
    }


@dataclass(frozen=True)
class DiffolioDatasetSplits:
    """Train/validation/test PyTorch datasets."""

    train: DiffolioDataset
    val: DiffolioDataset
    test: DiffolioDataset


@dataclass(frozen=True)
class DiffolioLoaderSplits:
    """Train/validation/test PyTorch loaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def make_diffolio_datasets(
    windows: DiffolioWindowSplits,
) -> DiffolioDatasetSplits:
    """Wrap chronological Diffolio window splits."""

    return DiffolioDatasetSplits(
        train=DiffolioDataset(
            windows.train
        ),
        val=DiffolioDataset(
            windows.val
        ),
        test=DiffolioDataset(
            windows.test
        ),
    )


def make_diffolio_dataloaders(
    datasets: DiffolioDatasetSplits,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> DiffolioLoaderSplits:
    """Create deterministic Diffolio data loaders.

    Training samples are shuffled.

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
    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        datasets.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_diffolio_batch,
        generator=generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        datasets.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_diffolio_batch,
        drop_last=False,
    )

    test_loader = DataLoader(
        datasets.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_diffolio_batch,
        drop_last=False,
    )

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

    return DiffolioLoaderSplits(
        train=train_loader,
        val=val_loader,
        test=test_loader,
    )