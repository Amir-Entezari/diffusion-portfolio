import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
)

from diffusion_portfolio.baselines.diffolio.dataset import (
    DiffolioWindows,
    DiffolioWindowSplits,
)
from diffusion_portfolio.baselines.diffolio.loaders import (
    DiffolioDataset,
    collate_diffolio_batch,
    make_diffolio_dataloaders,
    make_diffolio_datasets,
)


def make_windows(
    n_samples: int,
) -> DiffolioWindows:
    lookback = 5
    n_assets = 3
    n_characteristics = 2
    n_systematic = 4

    rng = np.random.default_rng(
        123
    )

    dates = pd.bdate_range(
        "2000-01-03",
        periods=n_samples,
    )

    return DiffolioWindows(
        return_history=rng.normal(
            size=(
                n_samples,
                lookback,
                n_assets,
            )
        ).astype(
            np.float32
        ),

        return_history_raw=rng.normal(
            size=(
                n_samples,
                lookback,
                n_assets,
            )
        ).astype(
            np.float32
        ),

        asset_covariates=rng.normal(
            size=(
                n_samples,
                lookback,
                n_assets,
                n_characteristics,
            )
        ).astype(
            np.float32
        ),

        systematic_covariates=rng.normal(
            size=(
                n_samples,
                lookback,
                n_systematic,
            )
        ).astype(
            np.float32
        ),

        target=rng.normal(
            size=(
                n_samples,
                n_assets,
            )
        ).astype(
            np.float32
        ),

        target_raw=rng.normal(
            size=(
                n_samples,
                n_assets,
            )
        ).astype(
            np.float32
        ),

        target_dates=dates,
    )


def make_splits():
    return DiffolioWindowSplits(
        train=make_windows(
            10
        ),
        val=make_windows(
            4
        ),
        test=make_windows(
            5
        ),
    )


def test_dataset_returns_expected_fields_and_shapes():
    windows = make_windows(
        10
    )

    dataset = DiffolioDataset(
        windows
    )

    sample = dataset[
        0
    ]

    assert len(
        dataset
    ) == 10

    assert sample[
        "return_history"
    ].shape == (
        5,
        3,
    )

    assert sample[
        "return_history_raw"
    ].shape == (
        5,
        3,
    )

    assert sample[
        "asset_covariates"
    ].shape == (
        5,
        3,
        2,
    )

    assert sample[
        "systematic_covariates"
    ].shape == (
        5,
        4,
    )

    assert sample[
        "target"
    ].shape == (
        3,
    )

    assert sample[
        "target_raw"
    ].shape == (
        3,
    )

    assert isinstance(
        sample["target"],
        torch.Tensor,
    )

    assert (
        sample["target_date"]
        == windows.target_dates[0]
    )


def test_dataset_split_lengths_are_preserved():
    window_splits = (
        make_splits()
    )

    datasets = (
        make_diffolio_datasets(
            window_splits
        )
    )

    assert len(
        datasets.train
    ) == 10

    assert len(
        datasets.val
    ) == 4

    assert len(
        datasets.test
    ) == 5


def test_collate_diffolio_batch_shapes():
    dataset = DiffolioDataset(
        make_windows(
            10
        )
    )

    batch = (
        collate_diffolio_batch(
            [
                dataset[0],
                dataset[1],
            ]
        )
    )

    assert batch[
        "return_history"
    ].shape == (
        2,
        5,
        3,
    )

    assert batch[
        "asset_covariates"
    ].shape == (
        2,
        5,
        3,
        2,
    )

    assert batch[
        "systematic_covariates"
    ].shape == (
        2,
        5,
        4,
    )

    assert batch[
        "target"
    ].shape == (
        2,
        3,
    )

    assert batch[
        "target_raw"
    ].shape == (
        2,
        3,
    )

    assert isinstance(
        batch["target_date"],
        pd.DatetimeIndex,
    )

    assert len(
        batch["target_date"]
    ) == 2


def test_loader_sampling_semantics():
    datasets = (
        make_diffolio_datasets(
            make_splits()
        )
    )

    loaders = (
        make_diffolio_dataloaders(
            datasets,
            batch_size=4,
            seed=42,
        )
    )

    assert isinstance(
        loaders.train.sampler,
        RandomSampler,
    )

    assert isinstance(
        loaders.val.sampler,
        SequentialSampler,
    )

    assert isinstance(
        loaders.test.sampler,
        SequentialSampler,
    )

    val_batch = next(
        iter(
            loaders.val
        )
    )

    assert val_batch[
        "return_history"
    ].shape == (
        4,
        5,
        3,
    )