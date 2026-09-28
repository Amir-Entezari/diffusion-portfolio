import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
)

from diffusion_portfolio.data import (
    INDUSTRY_COLUMNS,
    ReturnTable,
    TrainStandardizer,
    build_window_datasets,
    make_dataloaders,
)


def make_table(
    n_days: int = 200,
) -> ReturnTable:
    dates = pd.date_range(
        "2020-01-01",
        periods=n_days,
        freq="B",
    )

    time = np.arange(
        n_days,
        dtype=np.float32,
    )[:, None]

    assets = np.arange(
        len(INDUSTRY_COLUMNS),
        dtype=np.float32,
    )[None, :]

    returns = (
        time * 0.001
        + assets * 0.0001
    )

    return ReturnTable(
        dates=dates,
        returns=returns,
        columns=INDUSTRY_COLUMNS,
    )


def build_test_datasets():
    raw = make_table()

    train_end = raw.dates[119]
    val_end = raw.dates[159]

    scaler = TrainStandardizer.fit(
        raw,
        train_end=train_end,
    )

    model = scaler.transform(
        raw
    )

    datasets = build_window_datasets(
        model,
        raw,
        lookback=20,
        horizon=1,
        train_end=str(train_end.date()),
        val_end=str(val_end.date()),
        test_end=str(raw.dates[-1].date()),
    )

    return raw, scaler, datasets


def test_dataset_sample_contract():
    _, _, datasets = build_test_datasets()

    sample = datasets.train[0]

    assert sample["history"].shape == (
        20,
        12,
    )

    assert sample["target"].shape == (
        1,
        12,
    )

    assert sample["history_raw"].shape == (
        20,
        12,
    )

    assert sample["target_raw"].shape == (
        1,
        12,
    )

    assert isinstance(
        sample["target_date"],
        pd.Timestamp,
    )


def test_dataset_tensors_are_float32():
    _, _, datasets = build_test_datasets()

    sample = datasets.train[0]

    assert sample["history"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32
    assert sample["history_raw"].dtype == torch.float32
    assert sample["target_raw"].dtype == torch.float32


def test_raw_target_matches_inverse_transformed_model_target():
    _, scaler, datasets = build_test_datasets()

    sample = datasets.val[0]

    recovered = scaler.inverse_transform(
        sample["target"].numpy()
    )

    np.testing.assert_allclose(
        recovered,
        sample["target_raw"].numpy(),
        atol=1e-7,
    )


def test_model_and_raw_date_mismatch_is_rejected():
    raw = make_table()

    altered = ReturnTable(
        dates=raw.dates + pd.Timedelta(days=1),
        returns=raw.returns.copy(),
        columns=raw.columns,
    )

    with pytest.raises(
        ValueError,
        match="identical dates",
    ):
        build_window_datasets(
            altered,
            raw,
            lookback=20,
            horizon=1,
            train_end=str(raw.dates[119].date()),
            val_end=str(raw.dates[159].date()),
            test_end=str(raw.dates[-1].date()),
        )


def test_dataloader_batch_contract():
    _, _, datasets = build_test_datasets()

    loaders = make_dataloaders(
        datasets,
        batch_size=16,
        seed=42,
    )

    batch = next(
        iter(loaders.train)
    )

    assert batch["history"].shape == (
        16,
        20,
        12,
    )

    assert batch["target"].shape == (
        16,
        1,
        12,
    )

    assert batch["history_raw"].shape == (
        16,
        20,
        12,
    )

    assert batch["target_raw"].shape == (
        16,
        1,
        12,
    )

    assert len(
        batch["target_date"]
    ) == 16


def test_train_loader_shuffles_but_eval_loaders_do_not():
    _, _, datasets = build_test_datasets()

    loaders = make_dataloaders(
        datasets,
        batch_size=16,
        seed=42,
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


def test_validation_loader_preserves_chronological_dates():
    _, _, datasets = build_test_datasets()

    loaders = make_dataloaders(
        datasets,
        batch_size=16,
        seed=42,
    )

    collected_dates = []

    for batch in loaders.val:
        collected_dates.extend(
            batch["target_date"].tolist()
        )

    assert collected_dates == sorted(
        collected_dates
    )