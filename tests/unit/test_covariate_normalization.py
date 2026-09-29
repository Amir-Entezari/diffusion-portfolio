import numpy as np
import pandas as pd

from diffusion_portfolio.data import (
    AssetCharacteristicTable,
    CovariateStandardizer,
    SystematicCovariateTable,
)


def make_tables():
    dates = pd.bdate_range(
        "1999-12-20",
        periods=20,
    )

    rng = np.random.default_rng(
        123
    )

    asset = AssetCharacteristicTable(
        dates=dates,
        values=rng.normal(
            size=(
                len(dates),
                3,
                2,
            )
        ).astype(
            np.float32
        ),
        assets=(
            "A",
            "B",
            "C",
        ),
        characteristics=(
            "x",
            "y",
        ),
    )

    systematic = SystematicCovariateTable(
        dates=dates,
        values=rng.normal(
            size=(
                len(dates),
                2,
            )
        ).astype(
            np.float32
        ),
        columns=(
            "m1",
            "m2",
        ),
    )

    return (
        asset,
        systematic,
    )


def test_training_covariates_standardize_to_zero_mean_unit_std():
    asset, systematic = make_tables()

    train_end = asset.dates[
        9
    ]

    scaler = CovariateStandardizer.fit(
        asset,
        systematic,
        train_end=train_end,
    )

    asset_scaled = (
        scaler.transform_asset(
            asset
        )
    )

    systematic_scaled = (
        scaler.transform_systematic(
            systematic
        )
    )

    train_mask = (
        asset.dates
        <= train_end
    )

    asset_train = (
        asset_scaled.values[
            train_mask
        ]
    )

    systematic_train = (
        systematic_scaled.values[
            train_mask
        ]
    )

    np.testing.assert_allclose(
        asset_train.mean(
            axis=(0, 1)
        ),
        0.0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        asset_train.std(
            axis=(0, 1),
            ddof=0,
        ),
        1.0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        systematic_train.mean(
            axis=0
        ),
        0.0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        systematic_train.std(
            axis=0,
            ddof=0,
        ),
        1.0,
        atol=1e-6,
    )


def test_future_values_cannot_change_training_statistics():
    asset, systematic = make_tables()

    train_end = asset.dates[
        9
    ]

    scaler_a = CovariateStandardizer.fit(
        asset,
        systematic,
        train_end=train_end,
    )

    corrupted_asset_values = (
        asset.values.copy()
    )

    corrupted_systematic_values = (
        systematic.values.copy()
    )

    corrupted_asset_values[
        10:
    ] += 10000.0

    corrupted_systematic_values[
        10:
    ] -= 10000.0

    corrupted_asset = AssetCharacteristicTable(
        dates=asset.dates,
        values=corrupted_asset_values,
        assets=asset.assets,
        characteristics=asset.characteristics,
    )

    corrupted_systematic = SystematicCovariateTable(
        dates=systematic.dates,
        values=corrupted_systematic_values,
        columns=systematic.columns,
    )

    scaler_b = CovariateStandardizer.fit(
        corrupted_asset,
        corrupted_systematic,
        train_end=train_end,
    )

    np.testing.assert_allclose(
        scaler_a.asset_mean,
        scaler_b.asset_mean,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        scaler_a.asset_std,
        scaler_b.asset_std,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        scaler_a.systematic_mean,
        scaler_b.systematic_mean,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        scaler_a.systematic_std,
        scaler_b.systematic_std,
        rtol=0.0,
        atol=0.0,
    )


def test_transform_preserves_shapes_and_dates():
    asset, systematic = make_tables()

    scaler = CovariateStandardizer.fit(
        asset,
        systematic,
        train_end=asset.dates[9],
    )

    asset_scaled = (
        scaler.transform_asset(
            asset
        )
    )

    systematic_scaled = (
        scaler.transform_systematic(
            systematic
        )
    )

    assert (
        asset_scaled.values.shape
        == asset.values.shape
    )

    assert (
        systematic_scaled.values.shape
        == systematic.values.shape
    )

    assert (
        asset_scaled.dates.equals(
            asset.dates
        )
    )

    assert (
        systematic_scaled.dates.equals(
            systematic.dates
        )
    )