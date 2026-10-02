import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data import (
    INDUSTRY_COLUMNS,
    ReturnTable,
    TrainStandardizer,
)


def make_table(
    n_days: int = 10,
) -> ReturnTable:
    dates = pd.date_range(
        "2020-01-01",
        periods=n_days,
        freq="B",
    )

    base = np.arange(
        n_days,
        dtype=np.float32,
    )[:, None]

    offsets = np.arange(
        len(INDUSTRY_COLUMNS),
        dtype=np.float32,
    )[None, :]

    returns = (
        base * 0.01
        + offsets * 0.001
    )

    return ReturnTable(
        dates=dates,
        returns=returns,
        columns=INDUSTRY_COLUMNS,
    )


def test_standardizer_uses_training_period_only():
    table = make_table(10)

    train_end = table.dates[5]

    scaler = TrainStandardizer.fit(
        table,
        train_end=train_end,
    )

    expected = table.returns[:6]

    np.testing.assert_allclose(
        scaler.mean,
        expected.mean(axis=0),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        scaler.std,
        expected.std(axis=0),
        rtol=1e-6,
    )


def test_future_extreme_values_do_not_change_fitted_statistics():
    table = make_table(10)

    train_end = table.dates[5]

    scaler_a = TrainStandardizer.fit(
        table,
        train_end=train_end,
    )

    altered_returns = table.returns.copy()

    altered_returns[6:] = 1_000_000.0

    altered = ReturnTable(
        dates=table.dates,
        returns=altered_returns,
        columns=table.columns,
    )

    scaler_b = TrainStandardizer.fit(
        altered,
        train_end=train_end,
    )

    np.testing.assert_allclose(
        scaler_a.mean,
        scaler_b.mean,
    )

    np.testing.assert_allclose(
        scaler_a.std,
        scaler_b.std,
    )


def test_standardized_training_data_has_zero_mean_unit_std():
    table = make_table(10)

    train_end = table.dates[5]

    scaler = TrainStandardizer.fit(
        table,
        train_end=train_end,
    )

    transformed = scaler.transform(
        table
    )

    train_values = transformed.returns[:6]

    np.testing.assert_allclose(
        train_values.mean(axis=0),
        np.zeros(12),
        atol=1e-6,
    )

    np.testing.assert_allclose(
        train_values.std(axis=0),
        np.ones(12),
        atol=1e-6,
    )


def test_inverse_transform_recovers_original_returns():
    table = make_table(10)

    scaler = TrainStandardizer.fit(
        table,
        train_end=table.dates[5],
    )

    transformed = scaler.transform(
        table
    )

    recovered = scaler.inverse_transform(
        transformed.returns
    )

    np.testing.assert_allclose(
        recovered,
        table.returns,
        atol=1e-7,
    )


def test_transform_preserves_dates_and_columns():
    table = make_table(10)

    scaler = TrainStandardizer.fit(
        table,
        train_end=table.dates[5],
    )

    transformed = scaler.transform(
        table
    )

    assert transformed.dates.equals(
        table.dates
    )

    assert transformed.columns == table.columns


def test_inverse_transform_supports_scenario_batches():
    table = make_table(10)

    scaler = TrainStandardizer.fit(
        table,
        train_end=table.dates[5],
    )

    values = np.zeros(
        (32, 8, 12),
        dtype=np.float32,
    )

    restored = scaler.inverse_transform(
        values
    )

    assert restored.shape == (
        32,
        8,
        12,
    )

    np.testing.assert_allclose(
        restored[0, 0],
        scaler.mean,
        rtol=1e-6,
    )


def test_unknown_asset_dimension_is_rejected():
    table = make_table(10)

    scaler = TrainStandardizer.fit(
        table,
        train_end=table.dates[5],
    )

    with pytest.raises(
        ValueError,
        match="asset count",
    ):
        scaler.inverse_transform(
            np.zeros(
                (5, 11),
                dtype=np.float32,
            )
        )
        
        
def test_identity_standardizer_preserves_raw_returns():
    table = make_table(
        10
    )

    scaler = TrainStandardizer.identity(
        table.columns
    )

    transformed = scaler.transform(
        table
    )

    recovered = scaler.inverse_transform(
        transformed.returns
    )

    np.testing.assert_allclose(
        scaler.mean,
        np.zeros(
            len(
                table.columns
            )
        ),
    )

    np.testing.assert_allclose(
        scaler.std,
        np.ones(
            len(
                table.columns
            )
        ),
    )

    np.testing.assert_allclose(
        transformed.returns,
        table.returns,
        atol=0.0,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        recovered,
        table.returns,
        atol=0.0,
        rtol=0.0,
    )