import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data import (
    AssetCharacteristicTable,
    ReturnTable,
    SystematicCovariateTable,
)
from diffusion_portfolio.baselines.diffolio.dataset import (
    make_diffolio_windows,
    split_diffolio_windows_by_date,
)


def make_inputs(
    n_time: int = 20,
):
    dates = pd.bdate_range(
        "1999-12-20",
        periods=n_time,
    )

    n_assets = 3

    # Day index is encoded directly in the returns.
    #
    # This makes temporal alignment easy to verify.
    base = np.arange(
        n_time,
        dtype=np.float32,
    )

    raw_returns = np.column_stack(
        (
            base,
            base + 100.0,
            base + 200.0,
        )
    )

    model_returns = (
        raw_returns
        / 10.0
    )

    model_table = ReturnTable(
        dates=dates,
        returns=model_returns,
        columns=(
            "A",
            "B",
            "C",
        ),
    )

    raw_table = ReturnTable(
        dates=dates,
        returns=raw_returns,
        columns=(
            "A",
            "B",
            "C",
        ),
    )

    asset_values = np.empty(
        (
            n_time,
            n_assets,
            2,
        ),
        dtype=np.float32,
    )

    for t in range(
        n_time
    ):
        for asset in range(
            n_assets
        ):
            asset_values[
                t,
                asset,
                0,
            ] = t

            asset_values[
                t,
                asset,
                1,
            ] = (
                1000
                + 10 * t
                + asset
            )

    asset_covariates = (
        AssetCharacteristicTable(
            dates=dates,
            values=asset_values,
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
    )

    systematic_values = (
        np.column_stack(
            (
                base,
                base + 500.0,
            )
        )
    )

    systematic_covariates = (
        SystematicCovariateTable(
            dates=dates,
            values=systematic_values,
            columns=(
                "m1",
                "m2",
            ),
        )
    )

    return (
        model_table,
        raw_table,
        asset_covariates,
        systematic_covariates,
    )


def test_diffolio_window_shapes():
    (
        model,
        raw,
        asset,
        systematic,
    ) = make_inputs()

    windows = make_diffolio_windows(
        model,
        raw,
        asset,
        systematic,
        lookback=5,
    )

    assert windows.return_history.shape == (
        15,
        5,
        3,
    )

    assert (
        windows.return_history_raw.shape
        == (
            15,
            5,
            3,
        )
    )

    assert windows.asset_covariates.shape == (
        15,
        5,
        3,
        2,
    )

    assert (
        windows.systematic_covariates.shape
        == (
            15,
            5,
            2,
        )
    )

    assert windows.target.shape == (
        15,
        3,
    )

    assert windows.target_raw.shape == (
        15,
        3,
    )


def test_target_is_strictly_after_all_conditioning_data():
    (
        model,
        raw,
        asset,
        systematic,
    ) = make_inputs()

    windows = make_diffolio_windows(
        model,
        raw,
        asset,
        systematic,
        lookback=5,
    )

    # First sample:
    #
    # history indices = 0,1,2,3,4
    # target index    = 5

    np.testing.assert_allclose(
        windows.return_history_raw[
            0,
            :,
            0,
        ],
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
            ],
            dtype=np.float32,
        ),
    )

    assert (
        windows.target_raw[
            0,
            0,
        ]
        == pytest.approx(
            5.0
        )
    )

    # Asset characteristics use the SAME history indices.
    np.testing.assert_allclose(
        windows.asset_covariates[
            0,
            :,
            0,
            0,
        ],
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
            ],
            dtype=np.float32,
        ),
    )

    # Systematic covariates also use exactly those indices.
    np.testing.assert_allclose(
        windows.systematic_covariates[
            0,
            :,
            0,
        ],
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
            ],
            dtype=np.float32,
        ),
    )

    assert (
        windows.target_dates[
            0
        ]
        == model.dates[
            5
        ]
    )


def test_split_is_based_on_target_date():
    (
        model,
        raw,
        asset,
        systematic,
    ) = make_inputs(
        n_time=30
    )

    windows = make_diffolio_windows(
        model,
        raw,
        asset,
        systematic,
        lookback=5,
    )

    train_end = (
        windows.target_dates[
            9
        ]
    )

    val_end = (
        windows.target_dates[
            14
        ]
    )

    test_end = (
        windows.target_dates[
            -1
        ]
    )

    splits = (
        split_diffolio_windows_by_date(
            windows,
            train_end=train_end,
            val_end=val_end,
            test_end=test_end,
        )
    )

    assert len(
        splits.train.target_dates
    ) == 10

    assert len(
        splits.val.target_dates
    ) == 5

    assert len(
        splits.test.target_dates
    ) == 10

    assert (
        splits.train.target_dates[
            -1
        ]
        <= train_end
    )

    assert (
        splits.val.target_dates[
            0
        ]
        > train_end
    )

    assert (
        splits.test.target_dates[
            0
        ]
        > val_end
    )


def test_validation_history_can_cross_training_boundary():
    (
        model,
        raw,
        asset,
        systematic,
    ) = make_inputs(
        n_time=30
    )

    windows = make_diffolio_windows(
        model,
        raw,
        asset,
        systematic,
        lookback=5,
    )

    # Make the tenth sample the final training target.
    train_end = windows.target_dates[
        9
    ]

    val_end = windows.target_dates[
        14
    ]

    splits = (
        split_diffolio_windows_by_date(
            windows,
            train_end=train_end,
            val_end=val_end,
        )
    )

    first_val_target = (
        splits.val.target_raw[
            0,
            0,
        ]
    )

    first_val_history = (
        splits.val.return_history_raw[
            0,
            :,
            0,
        ]
    )

    # History immediately precedes target.
    assert (
        first_val_history[
            -1
        ]
        == pytest.approx(
            first_val_target
            - 1.0
        )
    )

    # Its history therefore contains observations from
    # before the validation target boundary.
    assert (
        first_val_history[
            0
        ]
        < first_val_target
    )


def test_misaligned_dates_are_rejected():
    (
        model,
        raw,
        asset,
        systematic,
    ) = make_inputs()

    shifted_systematic = (
        SystematicCovariateTable(
            dates=(
                systematic.dates
                + pd.Timedelta(
                    days=1
                )
            ),
            values=(
                systematic.values
            ),
            columns=(
                systematic.columns
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="systematic-covariate dates",
    ):
        make_diffolio_windows(
            model,
            raw,
            asset,
            shifted_systematic,
            lookback=5,
        )