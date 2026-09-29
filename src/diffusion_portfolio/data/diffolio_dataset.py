"""Aligned window construction for the Diffolio benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diffusion_portfolio.data.asset_characteristics import (
    AssetCharacteristicTable,
)
from diffusion_portfolio.data.ken_french import (
    ReturnTable,
)
from diffusion_portfolio.data.systematic_covariates import (
    SystematicCovariateTable,
)


@dataclass(frozen=True)
class DiffolioWindows:
    """Aligned supervised samples for Diffolio.

    Shapes
    ------
    return_history:
        [samples, lookback, assets]

    return_history_raw:
        [samples, lookback, assets]

    asset_covariates:
        [samples, lookback, assets, characteristics]

    systematic_covariates:
        [samples, lookback, systematic_features]

    target:
        [samples, assets]

    target_raw:
        [samples, assets]

    target_dates:
        [samples]
    """

    return_history: np.ndarray
    return_history_raw: np.ndarray

    asset_covariates: np.ndarray
    systematic_covariates: np.ndarray

    target: np.ndarray
    target_raw: np.ndarray

    target_dates: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if self.return_history.ndim != 3:
            raise ValueError(
                "return_history must have shape "
                "[samples, lookback, assets]"
            )

        if (
            self.return_history_raw.shape
            != self.return_history.shape
        ):
            raise ValueError(
                "return_history_raw must match "
                "return_history shape"
            )

        if self.asset_covariates.ndim != 4:
            raise ValueError(
                "asset_covariates must have shape "
                "[samples, lookback, assets, characteristics]"
            )

        if self.systematic_covariates.ndim != 3:
            raise ValueError(
                "systematic_covariates must have shape "
                "[samples, lookback, features]"
            )

        if self.target.ndim != 2:
            raise ValueError(
                "target must have shape [samples, assets]"
            )

        if self.target_raw.shape != self.target.shape:
            raise ValueError(
                "target_raw must match target shape"
            )

        n_samples = self.return_history.shape[0]

        if self.asset_covariates.shape[0] != n_samples:
            raise ValueError(
                "asset covariate sample count differs"
            )

        if (
            self.systematic_covariates.shape[0]
            != n_samples
        ):
            raise ValueError(
                "systematic covariate sample count differs"
            )

        if self.target.shape[0] != n_samples:
            raise ValueError(
                "target sample count differs"
            )

        if len(self.target_dates) != n_samples:
            raise ValueError(
                "target_dates length differs from sample count"
            )

        lookback = self.return_history.shape[1]

        if self.asset_covariates.shape[1] != lookback:
            raise ValueError(
                "asset covariate lookback differs"
            )

        if (
            self.systematic_covariates.shape[1]
            != lookback
        ):
            raise ValueError(
                "systematic covariate lookback differs"
            )

        n_assets = self.return_history.shape[2]

        if self.asset_covariates.shape[2] != n_assets:
            raise ValueError(
                "asset dimension differs across inputs"
            )

        if self.target.shape[1] != n_assets:
            raise ValueError(
                "target asset dimension differs"
            )

        if not self.target_dates.is_monotonic_increasing:
            raise ValueError(
                "target dates must be chronological"
            )

        if self.target_dates.has_duplicates:
            raise ValueError(
                "target dates must not contain duplicates"
            )

        arrays = (
            self.return_history,
            self.return_history_raw,
            self.asset_covariates,
            self.systematic_covariates,
            self.target,
            self.target_raw,
        )

        if not all(
            np.isfinite(array).all()
            for array in arrays
        ):
            raise ValueError(
                "Diffolio windows contain NaN "
                "or infinite values"
            )


@dataclass(frozen=True)
class DiffolioWindowSplits:
    """Chronological train/validation/test Diffolio windows."""

    train: DiffolioWindows
    val: DiffolioWindows
    test: DiffolioWindows


def _validate_alignment(
    model_returns: ReturnTable,
    raw_returns: ReturnTable,
    asset_covariates: AssetCharacteristicTable,
    systematic_covariates: SystematicCovariateTable,
) -> None:
    """Ensure every modality describes the exact same dates/assets."""

    if not model_returns.dates.equals(
        raw_returns.dates
    ):
        raise ValueError(
            "Model and raw return dates must match"
        )

    if not model_returns.dates.equals(
        asset_covariates.dates
    ):
        raise ValueError(
            "Return and asset-covariate dates must match"
        )

    if not model_returns.dates.equals(
        systematic_covariates.dates
    ):
        raise ValueError(
            "Return and systematic-covariate dates must match"
        )

    if model_returns.columns != raw_returns.columns:
        raise ValueError(
            "Model and raw return assets must match"
        )

    if model_returns.columns != asset_covariates.assets:
        raise ValueError(
            "Return assets and asset-covariate assets must match"
        )


def make_diffolio_windows(
    model_returns: ReturnTable,
    raw_returns: ReturnTable,
    asset_covariates: AssetCharacteristicTable,
    systematic_covariates: SystematicCovariateTable,
    *,
    lookback: int = 63,
) -> DiffolioWindows:
    """Build aligned one-day-ahead Diffolio samples.

    For each target index t:

        histories = [t-lookback, ..., t-1]
        target    = t

    Therefore the target date itself never appears in any
    conditioning tensor.
    """

    if lookback <= 0:
        raise ValueError(
            "lookback must be positive"
        )

    _validate_alignment(
        model_returns,
        raw_returns,
        asset_covariates,
        systematic_covariates,
    )

    n_time, n_assets = (
        model_returns.returns.shape
    )

    if n_time <= lookback:
        raise ValueError(
            "Not enough observations for requested "
            f"lookback={lookback}"
        )

    n_samples = (
        n_time
        - lookback
    )

    n_characteristics = (
        asset_covariates.values.shape[2]
    )

    n_systematic = (
        systematic_covariates.values.shape[1]
    )

    return_history = np.empty(
        (
            n_samples,
            lookback,
            n_assets,
        ),
        dtype=np.float32,
    )

    return_history_raw = np.empty_like(
        return_history
    )

    asset_history = np.empty(
        (
            n_samples,
            lookback,
            n_assets,
            n_characteristics,
        ),
        dtype=np.float32,
    )

    systematic_history = np.empty(
        (
            n_samples,
            lookback,
            n_systematic,
        ),
        dtype=np.float32,
    )

    target = np.empty(
        (
            n_samples,
            n_assets,
        ),
        dtype=np.float32,
    )

    target_raw = np.empty_like(
        target
    )

    target_dates: list[pd.Timestamp] = []

    for sample_index in range(
        n_samples
    ):
        target_index = (
            sample_index
            + lookback
        )

        history_start = (
            target_index
            - lookback
        )

        history_end = target_index

        return_history[
            sample_index
        ] = model_returns.returns[
            history_start:
            history_end
        ]

        return_history_raw[
            sample_index
        ] = raw_returns.returns[
            history_start:
            history_end
        ]

        asset_history[
            sample_index
        ] = asset_covariates.values[
            history_start:
            history_end
        ]

        systematic_history[
            sample_index
        ] = systematic_covariates.values[
            history_start:
            history_end
        ]

        target[
            sample_index
        ] = model_returns.returns[
            target_index
        ]

        target_raw[
            sample_index
        ] = raw_returns.returns[
            target_index
        ]

        target_dates.append(
            model_returns.dates[
                target_index
            ]
        )

    return DiffolioWindows(
        return_history=return_history,
        return_history_raw=return_history_raw,
        asset_covariates=asset_history,
        systematic_covariates=systematic_history,
        target=target,
        target_raw=target_raw,
        target_dates=pd.DatetimeIndex(
            target_dates
        ),
    )


def _select_windows(
    windows: DiffolioWindows,
    mask: np.ndarray,
) -> DiffolioWindows:
    """Select samples without changing their ordering."""

    return DiffolioWindows(
        return_history=windows.return_history[
            mask
        ],
        return_history_raw=windows.return_history_raw[
            mask
        ],
        asset_covariates=windows.asset_covariates[
            mask
        ],
        systematic_covariates=(
            windows.systematic_covariates[
                mask
            ]
        ),
        target=windows.target[
            mask
        ],
        target_raw=windows.target_raw[
            mask
        ],
        target_dates=windows.target_dates[
            mask
        ],
    )


def split_diffolio_windows_by_date(
    windows: DiffolioWindows,
    *,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
) -> DiffolioWindowSplits:
    """Split Diffolio samples according to TARGET date.

    Historical context is allowed to cross split boundaries.

    Example
    -------
    A validation target on 2000-01-03 may use historical
    observations from December 1999. This is valid because
    those observations were known at prediction time.
    """

    train_end = pd.Timestamp(
        train_end
    )

    val_end = pd.Timestamp(
        val_end
    )

    if train_end >= val_end:
        raise ValueError(
            "train_end must occur before val_end"
        )

    dates = windows.target_dates

    train_mask = (
        dates
        <= train_end
    )

    val_mask = (
        (dates > train_end)
        & (dates <= val_end)
    )

    if test_end is None:
        test_mask = (
            dates > val_end
        )

    else:
        test_end = pd.Timestamp(
            test_end
        )

        if val_end >= test_end:
            raise ValueError(
                "val_end must occur before test_end"
            )

        test_mask = (
            (dates > val_end)
            & (dates <= test_end)
        )

    if not train_mask.any():
        raise ValueError(
            "Training split is empty"
        )

    if not val_mask.any():
        raise ValueError(
            "Validation split is empty"
        )

    if not test_mask.any():
        raise ValueError(
            "Test split is empty"
        )

    return DiffolioWindowSplits(
        train=_select_windows(
            windows,
            train_mask,
        ),
        val=_select_windows(
            windows,
            val_mask,
        ),
        test=_select_windows(
            windows,
            test_mask,
        ),
    )