"""Train-only normalization for Diffolio covariates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diffusion_portfolio.data.asset_characteristics import (
    AssetCharacteristicTable,
)
from diffusion_portfolio.data.systematic_covariates import (
    SystematicCovariateTable,
)


@dataclass(frozen=True)
class CovariateStandardizer:
    """Training-set standardization statistics."""

    asset_mean: np.ndarray
    asset_std: np.ndarray

    systematic_mean: np.ndarray
    systematic_std: np.ndarray

    asset_columns: tuple[str, ...]
    systematic_columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        asset_characteristics: AssetCharacteristicTable,
        systematic_covariates: SystematicCovariateTable,
        *,
        train_end: str | pd.Timestamp,
    ) -> "CovariateStandardizer":
        """Fit normalization statistics using training dates only."""

        if not asset_characteristics.dates.equals(
            systematic_covariates.dates
        ):
            raise ValueError(
                "Asset and systematic covariates must "
                "have identical dates"
            )

        train_end = pd.Timestamp(
            train_end
        )

        train_mask = (
            asset_characteristics.dates
            <= train_end
        )

        if not train_mask.any():
            raise ValueError(
                "No observations fall inside the training period"
            )

        asset_train = (
            asset_characteristics.values[
                train_mask
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        systematic_train = (
            systematic_covariates.values[
                train_mask
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        # One statistic per characteristic, pooling
        # training dates and assets.
        asset_mean = asset_train.mean(
            axis=(0, 1)
        )

        asset_std = asset_train.std(
            axis=(0, 1),
            ddof=0,
        )

        # One statistic per systematic variable.
        systematic_mean = (
            systematic_train.mean(
                axis=0
            )
        )

        systematic_std = (
            systematic_train.std(
                axis=0,
                ddof=0,
            )
        )

        if np.any(
            asset_std <= 0.0
        ):
            raise ValueError(
                "At least one asset characteristic "
                "has zero training standard deviation"
            )

        if np.any(
            systematic_std <= 0.0
        ):
            raise ValueError(
                "At least one systematic covariate "
                "has zero training standard deviation"
            )

        return cls(
            asset_mean=asset_mean.astype(
                np.float32
            ),
            asset_std=asset_std.astype(
                np.float32
            ),
            systematic_mean=(
                systematic_mean.astype(
                    np.float32
                )
            ),
            systematic_std=(
                systematic_std.astype(
                    np.float32
                )
            ),
            asset_columns=(
                asset_characteristics.characteristics
            ),
            systematic_columns=(
                systematic_covariates.columns
            ),
        )

    def transform_asset(
        self,
        table: AssetCharacteristicTable,
    ) -> AssetCharacteristicTable:
        """Standardize asset characteristics."""

        if (
            table.characteristics
            != self.asset_columns
        ):
            raise ValueError(
                "Asset characteristic columns do not match"
            )

        values = (
            table.values.astype(
                np.float64,
                copy=False,
            )
            - self.asset_mean[
                None,
                None,
                :,
            ]
        ) / self.asset_std[
            None,
            None,
            :,
        ]

        return AssetCharacteristicTable(
            dates=table.dates.copy(),
            values=values.astype(
                np.float32
            ),
            assets=table.assets,
            characteristics=table.characteristics,
        )

    def transform_systematic(
        self,
        table: SystematicCovariateTable,
    ) -> SystematicCovariateTable:
        """Standardize systematic covariates."""

        if (
            table.columns
            != self.systematic_columns
        ):
            raise ValueError(
                "Systematic columns do not match"
            )

        values = (
            table.values.astype(
                np.float64,
                copy=False,
            )
            - self.systematic_mean[
                None,
                :,
            ]
        ) / self.systematic_std[
            None,
            :,
        ]

        return SystematicCovariateTable(
            dates=table.dates.copy(),
            values=values.astype(
                np.float32
            ),
            columns=table.columns,
        )