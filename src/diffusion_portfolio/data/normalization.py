"""Train-only standardization for multivariate excess returns.

The scientific MVP standardizes each asset independently using statistics
estimated only from the declared training period.

Validation and test observations never contribute to fitted statistics.

The transformation is reversible so generated samples can be converted back
to decimal excess returns before statistical or portfolio evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diffusion_portfolio.data.ken_french import ReturnTable


@dataclass(frozen=True)
class TrainStandardizer:
    """Per-asset Z-score standardizer fitted on training data only.

    For asset i:

        z[t, i] = (r[t, i] - mean[i]) / std[i]

    Parameters
    ----------
    mean:
        Per-asset training means, shape [N].

    std:
        Per-asset training standard deviations, shape [N].

    columns:
        Asset ordering associated with the statistics.
    """

    mean: np.ndarray
    std: np.ndarray
    columns: tuple[str, ...]
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError("mean must have shape [assets]")

        if self.std.ndim != 1:
            raise ValueError("std must have shape [assets]")

        if self.mean.shape != self.std.shape:
            raise ValueError(
                "mean and std must have identical shapes"
            )

        if len(self.columns) != self.mean.shape[0]:
            raise ValueError(
                "column count does not match fitted statistics"
            )

        if not np.isfinite(self.mean).all():
            raise ValueError(
                "mean contains NaN or infinite values"
            )

        if not np.isfinite(self.std).all():
            raise ValueError(
                "std contains NaN or infinite values"
            )

        if np.any(self.std <= self.eps):
            raise ValueError(
                "At least one asset has near-zero training "
                "standard deviation"
            )

    @classmethod
    def fit(
        cls,
        table: ReturnTable,
        *,
        train_end: str | pd.Timestamp,
        eps: float = 1e-8,
    ) -> "TrainStandardizer":
        """Fit statistics using observations up to ``train_end`` only."""

        train_end = pd.Timestamp(train_end)

        train_mask = table.dates <= train_end

        if not train_mask.any():
            raise ValueError(
                "No observations are available in the "
                "declared training period"
            )

        train_returns = table.returns[
            train_mask
        ].astype(np.float64)

        mean = train_returns.mean(
            axis=0
        )

        std = train_returns.std(
            axis=0,
            ddof=0,
        )

        return cls(
            mean=mean,
            std=std,
            columns=table.columns,
            eps=eps,
        )

    def transform(
        self,
        table: ReturnTable,
    ) -> ReturnTable:
        """Standardize a return table using the fitted statistics."""

        if table.columns != self.columns:
            raise ValueError(
                "Asset columns do not match fitted standardizer"
            )

        standardized = (
            table.returns.astype(np.float64)
            - self.mean
        ) / self.std

        if not np.isfinite(
            standardized
        ).all():
            raise ValueError(
                "Standardization produced NaN or infinite values"
            )

        return ReturnTable(
            dates=table.dates.copy(),
            returns=standardized.astype(
                np.float32
            ),
            columns=table.columns,
        )

    def inverse_transform(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Convert standardized values back to decimal returns.

        The final dimension must be the asset dimension.

        Supported examples include:

            [T, N]
            [B, H, N]
            [samples, B, N]
        """

        values = np.asarray(values)

        if values.shape[-1] != len(
            self.columns
        ):
            raise ValueError(
                "Last dimension does not match asset count"
            )

        restored = (
            values.astype(np.float64)
            * self.std
            + self.mean
        )

        if not np.isfinite(restored).all():
            raise ValueError(
                "Inverse transformation produced "
                "NaN or infinite values"
            )

        return restored.astype(
            np.float32
        )