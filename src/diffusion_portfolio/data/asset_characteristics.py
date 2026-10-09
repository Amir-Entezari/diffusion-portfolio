"""Diffolio-style asset characteristics from daily excess returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diffusion_portfolio.data.sources.fama_french import (
    FactorTable,
)
from diffusion_portfolio.data.core import (
    ReturnTable,
)


CHARACTERISTIC_COLUMNS = (
    "mom1m",
    "mom6m",
    "mom12m",
    "mom36m",
    "chmom",
    "retvol",
    "maxret",
    "beta",
    "betasq",
    "idiovol",
)


@dataclass(frozen=True)
class AssetCharacteristicTable:
    """Daily asset-specific characteristics.

    values has shape:

        [time, assets, characteristics]
    """

    dates: pd.DatetimeIndex
    values: np.ndarray
    assets: tuple[str, ...]
    characteristics: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(
                "values must have shape "
                "[time, assets, characteristics]"
            )

        if len(self.dates) != self.values.shape[0]:
            raise ValueError(
                "dates length does not match time dimension"
            )

        if len(self.assets) != self.values.shape[1]:
            raise ValueError(
                "asset count does not match values"
            )

        if (
            len(self.characteristics)
            != self.values.shape[2]
        ):
            raise ValueError(
                "characteristic count does not match values"
            )

        if not self.dates.is_monotonic_increasing:
            raise ValueError(
                "dates must be chronological"
            )

        if self.dates.has_duplicates:
            raise ValueError(
                "dates must not contain duplicates"
            )

        if not np.isfinite(
            self.values
        ).all():
            raise ValueError(
                "characteristics contain NaN or infinite values"
            )


def _cumulative_return(
    returns: np.ndarray,
) -> np.ndarray:
    """Compound a [time, assets] return window."""

    if np.any(
        returns <= -1.0
    ):
        raise ValueError(
            "Cannot compound returns <= -100%"
        )

    return (
        np.prod(
            1.0 + returns,
            axis=0,
        )
        - 1.0
    )


def build_asset_characteristics(
    excess_returns: ReturnTable,
    factors: FactorTable,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> AssetCharacteristicTable:
    """Construct the ten Diffolio asset characteristics.

    Characteristics are computed at date t using information
    available through date t only.

    The output may therefore be used as conditioning information
    when forecasting the return at t+1.

    Notes
    -----
    The longest required history is 756 trading days because of
    ``mom36m``.

    CAPM beta and Fama-French idiosyncratic volatility use rolling
    252-trading-day windows.
    """

    dates = excess_returns.dates

    returns = (
        excess_returns
        .returns
        .astype(
            np.float64,
            copy=False,
        )
    )

    if len(dates) == 0:
        raise ValueError(
            "excess return table is empty"
        )

    if np.any(
        returns <= -1.0
    ):
        raise ValueError(
            "excess returns contain values <= -100%"
        )

    start = pd.Timestamp(
        start
    )

    if end is None:
        end = dates[-1]
    else:
        end = pd.Timestamp(
            end
        )

    if end < start:
        raise ValueError(
            "end must not precede start"
        )

    # ---------------------------------------------------------
    # Align Fama-French factors exactly to return dates.
    #
    # No forward filling is allowed here.
    # ---------------------------------------------------------
    factor_indices = (
        factors.dates.get_indexer(
            dates
        )
    )

    if np.any(
        factor_indices < 0
    ):
        missing_position = int(
            np.flatnonzero(
                factor_indices < 0
            )[0]
        )

        raise ValueError(
            "Fama-French factors do not contain "
            "every asset-return date; first missing date is "
            f"{dates[missing_position].date()}"
        )

    aligned_factors = (
        factors
        .returns[
            factor_indices
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    try:
        market_index = (
            factors.columns.index(
                "Mkt-RF"
            )
        )

        smb_index = (
            factors.columns.index(
                "SMB"
            )
        )

        hml_index = (
            factors.columns.index(
                "HML"
            )
        )

    except ValueError as exc:
        raise ValueError(
            "Factor table must contain Mkt-RF, SMB, and HML"
        ) from exc

    output_mask = (
        (dates >= start)
        & (dates <= end)
    )

    output_indices = np.flatnonzero(
        output_mask
    )

    if len(output_indices) == 0:
        raise ValueError(
            "Requested date range contains no observations"
        )

    # Index 755 is the first position containing 756
    # observations including the current date.
    first_output_index = int(
        output_indices[0]
    )

    if first_output_index < 755:
        raise ValueError(
            "At least 756 observations ending at the first "
            "requested date are required for mom36m"
        )

    n_output = len(
        output_indices
    )

    n_assets = returns.shape[1]

    values = np.empty(
        (
            n_output,
            n_assets,
            len(
                CHARACTERISTIC_COLUMNS
            ),
        ),
        dtype=np.float64,
    )

    # Characteristic column positions.
    mom1m_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "mom1m"
        )
    )

    mom6m_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "mom6m"
        )
    )

    mom12m_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "mom12m"
        )
    )

    mom36m_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "mom36m"
        )
    )

    chmom_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "chmom"
        )
    )

    retvol_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "retvol"
        )
    )

    maxret_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "maxret"
        )
    )

    beta_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "beta"
        )
    )

    betasq_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "betasq"
        )
    )

    idiovol_idx = (
        CHARACTERISTIC_COLUMNS.index(
            "idiovol"
        )
    )

    for output_position, t in enumerate(
        output_indices
    ):
        # -----------------------------------------------------
        # Momentum
        # -----------------------------------------------------
        mom1m = _cumulative_return(
            returns[
                t - 20 :
                t + 1
            ]
        )

        mom6m = _cumulative_return(
            returns[
                t - 125 :
                t + 1
            ]
        )

        mom12m = _cumulative_return(
            returns[
                t - 251 :
                t + 1
            ]
        )

        mom36m = _cumulative_return(
            returns[
                t - 755 :
                t + 1
            ]
        )

        # Previous non-overlapping 6-month period:
        #
        # [t-251, ..., t-126]
        previous_mom6m = (
            _cumulative_return(
                returns[
                    t - 251 :
                    t - 125
                ]
            )
        )

        chmom = (
            mom6m
            - previous_mom6m
        )

        # -----------------------------------------------------
        # 21-day return volatility + maximum return
        #
        # ddof=1 gives denominator 20 for 21 observations,
        # matching the formula in Diffolio Appendix B.2.
        # -----------------------------------------------------
        short_window = returns[
            t - 20 :
            t + 1
        ]

        retvol = np.std(
            short_window,
            axis=0,
            ddof=1,
        )

        maxret = np.max(
            short_window,
            axis=0,
        )

        # -----------------------------------------------------
        # 252-day factor window
        # -----------------------------------------------------
        return_window = returns[
            t - 251 :
            t + 1
        ]

        factor_window = (
            aligned_factors[
                t - 251 :
                t + 1
            ]
        )

        market = factor_window[
            :,
            market_index,
        ]

        # -----------------------------------------------------
        # CAPM beta
        #
        # OLS with intercept:
        #
        # beta = Cov(r, market) / Var(market)
        # -----------------------------------------------------
        centered_market = (
            market
            - market.mean()
        )

        centered_returns = (
            return_window
            - return_window.mean(
                axis=0,
                keepdims=True,
            )
        )

        market_ss = float(
            centered_market
            @ centered_market
        )

        if (
            market_ss
            <= np.finfo(
                np.float64
            ).eps
        ):
            raise ValueError(
                "Market factor has zero variance in "
                f"252-day window ending {dates[t].date()}"
            )

        beta = (
            centered_market
            @ centered_returns
        ) / market_ss

        betasq = (
            beta ** 2
        )

        # -----------------------------------------------------
        # Fama-French 3-factor residual volatility
        #
        # One regression is solved for all assets:
        #
        # Y = X B + E
        #
        # X: [252, 4]
        # Y: [252, assets]
        # -----------------------------------------------------
        design = np.column_stack(
            (
                np.ones(
                    252,
                    dtype=np.float64,
                ),
                factor_window[
                    :,
                    market_index,
                ],
                factor_window[
                    :,
                    smb_index,
                ],
                factor_window[
                    :,
                    hml_index,
                ],
            )
        )

        coefficients, _, _, _ = (
            np.linalg.lstsq(
                design,
                return_window,
                rcond=None,
            )
        )

        residuals = (
            return_window
            - design
            @ coefficients
        )

        # The paper specifies the standard deviation of
        # regression residuals. We use sample standard
        # deviation consistently with retvol.
        idiovol = np.std(
            residuals,
            axis=0,
            ddof=1,
        )

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------
        values[
            output_position,
            :,
            mom1m_idx,
        ] = mom1m

        values[
            output_position,
            :,
            mom6m_idx,
        ] = mom6m

        values[
            output_position,
            :,
            mom12m_idx,
        ] = mom12m

        values[
            output_position,
            :,
            mom36m_idx,
        ] = mom36m

        values[
            output_position,
            :,
            chmom_idx,
        ] = chmom

        values[
            output_position,
            :,
            retvol_idx,
        ] = retvol

        values[
            output_position,
            :,
            maxret_idx,
        ] = maxret

        values[
            output_position,
            :,
            beta_idx,
        ] = beta

        values[
            output_position,
            :,
            betasq_idx,
        ] = betasq

        values[
            output_position,
            :,
            idiovol_idx,
        ] = idiovol

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Asset characteristic construction produced "
            "NaN or infinite values"
        )

    return AssetCharacteristicTable(
        dates=pd.DatetimeIndex(
            dates[
                output_indices
            ]
        ),
        values=values.astype(
            np.float32
        ),
        assets=(
            excess_returns.columns
        ),
        characteristics=(
            CHARACTERISTIC_COLUMNS
        ),
    )