"""Chronological splitting of windowed financial time series.

The split is performed AFTER sliding windows are constructed.

This allows, for example, the first validation sample to use historical
observations from the end of the training period. That is not leakage:
those observations would genuinely be known at validation time.

Targets themselves remain chronologically separated.
"""

from __future__ import annotations

import pandas as pd

from dataclasses import dataclass

from diffusion_portfolio.data.windows import WindowedReturns


@dataclass(frozen=True)
class WindowSplits:
    """Chronological train/validation/test partitions."""

    train: WindowedReturns
    val: WindowedReturns
    test: WindowedReturns


def _slice_windows(
    windows: WindowedReturns,
    start: int,
    end: int,
) -> WindowedReturns:
    """Slice WindowedReturns without changing sample ordering."""

    return WindowedReturns(
        history=windows.history[start:end],
        target=windows.target[start:end],
        target_dates=windows.target_dates[start:end],
    )


def split_windows_chronologically(
    windows: WindowedReturns,
    train_ratio: float,
    val_ratio: float,
    *,
    purge_target_overlap: bool = True,
) -> WindowSplits:
    """Split windowed samples chronologically.

    Parameters
    ----------
    windows:
        Chronologically ordered history/target samples.

    train_ratio:
        Fraction assigned to training.

    val_ratio:
        Fraction assigned to validation.

        The remaining fraction becomes the test split.

    purge_target_overlap:
        If the prediction horizon is greater than one, adjacent samples can
        contain overlapping future target observations.

        Example with horizon=3:

            sample i target:     [t, t+1, t+2]
            sample i+1 target:      [t+1, t+2, t+3]

        At split boundaries this would mean train and validation labels share
        future observations.

        When enabled, ``horizon - 1`` samples are discarded at each boundary
        to prevent this target overlap.

        For the current MVP horizon=1, the purge gap is zero.

    Returns
    -------
    WindowSplits
        Train, validation and test samples in chronological order.
    """

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")

    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n_samples = windows.history.shape[0]

    if n_samples < 3:
        raise ValueError(
            "At least three windowed samples are required "
            "for train/val/test splitting."
        )

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    if train_end == 0:
        raise ValueError("Training split would be empty")

    if val_end <= train_end:
        raise ValueError("Validation split would be empty")

    if val_end >= n_samples:
        raise ValueError("Test split would be empty")

    horizon = windows.target.shape[1]

    purge_gap = (
        max(0, horizon - 1)
        if purge_target_overlap
        else 0
    )

    val_start = train_end + purge_gap
    test_start = val_end + purge_gap

    if val_start >= val_end:
        raise ValueError(
            "Validation split became empty after purging "
            "overlapping targets."
        )

    if test_start >= n_samples:
        raise ValueError(
            "Test split became empty after purging "
            "overlapping targets."
        )

    train = _slice_windows(
        windows,
        0,
        train_end,
    )

    val = _slice_windows(
        windows,
        val_start,
        val_end,
    )

    test = _slice_windows(
        windows,
        test_start,
        n_samples,
    )

    return WindowSplits(
        train=train,
        val=val,
        test=test,
    )
    
    

def split_windows_by_date(
    windows: WindowedReturns,
    *,
    train_end: str,
    val_end: str,
    test_end: str | None = None,
) -> WindowSplits:
    """Split samples according to their target dates.

    Boundaries are inclusive:

        train: target_date <= train_end

        val:
            train_end < target_date <= val_end

        test:
            val_end < target_date <= test_end

    If ``test_end`` is None, every observation after ``val_end`` is
    assigned to the test set.

    Notes
    -----
    Splitting is based on target dates, not history dates. A validation
    or test sample may therefore use earlier observations in its history,
    which is valid because those observations would have been available
    at prediction time.
    """

    horizon = windows.target.shape[1]

    if horizon != 1:
        raise ValueError(
            "split_windows_by_date currently supports horizon=1 only; "
            "multi-step targets require explicit boundary purging"
        )
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    if train_end_ts >= val_end_ts:
        raise ValueError(
            "train_end must occur before val_end"
        )

    dates = windows.target_dates

    train_mask = dates <= train_end_ts

    val_mask = (
        (dates > train_end_ts)
        & (dates <= val_end_ts)
    )

    if test_end is None:
        test_mask = dates > val_end_ts
    else:
        test_end_ts = pd.Timestamp(test_end)

        if val_end_ts >= test_end_ts:
            raise ValueError(
                "val_end must occur before test_end"
            )

        test_mask = (
            (dates > val_end_ts)
            & (dates <= test_end_ts)
        )

    if not train_mask.any():
        raise ValueError("Training split is empty")

    if not val_mask.any():
        raise ValueError("Validation split is empty")

    if not test_mask.any():
        raise ValueError("Test split is empty")

    def select(mask) -> WindowedReturns:
        return WindowedReturns(
            history=windows.history[mask],
            target=windows.target[mask],
            target_dates=windows.target_dates[mask],
        )

    return WindowSplits(
        train=select(train_mask),
        val=select(val_mask),
        test=select(test_mask),
    )