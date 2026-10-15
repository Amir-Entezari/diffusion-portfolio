"""Continuity semantics using artificial panels, independent of any generator."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import ReturnTable, make_windows
from diffusion_portfolio.data import preparation
from diffusion_portfolio.data.preparation import save_standardizer


def panel(dates):
    values = np.arange(len(dates) * 3, dtype=np.float32).reshape(-1, 3) / 1000
    return ReturnTable(dates, values, ("a", "b", "c"))


@pytest.mark.parametrize("horizon", [1, 3])
def test_no_history_or_target_can_cross_a_multiyear_gap(horizon):
    before = pd.bdate_range("1999-06-23", periods=6)
    after = pd.bdate_range("2004-07-01", periods=8)
    table = panel(before.append(after))
    with pytest.warns(UserWarning, match="continuity"):
        windows = make_windows(table, lookback=3, horizon=horizon)

    # The expected windows come from each contiguous block separately. This
    # also catches gaps inside a multi-step target or at the history/target join.
    starts = [t for lo, hi in ((0, 6), (6, 14))
              for t in range(lo + 3, hi - horizon + 1)]
    np.testing.assert_array_equal(windows.history, np.stack([table.returns[t-3:t] for t in starts]))
    np.testing.assert_array_equal(windows.target, np.stack([table.returns[t:t+horizon] for t in starts]))
    assert windows.target_dates.equals(table.dates[starts])
    assert windows.target_dates[starts.index(9)] == after[3]


@pytest.mark.parametrize("dates", [
    # Ordinary weekends, Christmas and New Year closures.
    ["2023-12-21", "2023-12-22", "2023-12-26", "2023-12-27",
     "2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"],
    # The seven-day gap observed in the validated KF12 calendar.
    ["2001-09-07", "2001-09-10", "2001-09-17", "2001-09-18", "2001-09-19"],
])
def test_market_closures_within_limit_preserve_windows_exactly(dates):
    table = panel(pd.to_datetime(dates))
    windows = make_windows(table, lookback=2)
    np.testing.assert_array_equal(windows.history, np.stack([
        table.returns[i:i+2] for i in range(len(dates)-2)
    ]))
    np.testing.assert_array_equal(windows.target[:, 0], table.returns[2:])
    assert windows.target_dates.equals(table.dates[2:])


def test_segments_too_short_for_a_complete_window_fail_explicitly():
    dates = pd.date_range("2000-01-01", periods=3).append(pd.date_range("2010-01-01", periods=3))
    with pytest.raises(ValueError, match="No windows remain.*continuity"):
        make_windows(panel(dates), lookback=4)


@pytest.mark.parametrize("limit", [0, -1, None, "7", True, 2.5])
def test_invalid_continuity_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="positive integer"):
        make_windows(panel(pd.bdate_range("2020-01-01", periods=8)), lookback=2, max_gap_days=limit)


def test_preparation_segments_both_spaces_and_reuses_saved_scaler(tmp_path, monkeypatch):
    before = pd.date_range("2020-01-01", periods=12)
    after = pd.date_range(before[-1] + pd.Timedelta(days=6), periods=14)
    table = panel(before.append(after))
    cfg = load_config(Path(__file__).parents[2] / "configs/cde.yaml")
    cfg = replace(cfg, data=replace(
        cfg.data, dataset="test_panel", subtract_risk_free=False, lookback=3,
        max_gap_days=5, sample_start=str(before[0].date()),
        train_end=str(before[-1].date()), val_end=str(after[9].date()),
        sample_end=str(after[-1].date()),
    ))
    monkeypatch.setattr(preparation, "load_return_table", lambda *a, **k: table)
    monkeypatch.setattr(preparation, "load_daily_risk_free", lambda: pytest.fail("RF must remain disabled"))
    with pytest.warns(UserWarning, match="continuity"):
        datasets, scaler = preparation.prepare_return_data(cfg)
    assert (len(datasets.train), len(datasets.val), len(datasets.test)) == (9, 7, 4)
    assert datasets.val.raw_windows.target_dates[0] == after[3]
    np.testing.assert_array_equal(datasets.val.raw_windows.history[0], table.returns[12:15])
    np.testing.assert_array_equal(scaler.mean, table.returns[:12].astype(np.float64).mean(0))
    for dataset in (datasets.train, datasets.val, datasets.test):
        assert dataset.model_windows.target_dates.equals(dataset.raw_windows.target_dates)
        np.testing.assert_allclose(
            scaler.inverse_transform(dataset.model_windows.history), dataset.raw_windows.history,
            atol=1e-8,
        )
    path = tmp_path / "scaler.npz"
    save_standardizer(path, scaler)
    monkeypatch.setattr(preparation.TrainStandardizer, "fit", lambda *a, **k: pytest.fail("Saved scaler was refitted"))
    with pytest.warns(UserWarning, match="continuity"):
        restored, _ = preparation.prepare_return_data(cfg, standardizer_path=path)
    np.testing.assert_array_equal(restored.val.model_windows.history, datasets.val.model_windows.history)
