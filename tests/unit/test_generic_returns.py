from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data import ReturnTable, TrainStandardizer, build_window_datasets
from diffusion_portfolio.data.preparation import load_standardizer, save_standardizer
from diffusion_portfolio.data.registry import load_return_table
from diffusion_portfolio.data.sources.kenneth_french import INDUSTRY_COLUMNS


@pytest.mark.parametrize("n_assets", [3, 17])
def test_generic_panel_normalization_and_windows(n_assets, tmp_path):
    dates = pd.bdate_range("2019-01-01", periods=100)
    values = np.random.default_rng(7).normal(0, 0.01, (100, n_assets)).astype(np.float32)
    table = ReturnTable(dates, values, tuple(f"asset_{i}" for i in range(n_assets)))
    scaler = TrainStandardizer.fit(table, train_end=dates[49])
    expected = (values.astype(np.float64) - values[:50].astype(np.float64).mean(0)) / values[:50].astype(np.float64).std(0)
    np.testing.assert_array_equal(scaler.transform(table).returns, expected.astype(np.float32))
    save_standardizer(tmp_path / "scaler.npz", scaler)
    restored = load_standardizer(tmp_path / "scaler.npz")
    np.testing.assert_array_equal(restored.mean, scaler.mean)
    np.testing.assert_array_equal(restored.std, scaler.std)
    datasets = build_window_datasets(
        restored.transform(table), table, lookback=10, horizon=1,
        train_end=str(dates[49]), val_end=str(dates[74]), test_end=str(dates[-1]),
    )
    assert datasets.val[0]["history"].shape == (10, n_assets)
    np.testing.assert_array_equal(datasets.val.raw_windows.history[0], values[40:50])
    np.testing.assert_array_equal(datasets.val.raw_windows.target[0, 0], values[50])


def test_dataset_dispatch_preserves_kf12_source(tmp_path):
    import zipfile
    text = "Average Value Weighted Returns\n," + ",".join(INDUSTRY_COLUMNS)
    text += "\n20000103," + ",".join(["1.25"] * 12) + "\n\n"
    archive = tmp_path / "kf12.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("kf12.csv", text)
    table = load_return_table("ken_french_12", path=archive, download_if_missing=False)
    assert table.columns == INDUSTRY_COLUMNS
    assert table.returns.shape == (1, 12)
    np.testing.assert_array_equal(table.returns, np.full((1, 12), 0.0125, np.float32))
    with pytest.raises(ValueError, match="Unknown return dataset"):
        load_return_table("unimplemented")


def test_shared_preparation_uses_dataset_dispatch_and_saved_scaler_without_refitting(tmp_path, monkeypatch):
    from dataclasses import replace
    from diffusion_portfolio.config import load_config
    from diffusion_portfolio.data import RiskFreeSeries
    from diffusion_portfolio.data import preparation

    dates = pd.bdate_range("1999-11-01", periods=100)
    values = np.random.default_rng(9).normal(0, .01, (100, 17)).astype(np.float32)
    table = ReturnTable(dates, values, tuple(f"a{i}" for i in range(17)))
    cfg = load_config(Path(__file__).parents[2] / "configs/mvp.yaml")
    cfg = replace(cfg, data=replace(
        cfg.data, dataset="existing_loader", lookback=10,
        sample_start=str(dates[0].date()), train_end=str(dates[49].date()),
        val_end=str(dates[74].date()), sample_end=str(dates[-1].date()),
    ))
    requested = []
    def load(name):
        requested.append(name)
        return table
    monkeypatch.setattr(preparation, "load_return_table", load)
    monkeypatch.setattr(preparation, "load_daily_risk_free", lambda: RiskFreeSeries(dates, np.zeros(100, np.float32)))
    datasets, scaler = preparation.prepare_return_data(cfg)
    save_standardizer(tmp_path / "scaler.npz", scaler)
    monkeypatch.setattr(TrainStandardizer, "fit", lambda *a, **k: pytest.fail("Evaluation refitted preprocessing"))
    restored, _ = preparation.prepare_return_data(cfg, standardizer_path=tmp_path / "scaler.npz")
    assert requested == ["existing_loader", "existing_loader"]
    np.testing.assert_array_equal(restored.train.model_windows.history, datasets.train.model_windows.history)
    np.testing.assert_array_equal(restored.val.model_windows.target, datasets.val.model_windows.target)
