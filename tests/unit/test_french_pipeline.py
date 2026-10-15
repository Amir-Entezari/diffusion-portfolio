"""Exercise the existing preparation and model paths with 49 and 100 assets."""
from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import RiskFreeSeries, TrainStandardizer
from diffusion_portfolio.data import preparation
from diffusion_portfolio.data.diagnostics import run_data_check
from diffusion_portfolio.data.preparation import save_standardizer


@pytest.mark.parametrize("config_name, fixture, count", [
    ("cde_ff49.yaml", "ff49_daily.csv", 49),
    ("cde_ff100.yaml", "ff100_size_bm_daily.csv", 100),
])
def test_french_config_pipeline_saved_scaler_and_model_smoke(config_name, fixture, count, tmp_path, monkeypatch, capsys):
    root = Path(__file__).parents[2]
    cfg = load_config(root / "configs" / config_name)
    reference = load_config(root / "configs/cde.yaml")
    # No experimental changes other than source/preparation settings.
    assert cfg.model == reference.model
    assert cfg.training == reference.training
    assert cfg.seed == reference.seed
    for key in ("sample_start", "sample_end", "train_end", "val_end", "lookback", "horizon"):
        assert getattr(cfg.data, key) == getattr(reference.data, key)
    assert cfg.data.subtract_risk_free is True
    assert cfg.data.source_options == {"missing_policy": "drop"}

    dates = pd.bdate_range("2020-01-02", periods=40)
    fixture_text = (root / "tests/fixtures" / fixture).read_text()
    header = fixture_text.split("Average Value Weighted Returns -- Daily\n")[1].splitlines()[0]
    # Deterministic artificial percentage rows; no generator is integrated.
    values = np.random.default_rng(19).normal(0, 1, (40, count))
    text = "Average Value Weighted Returns -- Daily\n" + header + "\n"
    text += "\n".join(d.strftime("%Y%m%d") + "," + ",".join(map(str, row)) for d, row in zip(dates, values)) + "\n\n"
    archive_path = tmp_path / "daily.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("daily.csv", text)
    cfg = replace(cfg, data=replace(
        cfg.data, lookback=4, sample_start=str(dates[0].date()),
        sample_end=str(dates[-1].date()), train_end=str(dates[19].date()),
        val_end=str(dates[29].date()),
        source_options={"path": str(archive_path), "download_if_missing": False, "missing_policy": "drop"},
    ), model=replace(
        cfg.model, condition_dim=8, history_hidden_dim=16, channels=(8, 16),
        cde_hidden_dim=8, cde_drift_hidden_dim=16, cde_sensitivity_hidden_dim=16,
        time_embed_dim=8, diffusion_steps=5,
    ))
    rf = RiskFreeSeries(dates, np.full(40, .0001, np.float32))
    monkeypatch.setattr(preparation, "load_daily_risk_free", lambda: rf)
    datasets, scaler = preparation.prepare_return_data(cfg)
    raw = (values / 100).astype(np.float32) - rf.returns[:, None]
    np.testing.assert_array_equal(datasets.val.raw_windows.history[0], raw[16:20])
    np.testing.assert_array_equal(datasets.val.raw_windows.target[0, 0], raw[20])
    np.testing.assert_array_equal(scaler.mean, raw[:20].astype(np.float64).mean(0))
    np.testing.assert_array_equal(scaler.std, raw[:20].astype(np.float64).std(0))
    assert len(scaler.columns) == count

    # Existing CLI tooling performs a real CDE forward/loss/backward on train.
    raw_config = yaml.safe_load((root / "configs" / config_name).read_text())
    raw_config["data"] = asdict(cfg.data)
    raw_config["model"] = asdict(cfg.model)
    raw_config["model"]["channels"] = list(cfg.model.channels)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))
    run_data_check(Namespace(config=config_path, model_smoke=True, smoke_batch_size=2, device="cpu"))
    report = capsys.readouterr().out
    assert "Model forward/loss/backward smoke passed" in report
    assert "largest retained target-date gap:" in report

    scaler_path = tmp_path / "scaler.npz"
    save_standardizer(scaler_path, scaler)
    monkeypatch.setattr(TrainStandardizer, "fit", lambda *a, **k: pytest.fail("Saved scaler was refitted"))
    restored, _ = preparation.prepare_return_data(cfg, standardizer_path=scaler_path)
    np.testing.assert_array_equal(restored.val.model_windows.target, datasets.val.model_windows.target)

    # A saved KF12/other-panel scaler cannot silently normalize this dataset.
    with pytest.raises(ValueError, match="columns do not match"):
        preparation.prepare_return_data(cfg, scaler=TrainStandardizer.identity(("different_asset",)))
