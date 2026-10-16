"""Offline integration of controlled markets with the unchanged financial pipeline."""
from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import preparation, TrainStandardizer, make_dataloaders
from diffusion_portfolio.data.diagnostics import run_data_check
from diffusion_portfolio.data.preparation import save_standardizer
from diffusion_portfolio.data.registry import load_return_table
from diffusion_portfolio.data.sources.synthetic_market import simulate_market
from diffusion_portfolio.evaluation.synthetic_market import market_diagnostics, plot_market_diagnostics
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.training.diffusion import train_one_epoch

ROOT = Path(__file__).parents[2]
SCENARIOS = ("synthetic_gaussian", "synthetic_regime", "synthetic_regime_jump")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_config_source_splits_saved_scaler_and_hidden_state_isolation(scenario, tmp_path, monkeypatch):
    cfg = load_config(ROOT / f"configs/{scenario}.yaml")
    assert cfg.model == load_config(ROOT / "configs/mvp.yaml").model
    assert cfg.training == load_config(ROOT / "configs/mvp.yaml").training
    assert not cfg.data.subtract_risk_free
    monkeypatch.setattr(preparation, "load_daily_risk_free", lambda: pytest.fail("Synthetic data must not load French RF"))
    table = load_return_table(scenario, **cfg.data.source_options)
    market = simulate_market(scenario, **cfg.data.source_options)
    assert set(vars(table)) == {"dates", "returns", "columns"}
    np.testing.assert_array_equal(table.returns, market.returns)
    assert table.columns == market.columns and table.dates.equals(market.dates)
    assert table.returns.shape == (8000, 12)
    assert pd.Timestamp(cfg.data.train_end) == table.dates[4999]
    assert pd.Timestamp(cfg.data.val_end) == table.dates[6499]
    assert pd.Timestamp(cfg.data.sample_end) == table.dates[-1]
    datasets, scaler = preparation.prepare_return_data(cfg)
    assert [len(datasets.train), len(datasets.val), len(datasets.test)] == [4940, 1500, 1500]
    np.testing.assert_array_equal(scaler.mean, table.returns[:5000].mean(0))
    np.testing.assert_array_equal(scaler.std, table.returns[:5000].std(0))
    sample = datasets.val[0]
    assert set(sample) == {"history", "target", "history_raw", "target_raw", "target_date"}
    assert sample["history"].shape == (60, 12) and sample["target"].shape == (1, 12)
    np.testing.assert_array_equal(sample["history_raw"], table.returns[4940:5000].astype(np.float32))
    np.testing.assert_array_equal(sample["target_raw"], table.returns[5000:5001].astype(np.float32))
    assert datasets.train.raw_windows.target_dates[-1] <= pd.Timestamp(cfg.data.train_end) < datasets.val.raw_windows.target_dates[0]
    assert datasets.val.raw_windows.target_dates[-1] <= pd.Timestamp(cfg.data.val_end) < datasets.test.raw_windows.target_dates[0]
    save_standardizer(tmp_path / "scaler.npz", scaler)
    monkeypatch.setattr(TrainStandardizer, "fit", lambda *a, **k: pytest.fail("Refitted a saved scaler"))
    repeated, _ = preparation.prepare_return_data(replace(cfg, seed=43), standardizer_path=tmp_path / "scaler.npz")
    np.testing.assert_array_equal(repeated.val.model_windows.history, datasets.val.model_windows.history)
    np.testing.assert_array_equal(repeated.train.model_windows.target, datasets.train.model_windows.target)
    # Hidden labels can be altered for diagnostics, but never enter the source panel.
    market.ground_truth["regimes"][:] = 99
    np.testing.assert_array_equal(load_return_table(scenario, **cfg.data.source_options).returns, table.returns)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_generic_small_cpu_training_and_data_smoke(scenario, tmp_path, monkeypatch):
    import yaml
    from dataclasses import asdict

    cfg = load_config(ROOT / f"configs/{scenario}.yaml")
    dates = pd.bdate_range("2000-01-03", periods=120)
    cfg = replace(cfg, data=replace(cfg.data, lookback=10,
        sample_start=str(dates[0].date()), train_end=str(dates[59].date()),
        val_end=str(dates[89].date()), sample_end=str(dates[-1].date()),
        source_options={"n_assets": 5, "n_observations": 120, "n_factors": 2,
                        "seed": 42, "start_date": str(dates[0].date())}),
        model=replace(cfg.model, condition_dim=8, history_hidden_dim=16,
                      channels=(8,), time_embed_dim=8, diffusion_steps=4),
        training=replace(cfg.training, batch_size=25))
    monkeypatch.setattr(preparation, "load_daily_risk_free", lambda: pytest.fail("No RF for synthetic sources"))
    datasets, scaler = preparation.prepare_return_data(cfg)
    loaders = make_dataloaders(datasets, batch_size=25, seed=cfg.seed)
    torch.manual_seed(42)
    model = build_diffusion(cfg, n_assets=len(scaler.columns))
    before = [p.detach().clone() for p in model.parameters()]
    loss = train_one_epoch(model, loaders.train, torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate),
                           device=torch.device("cpu"), gradient_clip_norm=1.)
    assert np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg)))
    report = run_data_check(Namespace(config=path, model_smoke=True, smoke_batch_size=2, device="cpu"))
    assert report["n_assets"] == 5
    assert report["model_smoke"]["gradients_finite"]
    assert report["model_smoke"]["prediction_shape"] == [2, 5]


def test_future_changes_cannot_refit_normalization_or_change_training(monkeypatch):
    cfg = load_config(ROOT / "configs/synthetic_regime_jump.yaml")
    table = load_return_table(cfg.data.dataset, **cfg.data.source_options)
    monkeypatch.setattr(preparation, "load_return_table", lambda *a, **k: table)
    before, scaler = preparation.prepare_return_data(cfg)
    table.returns[5000:] += 10
    after, updated = preparation.prepare_return_data(cfg)
    np.testing.assert_array_equal(updated.mean, scaler.mean)
    np.testing.assert_array_equal(updated.std, scaler.std)
    np.testing.assert_array_equal(after.train.model_windows.history, before.train.model_windows.history)
    np.testing.assert_array_equal(after.train.model_windows.target, before.train.model_windows.target)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_diagnostics_are_cutoff_limited_and_plot_small_prefix(scenario):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    result = simulate_market(scenario, n_assets=3, n_observations=400, seed=13)
    end = result.dates[299]
    report = market_diagnostics(result, end_date=end)
    assert report["observations"] == 300 and report["end"] == str(end.date())
    assert sum(row["observations"] for row in report["summary"]) == 300
    assert np.asarray(report["transition_counts"]).sum() == 299
    json.dumps(report, allow_nan=False)
    result.log_returns[300:] = np.nan
    result.returns[300:] = np.nan
    result.ground_truth["regimes"][300:] = 99
    assert market_diagnostics(result, end_date=end) == report
    fig = plot_market_diagnostics(result, end_date=end)
    assert len(fig.axes) >= 6
    plt.close(fig)


def test_synthetic_demo_notebook_format_and_syntax():
    nbformat = pytest.importorskip("nbformat")
    transformer = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager()
    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_market_validation.ipynb", as_version=4)
    nbformat.validate(book)
    cells = {c.id: c for c in book.cells}
    reference = nbformat.read(ROOT / "notebooks/high_dimensional/ff49_vanilla.ipynb", as_version=4)
    assert cells["github-sync"].source == next(c.source for c in reference.cells if c.id == "github-sync")
    for cell in book.cells:
        if cell.cell_type == "code":
            compile(transformer.transform_cell(cell.source), cell.id, "exec")
            assert not any(term in cell.source for term in ("MODE", "subprocess", "%%bash", "%%capture"))
            assert cell.outputs == [] and cell.execution_count is None
            if "!PYTHONUNBUFFERED=1" in cell.source:
                assert cell.source.splitlines()[-1].startswith("assert _exit_code == 0")
    assert "end_date=cfg.data.train_end" in cells["diagnostic-reports"].source
    assert "--resume" in cells["optional-pilot"].source and "--stop-after-epoch 2" in cells["optional-pilot"].source


def test_diagnostics_handle_unvisited_regime_and_undefined_correlation():
    result = simulate_market("synthetic_regime", n_assets=2, n_factors=1, n_observations=50,
                             initial_regime=0, transition_matrix=[[1, 0], [0, 1]],
                             factor_loadings=0, idiosyncratic_vol=0)
    report = market_diagnostics(result, end_date=result.dates[-1])
    assert report["summary"][1]["observations"] == 0
    assert report["summary"][0]["expected_duration_days"] is None
    assert report["moments"]["normal"]["configured_correlation"] == [[None, None], [None, None]]
    json.dumps(report, allow_nan=False)
