"""End-to-end benchmark alignment, causality, serialization and scientific guards."""
from argparse import Namespace
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data.core import ReturnTable
from diffusion_portfolio.data.registry import load_return_table
from diffusion_portfolio.data import preparation
from diffusion_portfolio.evaluation import synthetic_benchmarks as benchmark
from diffusion_portfolio.evaluation.probabilistic import evaluate_probabilistic_forecast, require_finite_arithmetic_moments
from diffusion_portfolio.evaluation.scenarios import probabilistic_summary

ROOT = Path(__file__).parents[2]


@pytest.fixture
def small_market(tmp_path):
    cfg = load_config(ROOT / "configs/synthetic_gaussian.yaml")
    dates = pd.bdate_range("2000-01-03", periods=320)
    data = replace(cfg.data, sample_start=str(dates[0].date()), sample_end=str(dates[-1].date()),
                   train_end=str(dates[179].date()), val_end=str(dates[249].date()), lookback=6,
                   source_options={"n_assets": 5, "n_observations": 320, "n_factors": 2,
                                   "seed": 19, "start_date": str(dates[0].date())})
    cfg = replace(cfg, data=data)
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg)))
    table = load_return_table(data.dataset, **data.source_options)
    return cfg, path, table


def archives(output):
    result = {}
    for name in benchmark.MODEL_STREAMS:
        with np.load(output / name / "validation_scenarios.npz", allow_pickle=False) as data:
            result[name] = {key: data[key].copy() for key in data.files}
    return result


def test_benchmark_reuses_metrics_alignment_unique_training_rows_and_existing_archives(small_market, tmp_path, monkeypatch):
    cfg, path, table = small_market
    original_prepare = benchmark.prepare_return_data

    def validation_only(cfg):
        datasets, scaler = original_prepare(cfg)
        return SimpleNamespace(val=datasets.val), scaler  # no test partition exposed

    monkeypatch.setattr(benchmark, "prepare_return_data", validation_only)
    output = tmp_path / "benchmark"
    report = benchmark.benchmark_synthetic_forecasts(path, output_dir=output, n_scenarios=16, limit=8)
    assert report["training"]["n_observations"] == 180  # not 174*lookback duplicate rows
    assert report["n_dates"] == 8 and report["available_validation_dates"] == 70
    assert report["return_units"] == "arithmetic decimal asset returns"
    fitted_x = np.log1p(table.returns[:180])
    np.testing.assert_allclose(report["models"]["gaussian"]["parameters"]["mean_log"], fitted_x.mean(0))
    np.testing.assert_allclose(report["models"]["gaussian"]["parameters"]["covariance_log"], np.cov(fitted_x, rowvar=False, ddof=0))
    saved = archives(output)
    reference = saved["gaussian"]
    expected_observed = table.returns[180:188].astype(np.float32)
    for name, data in saved.items():
        assert set(data) == {"scenarios", "observed", "dates", "columns"}
        assert data["scenarios"].shape == (8, 16, 5)
        assert np.isfinite(data["scenarios"]).all() and (data["scenarios"] > -1).all()
        np.testing.assert_array_equal(data["observed"], expected_observed)
        np.testing.assert_array_equal(data["dates"], table.dates[180:188].values.astype("datetime64[D]"))
        np.testing.assert_array_equal(data["columns"], table.columns)
        np.testing.assert_array_equal(data["observed"], reference["observed"])
        metric = evaluate_probabilistic_forecast(data["scenarios"], data["observed"])
        expected = probabilistic_summary(metric, table.columns)
        for key, value in expected.items():
            assert report["models"][name][key] == value
        assert report["models"][name]["mean_abs_ace"] == np.abs(metric.calibration.ace).mean()
    assert json.loads((output / "report.json").read_text()) == report
    assert load_config(output / "config.yaml") == cfg
    assert (output / "standardizer.npz").exists()
    assert set(pd.read_csv(output / "comparison.csv").model) == set(saved)


def test_subset_model_order_and_later_subset_are_reproducible(small_market, tmp_path):
    cfg, path, table = small_market
    short, full = tmp_path / "short", tmp_path / "full"
    benchmark.benchmark_synthetic_forecasts(path, output_dir=short, n_scenarios=12, limit=4)
    benchmark.benchmark_synthetic_forecasts(path, output_dir=full, n_scenarios=12, models=("ewma", "ar1", "gaussian"))
    a, b = archives(short), archives(full)
    for name in a:
        np.testing.assert_array_equal(a[name]["scenarios"], b[name]["scenarios"][:4])
    # Starting at a later target must still replay every admissible prior
    # validation observation for EWMA, including unevaluated dates.
    from diffusion_portfolio.baselines.statistical import EWMAGaussian
    from diffusion_portfolio.data.windows import WindowedReturns
    dataset, _ = preparation.prepare_return_data(cfg)
    windows = dataset.val.raw_windows
    subset = WindowedReturns(windows.history[7:12], windows.target[7:12], windows.target_dates[7:12])
    model = EWMAGaussian.fit(table.returns[:180])
    forecasts = benchmark.forecast_baseline("ewma", model, table, subset, train_end=cfg.data.train_end, seed=42, n_scenarios=12)
    np.testing.assert_array_equal(forecasts, b["ewma"]["scenarios"][7:12])


def test_adversarial_validation_and_test_changes_cannot_leak_into_past_forecasts(small_market, tmp_path, monkeypatch):
    cfg, path, table = small_market
    original = tmp_path / "original"
    report = benchmark.benchmark_synthetic_forecasts(path, output_dir=original, limit=12, n_scenarios=16)
    changed_returns = table.returns.copy()
    changed_returns[184] += .05  # fifth validation target, unknown at its forecast origin
    changed_returns[250:] += 2  # the entire held-out test period
    altered = ReturnTable(table.dates, changed_returns, table.columns)
    monkeypatch.setattr(benchmark, "load_return_table", lambda *a, **k: altered)
    monkeypatch.setattr(preparation, "load_return_table", lambda *a, **k: altered)
    changed = tmp_path / "changed"
    other = benchmark.benchmark_synthetic_forecasts(path, output_dir=changed, limit=12, n_scenarios=16)
    a, b = archives(original), archives(changed)
    for name in a:
        assert report["models"][name]["parameters"] == other["models"][name]["parameters"]
        np.testing.assert_array_equal(a[name]["scenarios"][:5], b[name]["scenarios"][:5])
    np.testing.assert_array_equal(a["gaussian"]["scenarios"], b["gaussian"]["scenarios"])
    for name in ("ar1", "ewma"):
        assert not np.array_equal(a[name]["scenarios"][5], b[name]["scenarios"][5])
    # AR uses the last observation only; EWMA retains a decaying causal effect.
    np.testing.assert_array_equal(a["ar1"]["scenarios"][6:], b["ar1"]["scenarios"][6:])
    assert not np.array_equal(a["ewma"]["scenarios"][6:], b["ewma"]["scenarios"][6:])
    with np.load(original / "standardizer.npz") as first, np.load(changed / "standardizer.npz") as second:
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])


def test_test_only_changes_leave_all_validation_results_unchanged(small_market, tmp_path, monkeypatch):
    _, path, table = small_market
    first = tmp_path / "first"
    report = benchmark.benchmark_synthetic_forecasts(path, output_dir=first, limit=4, n_scenarios=8)
    altered_returns = table.returns.copy()
    altered_returns[250:] += 10
    altered = ReturnTable(table.dates, altered_returns, table.columns)
    monkeypatch.setattr(benchmark, "load_return_table", lambda *a, **k: altered)
    monkeypatch.setattr(preparation, "load_return_table", lambda *a, **k: altered)
    second = tmp_path / "second"
    other = benchmark.benchmark_synthetic_forecasts(path, output_dir=second, limit=4, n_scenarios=8)
    assert report["source_identity"] == other["source_identity"]
    for name, data in archives(first).items():
        for key, value in data.items():
            np.testing.assert_array_equal(value, archives(second)[name][key])
        for key in ("parameters", "crps_mean", "energy_score", "calibration"):
            assert report["models"][name][key] == other["models"][name][key]


@pytest.mark.parametrize("dataset", ["synthetic_regime", "synthetic_regime_jump", "synthetic_whipsaw"])
def test_other_finite_moment_synthetic_scenarios(small_market, tmp_path, dataset):
    cfg, _, _ = small_market
    cfg = replace(cfg, data=replace(cfg.data, dataset=dataset))
    path = tmp_path / f"{dataset}.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg)))
    report = benchmark.benchmark_synthetic_forecasts(path, output_dir=tmp_path / dataset, limit=3, n_scenarios=4)
    assert report["dataset"] == dataset and report["n_dates"] == 3


@pytest.mark.parametrize("kwargs,match", [
    ({"split": "test"}, "validation only"), ({"limit": 0}, "limit"), ({"limit": True}, "limit"),
    ({"n_scenarios": 1}, "n_scenarios"), ({"seed": -1}, "seed"), ({"models": ()}, "models"),
    ({"models": ("oracle",)}, "models"), ({"models": ("gaussian", "gaussian")}, "models"),
    ({"ewma_decay": 1}, "decay"),
])
def test_invalid_settings_fail_before_output(small_market, tmp_path, kwargs, match):
    _, path, _ = small_market
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        benchmark.benchmark_synthetic_forecasts(path, output_dir=output, **kwargs)
    assert not output.exists()


def test_output_isolation_and_source_alignment(small_market, tmp_path, monkeypatch):
    cfg, path, table = small_market
    output = tmp_path / "existing"
    output.mkdir()
    (output / "best.pt").write_bytes(b"historical checkpoint")
    with pytest.raises(FileExistsError, match="not empty"):
        benchmark.benchmark_synthetic_forecasts(path, output_dir=output)
    assert (output / "best.pt").read_bytes() == b"historical checkpoint"
    permuted = ReturnTable(table.dates, table.returns[:, ::-1], tuple(reversed(table.columns)))
    monkeypatch.setattr(benchmark, "load_return_table", lambda *a, **k: permuted)
    with pytest.raises(ValueError, match="asset order"):
        benchmark.benchmark_synthetic_forecasts(path, output_dir=tmp_path / "bad-order")


def test_student_guard_at_benchmark_diffusion_and_portfolio_entrypoints(tmp_path):
    from diffusion_portfolio.evaluation.forecasts import evaluate_diffusion
    from diffusion_portfolio.evaluation.portfolio_experiment import run_portfolio_evaluation

    cfg = load_config(ROOT / "configs/synthetic_student_t.yaml")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg)))
    for operation in (
        lambda: benchmark.benchmark_synthetic_forecasts(path, output_dir=tmp_path / "benchmark"),
        lambda: evaluate_diffusion(tmp_path, device="cpu"),
        lambda: run_portfolio_evaluation(Namespace(run_dir=tmp_path)),
    ):
        with pytest.raises(ValueError, match="no finite positive arithmetic-return population moments"):
            operation()  # Fails before reading scaler/checkpoint/scenarios or downloading any data.
    assert not (tmp_path / "benchmark").exists()
    for name in benchmark.SUPPORTED_DATASETS | {"ken_french_12", "ken_french_49", "ken_french_100_size_bm"}:
        require_finite_arithmetic_moments(name)


def test_cli_accepts_actual_supported_arguments(monkeypatch):
    import runpy
    import sys
    monkeypatch.setattr(sys, "argv", ["benchmark_synthetic_forecasts.py", "--config", "configs/synthetic_gaussian.yaml",
                                    "--output-dir", "outputs/example", "--models", "gaussian,ar1,ewma",
                                    "--split", "validation", "--n-scenarios", "16", "--limit", "8", "--seed", "43"])
    args = runpy.run_path(str(ROOT / "scripts/benchmark_synthetic_forecasts.py"))["parse_args"]()
    assert args.limit == 8 and args.n_scenarios == 16 and args.seed == 43 and args.ewma_decay == .94


def test_notebook_schema_sync_syntax_and_real_cli_arguments(monkeypatch):
    import ast
    import runpy
    import shlex
    import sys

    nbformat = pytest.importorskip("nbformat")
    transform = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager()
    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_forecast_benchmarks.ipynb", as_version=4)
    reference = nbformat.read(ROOT / "notebooks/synthetic/synthetic_market_stage4c_validation.ipynb", as_version=4)
    nbformat.validate(book)
    cells = {cell.id: cell for cell in book.cells}
    assert cells["github-sync"].source == next(cell.source for cell in reference.cells if cell.id == "github-sync")
    assert 'BRANCH = "feature/synthetic-market"' in cells["github-sync"].source
    seen, outputs = set(), set()
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None and cell.outputs == []
        assert not any(word in cell.source for word in ("subprocess", "%%bash", "MODE", "train_diffusion.py"))
        source = transform.transform_cell(cell.source)
        compile(source, cell.id, "exec")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "system":
                continue
            command = ast.literal_eval(node.args[0])
            if "scripts/benchmark_synthetic_forecasts.py" not in command:
                continue  # Never execute Git or dependency-installation cells.
            argv = shlex.split(command)
            index = argv.index("scripts/benchmark_synthetic_forecasts.py")
            monkeypatch.setattr(sys, "argv", argv[index:])
            args = runpy.run_path(str(ROOT / argv[index]))["parse_args"]()
            cfg = load_config(ROOT / args.config)
            assert args.split == "validation" and args.limit == 64 and args.n_scenarios == 128
            assert cell.source.splitlines()[-1].startswith("assert _exit_code == 0")
            seen.add(cfg.data.dataset)
            outputs.add(args.output_dir)
    assert seen == {"synthetic_gaussian", "synthetic_mean_reverting"}
    assert len(outputs) == 2
