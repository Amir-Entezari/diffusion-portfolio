"""Offline research-workflow checks using tiny artificial panels, never real runs."""
import ast
from argparse import Namespace
from dataclasses import asdict, replace
import json
from pathlib import Path
import runpy
import shlex
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import ReturnTable, TrainStandardizer
from diffusion_portfolio.data.dataset import build_window_datasets
from diffusion_portfolio.data.preparation import save_standardizer
from diffusion_portfolio.data import precompute
from diffusion_portfolio.models.diffusion import PrecomputedConditionDiffusion
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.training import multiseed, probes, routing
from diffusion_portfolio.training import runs
from diffusion_portfolio.utils.research_checks import (
    check_checkpoint_pair, require_completed_run, source_identity,
)

ROOT = Path(__file__).parents[2]
NEW_NOTEBOOKS = [ROOT / f"notebooks/high_dimensional/ff{n}_{name}.ipynb"
                 for n in (49, 100) for name in ("multiseed", "evidential_router", "tda", "spd")]


def saved_run(folder, config, assets):
    """Metadata fixtures only; these are not trained model outputs."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config, folder / "config.yaml")
    cfg = load_config(config)
    save_standardizer(folder / "standardizer.npz", TrainStandardizer.identity(tuple(f"a{i}" for i in range(assets))))
    torch.save({"epoch": 1, "val_loss": .2, "model_state_dict": {}}, folder / "best.pt")
    torch.save({"epoch": cfg.training.epochs, "best_epoch": 1, "best_val_loss": .2}, folder / "last.pt")
    (folder / "summary.json").write_text(json.dumps({"best_epoch": 1, "best_val_loss": .2}))
    (folder / "validation_probabilistic_metrics.json").write_text(json.dumps({
        "split": "validation", "checkpoint_epoch": 1, "checkpoint_val_loss": .2,
        "n_scenarios": cfg.evaluation.n_scenarios, "sampling_seed": cfg.training.validation_seed,
        "crps_mean": .01, "energy_score": .02, "calibration": [{"ace": -.1}],
    }))


@pytest.mark.parametrize("assets", [49, 100])
def test_matching_multiseed_configs_resume_and_seed42_reuse(assets, tmp_path, monkeypatch):
    vanilla, cde = tmp_path / "vanilla42", tmp_path / "cde42"
    vc, cc = ROOT / f"configs/vanilla_ff{assets}.yaml", ROOT / f"configs/cde_ff{assets}.yaml"
    saved_run(vanilla, vc, assets)
    saved_run(cde, cc, assets)
    calls = []

    def command(argv, *, cwd):
        calls.append(argv)
        if "scripts/train_diffusion.py" in argv:
            assert "--resume" in argv and "-u" in argv
            config = Path(argv[argv.index("--config") + 1])
            cfg = load_config(config)
            assert cfg.seed in (43, 44) and cfg.data == load_config(cc).data
            run = Path(argv[argv.index("--output-dir") + 1])
            assert run not in (vanilla, cde)
            saved_run(run, config, assets)
            (run / "validation_probabilistic_metrics.json").unlink()
        else:
            assert argv[argv.index("--split") + 1] == "validation"
            run = Path(argv[argv.index("--run-dir") + 1])
            metrics = json.loads((vanilla / "validation_probabilistic_metrics.json").read_text())
            (run / "validation_probabilistic_metrics.json").write_text(json.dumps(metrics))

    monkeypatch.setattr(multiseed, "run_command", command)
    args = Namespace(seed42_vanilla_dir=str(vanilla), seed42_cde_dir=str(cde),
                     vanilla_config=str(vc), cde_config=str(cc), output_root=str(tmp_path / "paired"),
                     seeds=[43, 44], device="cpu")
    frozen = (cde / "best.pt").read_bytes()
    multiseed.run_multiseed(args, repo_root=ROOT)
    assert len(calls) == 8
    result = pd.read_csv(tmp_path / "paired/per_seed_results.csv")
    assert len(result) == 6 and set(result.seed) == {42, 43, 44}
    multiseed.run_multiseed(args, repo_root=ROOT)
    assert len(calls) == 8 and (cde / "best.pt").read_bytes() == frozen
    with pytest.raises(ValueError, match="KF12 defaults are unsafe"):
        multiseed.run_multiseed(Namespace(**{**vars(args), "vanilla_config": None, "cde_config": None}), repo_root=ROOT)
    bad = yaml.safe_load((cde / "config.yaml").read_text())
    bad["data"]["dataset"] = "ken_french_12"
    (cde / "config.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(RuntimeError, match="configuration"):
        multiseed.run_multiseed(args, repo_root=ROOT)
    assert len(calls) == 8


def test_completed_reference_rejects_missing_partial_or_stale_validation(tmp_path):
    run = tmp_path / "reference"
    with pytest.raises(FileNotFoundError, match="prerequisite"):
        require_completed_run(run)
    saved_run(run, ROOT / "configs/cde_ff49.yaml", 49)
    assert require_completed_run(run, assets=49, validation=True).data.dataset == "ken_french_49"
    with pytest.raises(RuntimeError, match="universe"):
        require_completed_run(run, assets=100)
    last = torch.load(run / "last.pt", weights_only=False)
    torch.save({**last, "epoch": 2}, run / "last.pt")
    with pytest.raises(RuntimeError, match="incomplete"):
        require_completed_run(run)
    torch.save(last, run / "last.pt")
    metrics_path = run / "validation_probabilistic_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics_path.write_text(json.dumps({**metrics, "split": "test"}))
    with pytest.raises(RuntimeError, match="Validation metrics"):
        require_completed_run(run, validation=True)
    (run / "last.pt").unlink()
    with pytest.raises(RuntimeError, match="Incomplete checkpoint pair"):
        check_checkpoint_pair(run)


def test_partial_feature_run_cannot_change_variant_or_source(tmp_path, monkeypatch):
    source, features, output = tmp_path / "cde", tmp_path / "features", tmp_path / "ablation"
    saved_run(source, ROOT / "configs/cde_ff49.yaml", 49)
    features.mkdir()
    (features / "metadata.json").write_text('{"fixture": "feature cache"}')
    prepare = Mock(side_effect=RuntimeError("stop before expensive work"))
    monkeypatch.setattr(runs, "prepare_return_data", prepare)
    with pytest.raises(RuntimeError, match="stop before expensive"):
        runs.train_feature_ablation("covariance", source, features, output, device="cpu", resume=True)
    before = (output / "experiment.json").read_bytes()
    with pytest.raises(RuntimeError, match="another source"):
        runs.train_feature_ablation("log_spd", source, features, output, device="cpu", resume=True)
    assert (output / "experiment.json").read_bytes() == before
    assert prepare.call_count == 1


@pytest.mark.parametrize("assets", [49, 100])
@pytest.mark.parametrize("kind", ["topology", "spd"])
def test_feature_smoke_cache_train_scaling_dates_and_downstream_dimensions(assets, kind, tmp_path, monkeypatch):
    if kind == "topology":
        pytest.importorskip("gudhi")
    cfg = load_config(ROOT / f"configs/cde_ff{assets}.yaml")
    dates = pd.bdate_range("1998-01-02", periods=70)
    cfg = replace(cfg, data=replace(cfg.data, sample_start=str(dates[0].date()), sample_end=str(dates[-1].date()),
                  train_end=str(dates[64].date()), val_end=str(dates[68].date())),
                  model=replace(cfg.model, condition_dim=8, history_hidden_dim=8, channels=(8, 16),
                                time_embed_dim=8, diffusion_steps=4), training=replace(cfg.training, epochs=2))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)))
    source = tmp_path / "source"
    saved_run(source, config, assets)
    table = ReturnTable(dates, np.random.default_rng(21).normal(0, .01, (70, assets)).astype(np.float32),
                        tuple(f"a{i}" for i in range(assets)))
    scaler = TrainStandardizer.fit(table, train_end=dates[64])
    datasets = build_window_datasets(scaler.transform(table), table, lookback=60, horizon=1,
                                     train_end=dates[64], val_end=dates[68], test_end=dates[-1], max_gap_days=7)

    class TinyFrozenEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(assets, 8)

        def forward(self, history):
            assert not self.training and not any(p.requires_grad for p in self.parameters())
            return self.projection(history[:, -1])

        def encode_trajectory(self, history):
            return self.projection(history)

    encoder = TinyFrozenEncoder()
    model = SimpleNamespace(history_encoder=encoder, load_state_dict=Mock())
    monkeypatch.setattr(precompute, "build_diffusion", lambda *a, **k: model)
    monkeypatch.setattr(precompute, "prepare_return_data", lambda *a, **k: (datasets, scaler))
    cache = tmp_path / "features"
    report = precompute.precompute_features(kind, source, cache, device="cpu", smoke=True, batch_size=2)
    assert report["assets"] == assets and not cache.exists()
    metadata = precompute.precompute_features(kind, source, cache, device="cpu", batch_size=2, reuse=True)
    precompute.validate_feature_cache(cache, datasets, kind)
    variant = "log_spd" if kind == "spd" else "tda"
    if kind == "spd":
        assert metadata["log_spd_dim"] == assets * (assets + 1) // 2
    with np.load(cache / "train.npz") as saved:
        features = np.concatenate((saved["cde_condition"], saved[variant]), axis=1)
        extra = saved[variant].shape[1]
    diffusion = PrecomputedConditionDiffusion(build_diffusion(cfg, n_assets=assets, precomputed=True), extra_dim=extra)
    output = diffusion.training_loss(torch.from_numpy(features[:2]), torch.from_numpy(datasets.train.model_windows.target[:2]))
    output.loss.backward()
    assert output.prediction.shape == (2, assets) and torch.isfinite(output.loss)
    assert all(torch.isfinite(p.grad).all() for p in diffusion.parameters() if p.grad is not None)
    monkeypatch.setattr(precompute, "extract_spd_split" if kind == "spd" else "extract_topology_split",
                        Mock(side_effect=AssertionError("Feature extraction was repeated")))
    assert precompute.precompute_features(kind, source, cache, device="cpu", reuse=True) == metadata
    metadata_path = cache / "metadata.json"
    original_metadata = metadata_path.read_text()
    metadata_path.write_text(json.dumps({**metadata, "source": {"wrong": "checkpoint"}}))
    with pytest.raises(RuntimeError, match="another source"):
        precompute.precompute_features(kind, source, cache, device="cpu", reuse=True)
    metadata_path.write_text(original_metadata)
    original_target = datasets.val.raw_windows.target[0, 0, 0].copy()
    datasets.val.raw_windows.target[0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="another source"):
        precompute.precompute_features(kind, source, cache, device="cpu", reuse=True)
    datasets.val.raw_windows.target[0, 0, 0] = original_target
    with np.load(cache / "validation.npz") as saved:
        changed = dict(saved)
    changed["dates"] = changed["dates"][::-1]
    np.savez(cache / "validation.npz", **changed)
    with pytest.raises(RuntimeError, match="dates"):
        precompute.precompute_features(kind, source, cache, device="cpu", reuse=True)


def test_router_rejects_wrong_probe_before_encoding(tmp_path, monkeypatch):
    probe = tmp_path / "probe"
    probe.mkdir()
    for filename in ("evidential_best.pt", "softmax_best.pt"):
        (probe / filename).write_bytes(b"metadata guard runs before loading weights")
    (probe / "regime_thresholds.json").write_text('{"regime_target": "next_day"}')
    (probe / "validation_metrics.json").write_text('{}')
    args = Namespace(cde_run_dir=str(tmp_path / "missing_cde"), probe_run_dir=str(probe),
                     config="configs/router.yaml", output_dir=str(tmp_path / "router"), device="cpu")
    monkeypatch.setattr(routing, "extract_conditions", Mock(side_effect=AssertionError("Expensive extraction")))
    with pytest.raises(ValueError, match="current-regime"):
        routing.run_router(args)
    assert not Path(args.output_dir).exists()
    (probe / "regime_thresholds.json").write_text('{"regime_target": "current"}')
    (probe / "validation_metrics.json").write_text('{"source": {"wrong": "backbone"}}')
    saved_run(Path(args.cde_run_dir), ROOT / "configs/cde_ff49.yaml", 49)
    with pytest.raises(ValueError, match="another CDE"):
        routing.run_router(args)
    routing.extract_conditions.assert_not_called()


@pytest.mark.parametrize("assets", [49, 100])
@pytest.mark.parametrize("target", ["next_day", "current"])
def test_probe_pipeline_frozen_dimensions_and_train_only_thresholds(assets, target, tmp_path, monkeypatch):
    cfg = load_config(ROOT / f"configs/cde_ff{assets}.yaml")
    dates = pd.bdate_range("2020-01-02", periods=40)
    cfg = replace(cfg, data=replace(cfg.data, sample_start=str(dates[0].date()), sample_end=str(dates[-1].date()),
                  train_end=str(dates[25].date()), val_end=str(dates[35].date()), lookback=20),
                  model=replace(cfg.model, condition_dim=8), training=replace(cfg.training, epochs=2))
    config = tmp_path / "cde_config.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)))
    source = tmp_path / "cde"
    saved_run(source, config, assets)
    values = np.random.default_rng(14).normal(0, .01, (40, assets)).astype(np.float32)
    values[26:] *= 10  # Validation stress must not move training thresholds.
    table = ReturnTable(dates, values, tuple(f"a{i}" for i in range(assets)))
    scaler = TrainStandardizer.fit(table, train_end=dates[25])
    datasets = build_window_datasets(scaler.transform(table), table, lookback=20, horizon=1,
                                     train_end=dates[25], val_end=dates[35], test_end=dates[-1])
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(20 * assets, 8))
    model = SimpleNamespace(history_encoder=encoder, load_state_dict=Mock())
    monkeypatch.setattr(probes, "build_diffusion", lambda *a, **k: model)
    monkeypatch.setattr(probes, "prepare_return_data", lambda config: (datasets, scaler))
    probe_config = yaml.safe_load((ROOT / "configs/evidential.yaml").read_text())
    probe_config["training"]["epochs"] = 1
    probe_config["model"]["input_dim"] = 8
    probe_path = tmp_path / "probe.yaml"
    probe_path.write_text(yaml.safe_dump(probe_config))
    output = tmp_path / target
    frozen = source_identity(source)
    probes.run_probe(Namespace(cde_run_dir=source, config=probe_path, output_dir=output,
                               regime_target=target, device="cpu"))
    assert source_identity(source) == frozen and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    metadata = json.loads((output / "validation_metrics.json").read_text())
    assert metadata["source"] == frozen and metadata["dataset"] == cfg.data.dataset
    windows = datasets.train.raw_windows
    scores = (probes.cross_sectional_rms(windows.target[:, 0]) if target == "next_day"
              else probes.trailing_cross_sectional_rms(windows.history, recent_days=20))
    expected = np.quantile(scores, [.333333, .666667])
    actual = json.loads((output / "regime_thresholds.json").read_text())
    assert actual["regime_target"] == target
    np.testing.assert_allclose(list(actual["thresholds"].values()), expected)
    with np.load(output / "validation_predictions.npz") as saved:
        assert np.isfinite(saved["evidential_probabilities"]).all()
        assert saved["evidential_probabilities"].shape == (len(datasets.val), 3)
        np.testing.assert_array_equal(saved["dates"], datasets.val.raw_windows.target_dates.values)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    book = json.loads((ROOT / f"notebooks/high_dimensional/ff{assets}_evidential_router.ipynb").read_text())
    report = next(cell for cell in book["cells"] if cell["id"] == f"probe-{target}-report")
    source = report["source"] if isinstance(report["source"], str) else "".join(report["source"])
    exec(source, {"ROOT": tmp_path, "SOURCE": frozen, "json": json, "pd": pd, "np": np,
                         "display": Mock(), "validation_dates": datasets.val.raw_windows.target_dates.values})
    plt.close("all")


@pytest.mark.parametrize("assets", [49, 100])
def test_router_loss_supports_highdim_targets_with_frozen_denoiser(assets):
    from diffusion_portfolio.models.uncertainty.routing import SoftmaxResidualRouter, EvidentialResidualRouter

    cfg = load_config(ROOT / f"configs/cde_ff{assets}.yaml")
    cfg = replace(cfg, model=replace(cfg.model, condition_dim=8, history_hidden_dim=8,
                                    channels=(8, 16), time_embed_dim=8, diffusion_steps=4))
    model = build_diffusion(cfg, n_assets=assets, precomputed=True)
    routing.freeze_module(model)
    condition, target, noise = torch.randn(2, 8), torch.randn(2, assets), torch.randn(2, assets)
    for router in (SoftmaxResidualRouter(condition_dim=8, expert_hidden_dim=8),
                   EvidentialResidualRouter(condition_dim=8, expert_hidden_dim=8, uncertainty_threshold=.5)):
        loss = routing.routed_diffusion_loss(model, router, condition, target, torch.tensor([0, 2]), noise)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.parameters())
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in router.experts.parameters())


@pytest.fixture(params=NEW_NOTEBOOKS, ids=lambda path: path.stem)
def research_book(request):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("IPython")
    book = nbformat.read(request.param, as_version=4)
    return request.param, book, {cell.id: cell for cell in book.cells}


def test_notebook_schema_syntax_cli_arguments_and_isolation(research_book, monkeypatch, tmp_path):
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    from diffusion_portfolio.evaluation import forecasts
    from diffusion_portfolio.training import runs
    from diffusion_portfolio.data import diagnostics

    path, book, cells = research_book
    nbformat.validate(book)
    reference = nbformat.read(ROOT / "notebooks/high_dimensional/ff49_neural_cde.ipynb", as_version=4)
    assert cells["github-sync"].source == next(c.source for c in reference.cells if c.id == "github-sync")
    if path.stem.endswith("_tda"):
        assert cells["dependencies"].source == "!pip install -q -e '.[dev,research]'"
        order = [cell.id for cell in book.cells]
        assert order.index("dependencies") < order.index("topology-dependency") < order.index("tests")
        assert order.index("topology-dependency") < order.index("feature-smoke")
        with monkeypatch.context() as missing_gudhi:
            missing_gudhi.setitem(sys.modules, "gudhi", None)
            with pytest.raises(ModuleNotFoundError, match="gudhi"):
                exec(cells["topology-dependency"].source, {})
    for module, name in ((multiseed, "run_multiseed"), (probes, "run_probe"), (routing, "run_router"),
                         (precompute, "precompute_features"), (runs, "train_feature_ablation"),
                         (forecasts, "evaluate_diffusion"), (diagnostics, "run_data_check")):
        monkeypatch.setattr(module, name, Mock(return_value={}))
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        assert cell.outputs == [] and cell.execution_count is None
        source = cell.source
        assert not any(term in source for term in ("%%bash", "subprocess", "MODE", "REVISION", "n_assets=12"))
        transformed = TransformerManager().transform_cell(source)
        compile(transformed, f"{path.name}:{cell.id}", "exec")
        for node in ast.walk(ast.parse(transformed)):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "system":
                argv = shlex.split(ast.literal_eval(node.args[0]))
                scripts = [i for i, value in enumerate(argv) if value.startswith("scripts/")]
                if not scripts:
                    continue  # Compile Git/pip cells; never execute them.
                i = scripts[0]
                script = argv[i]
                assert source.splitlines()[-1].startswith("assert _exit_code == 0")
                if "--split" in argv:
                    assert argv[argv.index("--split") + 1] == "validation"
                if "--cde-run-dir" in argv:
                    expected = "/kaggle/working/ff49_cde_clean" if path.stem.startswith("ff49") else "/kaggle/working/ff100_cde"
                    assert argv[argv.index("--cde-run-dir") + 1] == expected
                assert "configs/mvp.yaml" not in argv and "configs/cde.yaml" not in argv
                if "--report-json" in argv:
                    argv[argv.index("--report-json") + 1] = str(tmp_path / "cli_report.json")
                monkeypatch.setattr(sys, "argv", [script, *argv[i + 1:]])
                runpy.run_path(str(ROOT / script), run_name="__main__")
    ns = {}
    hub = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    exec(cells["paths"].source, ns)
    assert ns["EXPERIMENT"] == "high_dimensional/" + path.stem
    assert ns["ROOT"] not in ns["REFERENCES"]
    assert not any(ns["ROOT"].is_relative_to(p) for p in ns["REFERENCES"])


def test_missing_prerequisite_and_failed_shell_stop_execution(research_book, tmp_path, monkeypatch):
    from IPython.core.inputtransformer2 import TransformerManager
    _, _, cells = research_book
    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace())
    ns = {}
    exec(cells["paths"].source.replace("/kaggle/working/", f"{tmp_path}/"), ns)
    with pytest.raises(FileNotFoundError, match="prerequisite"):
        exec(cells["prerequisites"].source, ns)
    ns.update(require_completed_run=Mock(), check_checkpoint_pair=Mock())
    shell = SimpleNamespace(system=Mock(side_effect=lambda command: ns.update(_exit_code=9)))
    ns["get_ipython"] = lambda: shell
    for cell in cells.values():
        if cell.cell_type == "code" and "!PYTHONUNBUFFERED=1 python -u scripts/" in cell.source:
            with pytest.raises(AssertionError, match="failed"):
                exec(TransformerManager().transform_cell(cell.source), ns)


def test_notebook_restores_only_references_and_own_namespace(research_book, tmp_path, monkeypatch):
    path, _, cells = research_book
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "unrelated.pt").write_bytes(b"not a prerequisite")
    hub = SimpleNamespace(dataset_download=Mock(return_value=str(remote)))
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    ns = {}
    exec(cells["paths"].source.replace("/kaggle/working/", f"{tmp_path}/local/"), ns)
    assets = 49 if path.stem.startswith("ff49") else 100
    for destination, config in ns["REFERENCES"].items():
        saved_run(remote / "high_dimensional" / destination.name, ROOT / config, assets)
    own = remote / ns["EXPERIMENT"]
    own.mkdir(parents=True)
    (own / "previous_results.txt").write_text("restore this experiment only")
    exec(cells["restore"].source, ns)
    exec(cells["prerequisites"].source, ns)
    assert (ns["ROOT"] / "previous_results.txt").is_file()
    assert not (ns["ROOT"] / "unrelated.pt").exists()
    for destination in ns["REFERENCES"]:
        assert (destination / "best.pt").is_file()


def test_notebook_backup_preserves_other_runs_and_stops_on_download_failure(research_book, tmp_path, monkeypatch):
    path, _, cells = research_book
    remote = tmp_path / "remote"
    protected = remote / "phase0/expensive.pt"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"historical checkpoint")
    hub = SimpleNamespace(dataset_download=Mock(return_value=str(remote)), dataset_upload=Mock())
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    ns = {}
    exec(cells["paths"].source.replace("/kaggle/working/", f"{tmp_path}/"), ns)
    root = ns["ROOT"]
    root.mkdir()
    ns["SOURCE"] = {"fixture": "source identity"}
    if path.stem.endswith("multiseed"):
        run = root / "vanilla/seed_43"
        saved_run(run, ROOT / "configs/vanilla_ff49.yaml", 49)
    elif path.stem.endswith(("tda", "spd")):
        (root / "features").mkdir()
        (root / "features/metadata.json").write_text(json.dumps({"source": ns["SOURCE"]}))
        for name in ("train.npz", "validation.npz"):
            np.savez(root / "features" / name, dates=np.asarray(["2000-01-01"]))
    else:
        probe = root / "current"
        probe.mkdir()
        for name in ("evidential_best.pt", "softmax_best.pt", "regime_thresholds.json", "validation_predictions.npz", "config.yaml"):
            (probe / name).write_bytes(b"backup fixture")
        (probe / "validation_metrics.json").write_text(json.dumps({"source": ns["SOURCE"]}))
    exec(cells["backup"].source, ns)
    staging = Path(hub.dataset_upload.call_args.args[1])
    assert (staging / "phase0/expensive.pt").read_bytes() == protected.read_bytes()
    assert (staging / ns["EXPERIMENT"]).is_dir()
    hub.dataset_upload.reset_mock()
    hub.dataset_download.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        exec(cells["backup"].source, ns)
    hub.dataset_upload.assert_not_called()
    assert (staging / "phase0/expensive.pt").read_bytes() == b"historical checkpoint"
