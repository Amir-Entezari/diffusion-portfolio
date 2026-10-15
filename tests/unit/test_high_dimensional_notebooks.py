"""Offline notebook orchestration checks; checkpoint numbers are test fixtures only."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import torch

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import TrainStandardizer
from diffusion_portfolio.data.preparation import load_standardizer, save_standardizer


ROOT = Path(__file__).parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks/high_dimensional").glob("*.ipynb"))


def cells(path):
    result = {}
    for cell in json.loads(path.read_text())["cells"]:
        if isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])
        result[cell["id"]] = cell
    return result


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_structure_defaults_and_matched_configuration(path):
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    by_id = cells(path)
    exec(by_id["configuration"]["source"], namespace)
    assert namespace["MODE"] == "smoke" and namespace["BRANCH"] == "main"
    assert namespace["UPLOAD_ARTIFACTS"] is False and namespace["PILOT_EPOCHS"] == 2
    assert namespace["FULL_STOP_AFTER_EPOCH"] is None
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:{cell.id}", "exec")
            assert cell.outputs == [] and cell.execution_count is None
            assert "n_assets=12" not in cell.source
    cfg = load_config(ROOT / namespace["CONFIG_PATH"])
    reference = load_config(ROOT / namespace["DATA_REFERENCE"])
    assert cfg.data == reference.data and cfg.seed == reference.seed
    if namespace["EXPECTED_ENCODER"] == "mlp":
        vanilla = load_config(ROOT / "configs/mvp.yaml")
        assert cfg.model == vanilla.model
        assert cfg.training == vanilla.training
        assert cfg.evaluation == vanilla.evaluation
    else:
        assert cfg == reference
    evaluation = by_id["evaluation"]["source"]
    assert '"--split", "validation"' in evaluation and '"--split", "test"' not in evaluation
    assert "--model-smoke" in by_id["smoke"]["source"]


@pytest.fixture(params=NOTEBOOKS, ids=lambda p: p.stem)
def notebook_run(request, tmp_path):
    by_id = cells(request.param)
    ns = dict(Path=Path, np=np, pd=pd, torch=torch, json=json, asdict=asdict,
              load_config=load_config, load_standardizer=load_standardizer,
              hashlib=hashlib, shutil=shutil, subprocess=subprocess, sys=sys, display=Mock())
    exec(by_id["configuration"]["source"], ns)
    ns["CONFIG_PATH"] = str(ROOT / ns["CONFIG_PATH"])
    cfg = load_config(ns["CONFIG_PATH"])
    count = ns["EXPECTED_ASSETS"]
    ns.update(cfg=cfg, CONFIG=json.loads(json.dumps(asdict(cfg))),
              RUN_ROOT=tmp_path / "local", GIT_SHA="unit-test-commit",
              device="cpu",
              ARTIFACT_DATASET="amirentezari/diffusion-portfolio",
              ARTIFACT_NAMESPACE=f"high_dimensional/{ns['EXPERIMENT']}/seed42",
              SMOKE_PASSED=True, FULL_COMPLETE=False, EVAL_BATCH_SIZE=32)
    for name in ("pilot", "full", "smoke"):
        ns[name.upper() + "_DIR"] = ns["RUN_ROOT"] / name
    ns["SMOKE"] = {"columns": [f"asset{i}" for i in range(count)],
                   "standardizer": {"mean": [0.] * count, "std": [1.] * count},
                   "train_samples": 12, "validation_samples": 4, "test_samples": 5}
    definitions = [node for node in ast.parse(by_id["dependencies"]["source"]).body
                   if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook-helpers", "exec"), ns)
    ns["ARCHIVES"] = [tmp_path / "returns.zip", tmp_path / "rf.zip"]
    ns["SMOKE_DIR"].mkdir(parents=True)
    (ns["SMOKE_DIR"] / "report.json").write_text('{"unit_fixture": true}')
    (ns["RUN_ROOT"] / "inputs").mkdir()
    for archive in ns["ARCHIVES"]:
        archive.write_bytes(b"unit fixture, not downloaded data")
        shutil.copy2(archive, ns["RUN_ROOT"] / "inputs" / archive.name)
    ns["provenance"] = {"git_commit": ns["GIT_SHA"], "config": ns["CONFIG"],
                        "archives": {p.name: ns["sha256"](p) for p in ns["ARCHIVES"]}}
    (ns["RUN_ROOT"] / "provenance.json").write_text(json.dumps(ns["provenance"]))
    exec(by_id["run-guards"]["source"], ns)
    exec(by_id["training-command"]["source"], ns)
    ns["run_logged"] = Mock(side_effect=AssertionError("Unexpected subprocess"))
    return by_id, ns


def write_run(folder, ns, epoch, *, complete=False):
    """Minimal valid-format checkpoint metadata; never represented as research output."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ns["CONFIG_PATH"], folder / "config.yaml")
    save_standardizer(folder / "standardizer.npz", TrainStandardizer.identity(tuple(ns["SMOKE"]["columns"])))
    history = [{"epoch": i, "train_loss": 1 / i, "val_loss": 2 / i} for i in range(1, epoch + 1)]
    state = {"epoch": epoch, "best_epoch": epoch, "best_val_loss": 2 / epoch,
             "history": history, "model_state_dict": {}, "optimizer_state_dict": {},
             "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": None,
             "loader_rng_state": torch.Generator().get_state()}
    torch.save(state, folder / "last.pt")
    torch.save({"epoch": epoch, "val_loss": 2 / epoch, "model_state_dict": {}}, folder / "best.pt")
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    if complete:
        (folder / "summary.json").write_text(json.dumps({
            "best_epoch": epoch, "best_val_loss": 2 / epoch, "parameter_count": 1,
            "train_samples": 12, "val_samples": 4,
        }))


def test_smoke_default_does_not_train_evaluate_or_upload(notebook_run):
    by_id, ns = notebook_run
    ns["run_training"] = Mock(side_effect=AssertionError("Training must be opt-in"))
    ns["kagglehub"] = types.SimpleNamespace(dataset_upload=Mock(side_effect=AssertionError("Upload must be opt-in")))
    for cell_id in ("pilot", "full", "evaluation", "backup"):
        exec(by_id[cell_id]["source"], ns)
    ns["run_training"].assert_not_called()
    ns["run_logged"].assert_not_called()
    ns["kagglehub"].dataset_upload.assert_not_called()


def test_resume_guard_accepts_interrupted_history_and_rejects_partial_summary(notebook_run):
    _, ns = notebook_run
    folder = ns["PILOT_DIR"]
    write_run(folder, ns, 2)
    # history.csv is only written when the CLI returns; last.pt survives an interruption.
    (folder / "history.csv").unlink()
    assert ns["inspect_run"](folder)["epoch"] == 2
    assert ns["run_training"](folder, 2)["complete"] is False
    ns["run_logged"].assert_not_called()
    (folder / "summary.json").write_text('{}')
    with pytest.raises(AssertionError, match="Partial run"):
        ns["inspect_run"](folder)


def test_missing_checkpoint_and_mixed_config_fail_before_training(notebook_run):
    _, ns = notebook_run
    folder = ns["PILOT_DIR"]
    write_run(folder, ns, 2)
    (folder / "last.pt").unlink()
    with pytest.raises(RuntimeError, match="without last.pt"):
        ns["run_training"](folder, 2)
    write_run(folder, ns, 2)
    (folder / "config.yaml").write_text('seed: 999\n')
    with pytest.raises(AssertionError, match="Saved config mismatch"):
        ns["run_training"](folder, 2)
    ns["run_logged"].assert_not_called()


def test_completed_run_is_skipped_and_fresh_full_requires_pilot(notebook_run):
    by_id, ns = notebook_run
    ns["MODE"] = "full"
    with pytest.raises(RuntimeError, match="Complete the separate pilot"):
        exec(by_id["full"]["source"], ns)
    write_run(ns["FULL_DIR"], ns, 50, complete=True)
    exec(by_id["full"]["source"], ns)
    assert ns["FULL_COMPLETE"] is True
    ns["run_logged"].assert_not_called()


def test_pilot_command_and_partial_full_never_trigger_evaluation(notebook_run):
    by_id, ns = notebook_run
    ns.update(time=time, datetime=datetime, timezone=timezone, device="cpu", GPU_NAME=None,
              shutil=types.SimpleNamespace(which=lambda name: None))
    def train(command, log):
        assert "--resume" in command
        assert command[command.index("--stop-after-epoch") + 1] == "2"
        assert ns["cfg"].training.epochs == 50
        write_run(Path(command[command.index("--output-dir") + 1]), ns, 2)
    ns["run_logged"] = Mock(side_effect=train)
    ns["MODE"] = "pilot"
    exec(by_id["pilot"]["source"], ns)
    ns.update(MODE="full", FULL_STOP_AFTER_EPOCH=2)
    exec(by_id["full"]["source"], ns)
    assert ns["FULL_COMPLETE"] is False
    exec(by_id["evaluation"]["source"], ns)
    assert ns["run_logged"].call_count == 2
    assert not (ns["FULL_DIR"] / "summary.json").exists()


def test_failed_smoke_blocks_training(notebook_run):
    _, ns = notebook_run
    ns["SMOKE_PASSED"] = False
    with pytest.raises(AssertionError, match="smoke/provenance"):
        ns["run_training"](ns["PILOT_DIR"], 2)
    ns["run_logged"].assert_not_called()


def test_validation_reporting_reads_cli_artifacts_and_reuses_matching_outputs(notebook_run, monkeypatch):
    by_id, ns = notebook_run
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    ns.update(MODE="full", FULL_COMPLETE=True, plt=plt)
    write_run(ns["FULL_DIR"], ns, 50, complete=True)
    metrics = {"split": "validation", "checkpoint_epoch": 50, "checkpoint_val_loss": .04,
               "n_validation": 4, "n_scenarios": ns["cfg"].evaluation.n_scenarios,
               "sampling_seed": ns["cfg"].training.validation_seed,
               "sampling_batch_size": 32, "crps_mean": .01, "energy_score": .02,
               "calibration": [{"level": .9, "picp": .8, "ace": -.1}]}
    def evaluate(command, log):
        assert command[command.index("--split") + 1] == "validation"
        assert Path(command[command.index("--run-dir") + 1]) == ns["FULL_DIR"]
        (ns["FULL_DIR"] / "validation_probabilistic_metrics.json").write_text(json.dumps(metrics))
        (ns["FULL_DIR"] / "validation_scenarios.npz").write_bytes(b"unit test artifact")
    ns["run_logged"] = Mock(side_effect=evaluate)
    exec(by_id["evaluation"]["source"], ns)
    exec(by_id["evaluation"]["source"], ns)
    assert ns["run_logged"].call_count == 1
    result = pd.read_csv(ns["FULL_DIR"] / "validation_report.csv")
    assert result.iloc[0]["CRPS"] == .01
    assert result.iloc[0]["Encoder"] == ns["cfg"].model.history_encoder
    assert (ns["FULL_DIR"] / "validation_calibration.csv").is_file()
    assert (ns["FULL_DIR"] / "loss_curves.png").is_file()
    plt.close("all")


@pytest.mark.parametrize("empty_snapshot", [False, True])
def test_backup_download_failure_cannot_upload_empty_dataset(notebook_run, tmp_path, empty_snapshot):
    by_id, ns = notebook_run
    write_run(ns["PILOT_DIR"], ns, 2)
    ns.update(UPLOAD_ARTIFACTS=True, UPLOAD_STAGE="pilot")
    if empty_snapshot:
        empty = tmp_path / "empty_remote"
        empty.mkdir()
        download = Mock(return_value=str(empty))
    else:
        download = Mock(side_effect=RuntimeError("offline"))
    ns["kagglehub"] = types.SimpleNamespace(dataset_download=download, dataset_upload=Mock())
    ns["tempfile"] = types.SimpleNamespace(TemporaryDirectory=Mock())
    with pytest.raises(RuntimeError, match="Backup failed"):
        exec(by_id["backup"]["source"], ns)
    ns["tempfile"].TemporaryDirectory.assert_not_called()
    ns["kagglehub"].dataset_upload.assert_not_called()


def test_backup_preserves_other_stage_and_historical_artifacts(notebook_run, tmp_path):
    by_id, ns = notebook_run
    write_run(ns["PILOT_DIR"], ns, 2)
    remote = tmp_path / "remote"
    namespace = remote / ns["ARTIFACT_NAMESPACE"]
    write_run(namespace / "pilot", ns, 1)
    (namespace / "provenance.json").write_text(json.dumps(ns["provenance"]))
    protected = [remote / "phase0/neural_cde_rk4x4/best.pt", namespace / "full/sentinel"]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve exactly")
    uploads = []
    def upload(handle, directory, **kwargs):
        uploads.append({str(p.relative_to(directory)): p.read_bytes()
                        for p in Path(directory).rglob("*") if p.is_file()})
    ns.update(UPLOAD_ARTIFACTS=True, UPLOAD_STAGE="pilot")
    ns["kagglehub"] = types.SimpleNamespace(dataset_download=Mock(return_value=str(remote)), dataset_upload=Mock(side_effect=upload))
    ns["tempfile"] = types.SimpleNamespace(TemporaryDirectory=lambda **kwargs: tempfile.TemporaryDirectory(dir=tmp_path))
    exec(by_id["backup"]["source"], ns)
    ns["kagglehub"].dataset_download.assert_called_once_with(ns["ARTIFACT_DATASET"], force_download=True)
    assert len(uploads) == 1
    for path in protected:
        assert uploads[0][str(path.relative_to(remote))] == b"preserve exactly"
    assert uploads[0][ns["ARTIFACT_NAMESPACE"] + "/pilot/last.pt"] == (ns["PILOT_DIR"] / "last.pt").read_bytes()
    assert (ns["PILOT_DIR"] / "last.pt").exists()


def test_restore_copies_only_current_namespace(notebook_run, tmp_path, monkeypatch):
    by_id, ns = notebook_run
    write_run(ns["PILOT_DIR"], ns, 2)
    remote = tmp_path / "remote"
    shutil.copytree(ns["RUN_ROOT"], remote / ns["ARTIFACT_NAMESPACE"])
    protected = remote / "phase0/historical.txt"
    protected.parent.mkdir()
    protected.write_text("historical artifact")
    shutil.rmtree(ns["RUN_ROOT"])
    hub = types.ModuleType("kagglehub")
    hub.dataset_download = Mock(return_value=str(remote))
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    exec(by_id["restore"]["source"], ns)
    assert (ns["PILOT_DIR"] / "last.pt").exists()
    assert not (ns["RUN_ROOT"] / "phase0").exists()


def test_smoke_only_directory_does_not_block_restore_or_replace_local_training(notebook_run, tmp_path, monkeypatch):
    by_id, ns = notebook_run
    remote = tmp_path / "remote"
    namespace = remote / ns["ARTIFACT_NAMESPACE"]
    write_run(namespace / "pilot", ns, 2)
    (namespace / "provenance.json").write_text(json.dumps(ns["provenance"]))
    hub = types.ModuleType("kagglehub")
    hub.dataset_download = Mock(return_value=str(remote))
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    exec(by_id["restore"]["source"], ns)
    assert ns["inspect_run"](ns["PILOT_DIR"])["epoch"] == 2
    assert json.loads((ns["SMOKE_DIR"] / "report.json").read_text()) == {"unit_fixture": True}
    write_run(namespace / "pilot", ns, 1)
    exec(by_id["restore"]["source"], ns)
    assert ns["inspect_run"](ns["PILOT_DIR"])["epoch"] == 2


def test_backup_refuses_a_newer_remote_checkpoint(notebook_run, tmp_path):
    by_id, ns = notebook_run
    write_run(ns["PILOT_DIR"], ns, 2)
    remote = tmp_path / "remote"
    write_run(remote / ns["ARTIFACT_NAMESPACE"] / "pilot", ns, 3)
    (remote / ns["ARTIFACT_NAMESPACE"] / "provenance.json").write_text(json.dumps(ns["provenance"]))
    ns.update(UPLOAD_ARTIFACTS=True, UPLOAD_STAGE="pilot")
    ns["kagglehub"] = types.SimpleNamespace(dataset_download=Mock(return_value=str(remote)), dataset_upload=Mock())
    ns["tempfile"] = types.SimpleNamespace(TemporaryDirectory=lambda **kwargs: tempfile.TemporaryDirectory(dir=tmp_path))
    with pytest.raises(RuntimeError, match="Backup failed") as error:
        exec(by_id["backup"]["source"], ns)
    assert "newer" in str(error.value.__cause__)
    ns["kagglehub"].dataset_upload.assert_not_called()


def test_interrupted_cli_stops_its_process_group_before_returning(tmp_path):
    source = cells(NOTEBOOKS[0])["dependencies"]["source"]
    definition = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run_logged")
    process = Mock(pid=1234)
    process.stdout = MagicMock()
    process.stdout.__iter__.side_effect = KeyboardInterrupt
    context = MagicMock()
    context.__enter__.return_value = process
    terminate_group = Mock()
    popen = Mock(return_value=context)
    ns = {"Path": Path, "env": {}, "os": types.SimpleNamespace(killpg=terminate_group),
          "signal": types.SimpleNamespace(SIGTERM=15),
          "subprocess": types.SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "notebook-helper", "exec"), ns)
    with pytest.raises(KeyboardInterrupt):
        ns["run_logged"](["unit-test-only"], tmp_path / "log")
    assert popen.call_args.kwargs["start_new_session"] is True
    terminate_group.assert_called_once_with(1234, 15)
    process.wait.assert_called_once()
