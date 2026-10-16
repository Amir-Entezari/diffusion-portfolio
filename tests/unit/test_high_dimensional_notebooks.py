"""Offline checks for interactive notebooks; fixture losses are not research results."""
import ast
import json
from pathlib import Path
import shlex
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from diffusion_portfolio.config import load_config


nbformat = pytest.importorskip("nbformat")
TransformerManager = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager
ROOT = Path(__file__).parents[2]
RUN_NAMES = {
    "ff49_vanilla": "ff49_vanilla",
    "ff49_neural_cde": "ff49_cde_clean",
    "ff100_vanilla": "ff100_vanilla",
    "ff100_neural_cde": "ff100_cde",
}
NOTEBOOKS = sorted(ROOT / "notebooks/high_dimensional" / f"{name}.ipynb" for name in RUN_NAMES)
REQUIRED = ("config.yaml", "standardizer.npz", "best.pt", "last.pt")


@pytest.fixture(params=NOTEBOOKS, ids=lambda path: path.stem)
def notebook(request):
    book = nbformat.read(request.param, as_version=4)
    return request.param, book, {cell.id: cell for cell in book.cells}


def script_arguments(cell, script):
    """Read literal CLI arguments without executing a notebook command."""
    tree = ast.parse(TransformerManager().transform_cell(cell.source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "system":
            command = ast.literal_eval(node.args[0])
            if script in command:
                return shlex.split(command[command.index(script):])
    raise AssertionError(f"No direct IPython command for {script}")


def write_checkpoint_pair(run_dir):
    """Minimal checkpoint metadata used by notebook safety checks."""
    torch.save({"epoch": 1, "val_loss": .2}, run_dir / "best.pt")
    torch.save({"epoch": 2, "best_epoch": 1, "best_val_loss": .2}, run_dir / "last.pt")


def test_interactive_structure_syntax_and_matched_configuration(notebook):
    path, book, cells = notebook
    nbformat.validate(book)
    original = nbformat.read(ROOT / "notebooks/spd/spd_kf12.ipynb", as_version=4)
    assert cells["github-sync"].source == original.cells[2].source.strip().replace(
        'BRANCH = "main"', 'BRANCH = "feature/high-dimensional-kaggle"'
    )
    transformer = TransformerManager()
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        transformed = transformer.transform_cell(cell.source)
        compile(transformed, f"{path.name}:{cell.id}", "exec")
        tree = ast.parse(transformed)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert cell.id == "checkpoint-check" and node.name == "check_checkpoints"
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not names.intersection({"MODE", "FULL_COMPLETE", "REVISION", "run_training", "inspect_run"})
        assert not any(term in cell.source for term in (
            "subprocess.run", "subprocess.Popen", "%%bash", "%%capture",
        ))
        # Saved executions are research history; nbformat validates their schema.
        assert "n_assets=12" not in cell.source

    order = list(cells)
    assert order.index("regression-tests") < order.index("data-smoke") < order.index("model-smoke")
    assert order.index("model-smoke") < order.index("restore") < order.index("pilot")
    assert order.index("checkpoint-check") < order.index("restore")
    assert order[order.index("pilot") + 1] == "pilot-losses"
    assert order[order.index("full") + 1] == "full-losses"
    pilot = script_arguments(cells["pilot"], "scripts/train_diffusion.py")
    full = script_arguments(cells["full"], "scripts/train_diffusion.py")
    run_name = RUN_NAMES[path.stem]
    config = f'configs/{"cde" if "neural_cde" in path.stem else "vanilla"}_{path.stem.split("_")[0]}.yaml'
    for command in (pilot, full):
        assert "--resume" in command
        assert command[command.index("--config") + 1] == config
        assert command[command.index("--output-dir") + 1] == f"/kaggle/working/{run_name}"
    for cell_id in ("pilot", "full"):
        assert cells[cell_id].source.startswith("check_checkpoints(RUN_DIR)\n\n!PYTHONUNBUFFERED=1 ")
        assert " python -u scripts/train_diffusion.py" in cells[cell_id].source
        assert "|" not in cells[cell_id].source
        if path.stem == "ff49_neural_cde":
            assert "OMP_NUM_THREADS=1" in cells[cell_id].source
    assert pilot[pilot.index("--stop-after-epoch") + 1] == "2"
    assert "--stop-after-epoch" not in full
    for cell_id in ("pilot", "full", "evaluation"):
        lines = cells[cell_id].source.splitlines()
        assert lines[-1].startswith("assert _exit_code == 0,")
        assert lines[-2].strip() in ("--device cuda", "--resume", "--stop-after-epoch 2")
    assert cells["backup"].source.index("check_checkpoints(RUN_DIR)") < cells["backup"].source.index("dataset_download")
    evaluation = script_arguments(cells["evaluation"], "scripts/evaluate_diffusion.py")
    assert evaluation[evaluation.index("--split") + 1] == "validation"
    assert evaluation[evaluation.index("--run-dir") + 1] == f"/kaggle/working/{run_name}"
    assert "--model-smoke" in cells["model-smoke"].source
    assert config in cells["data-smoke"].source and config in cells["model-smoke"].source
    assert f'EXPERIMENT = "high_dimensional/{run_name}"' in cells["restore"].source

    cfg = load_config(ROOT / config)
    reference = load_config(ROOT / f'configs/cde_{path.stem.split("_")[0]}.yaml')
    assert cfg.data == reference.data and cfg.seed == reference.seed
    assert cfg.training.epochs == 50
    if "vanilla" in path.stem:
        baseline = load_config(ROOT / "configs/mvp.yaml")
        assert cfg.model == baseline.model
        assert cfg.training == baseline.training
        assert cfg.evaluation == baseline.evaluation
    else:
        assert cfg == reference


@pytest.mark.parametrize("complete", [False, True], ids=["partial", "complete"])
def test_loss_plots_read_saved_csv_and_refresh_after_resume(notebook, tmp_path, monkeypatch, complete):
    _, _, cells = notebook
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    count = 50 if complete else 2
    history = pd.DataFrame({
        "epoch": range(1, count + 1),
        "train_loss": [1 / epoch for epoch in range(1, count + 1)],
        "val_loss": [2 / epoch for epoch in range(1, count + 1)],
    })
    history.to_csv(tmp_path / "history.csv", index=False)
    if complete:
        (tmp_path / "summary.json").write_text(json.dumps({"best_epoch": count, "best_val_loss": 2 / count}))
    ns = {"RUN_DIR": tmp_path, "display": Mock()}
    for cell_id in ("pilot-losses", "full-losses"):
        exec(cells[cell_id].source, ns)
        assert ns["history"].shape == (count, 3)
        assert ns["ax"].get_xlabel() == "Epoch"
        assert ns["ax"].get_ylabel() == "Diffusion loss"
        assert [line.get_label() for line in ns["ax"].lines] == ["Train Loss", "Validation Loss"]
        assert ns["ax"].lines[1].get_ydata()[-1] == pytest.approx(2 / count)
        plt.close("all")
    history.loc[len(history)] = [count + 1, .01, .02]
    history.to_csv(tmp_path / "history.csv", index=False)
    exec(cells["pilot-losses"].source, ns)
    assert ns["ax"].lines[0].get_xdata()[-1] == count + 1
    assert ns["ax"].lines[1].get_ydata()[-1] == .02
    plt.close("all")


@pytest.fixture
def backup_run(notebook, tmp_path, monkeypatch):
    path, _, cells = notebook
    run = tmp_path / path.stem
    run.mkdir()
    for name in REQUIRED:
        (run / name).write_bytes(b"local checkpoint fixture")
    write_checkpoint_pair(run)
    hub = SimpleNamespace(dataset_download=Mock(), dataset_upload=Mock())
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    namespace = {"RUN_DIR": run, "HANDLE": "amirentezari/diffusion-portfolio",
                 "EXPERIMENT": f'high_dimensional/{RUN_NAMES[path.stem]}'}
    exec(cells["checkpoint-check"].source, namespace)
    return cells["backup"].source, namespace, hub


@pytest.mark.parametrize("failure", ["download", "empty", "missing-checkpoint", "epoch-mismatch", "loss-mismatch"])
def test_backup_failure_never_uploads_or_replaces_staging(backup_run, tmp_path, failure):
    source, ns, hub = backup_run
    staging = tmp_path / "artifact_sync"
    staging.mkdir()
    (staging / "previous.txt").write_text("keep")
    empty = tmp_path / "empty"
    empty.mkdir()
    hub.dataset_download.return_value = str(empty)
    if failure == "download":
        hub.dataset_download.side_effect = RuntimeError("offline")
    elif failure == "missing-checkpoint":
        (ns["RUN_DIR"] / "last.pt").unlink()
    elif failure in ("epoch-mismatch", "loss-mismatch"):
        best = {"epoch": 1, "val_loss": .2}
        best["epoch" if failure == "epoch-mismatch" else "val_loss"] = 99
        torch.save(best, ns["RUN_DIR"] / "best.pt")
    with pytest.raises((RuntimeError, FileNotFoundError, AssertionError)):
        exec(source, ns)
    hub.dataset_upload.assert_not_called()
    assert (staging / "previous.txt").read_text() == "keep"


def test_partial_backup_preserves_all_unrelated_experiments(backup_run, tmp_path):
    source, ns, hub = backup_run
    remote = tmp_path / "remote"
    protected = ("phase0/neural_cde_rk4x4/best.pt", "baselines/vanilla_vpred/history.csv",
                 "high_dimensional/another_experiment/last.pt")
    for name in protected:
        target = remote / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"preserve exactly")
    hub.dataset_download.return_value = str(remote)
    assert not (ns["RUN_DIR"] / "summary.json").exists()
    exec(source, ns)
    hub.dataset_download.assert_called_once_with(ns["HANDLE"], force_download=True)
    hub.dataset_upload.assert_called_once()
    staging = Path(hub.dataset_upload.call_args.args[1])
    for name in protected:
        assert (staging / name).read_bytes() == (remote / name).read_bytes() == b"preserve exactly"
    for name in REQUIRED:
        assert (staging / ns["EXPERIMENT"] / name).read_bytes() == (ns["RUN_DIR"] / name).read_bytes()


def test_restore_copies_only_selected_run_and_keeps_local_changes(notebook, tmp_path, monkeypatch):
    path, _, cells = notebook
    run_name = RUN_NAMES[path.stem]
    config = f'configs/{"cde" if "neural_cde" in path.stem else "vanilla"}_{path.stem.split("_")[0]}.yaml'
    remote = tmp_path / "remote"
    saved = remote / "high_dimensional" / run_name
    saved.mkdir(parents=True)
    for name in REQUIRED:
        (saved / name).write_bytes(b"checkpoint fixture")
    write_checkpoint_pair(saved)
    shutil.copy2(ROOT / config, saved / "config.yaml")
    (remote / "unrelated.txt").write_text("historical artifact")
    hub = SimpleNamespace(dataset_download=Mock(return_value=str(remote)))
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    source = cells["restore"].source.replace("/kaggle/working/", f"{tmp_path}/")
    ns = {}
    exec(cells["checkpoint-check"].source, ns)
    exec(source, ns)
    assert (ns["RUN_DIR"] / "last.pt").is_file()
    assert not (ns["RUN_DIR"] / "unrelated.txt").exists()
    last = torch.load(ns["RUN_DIR"] / "last.pt", weights_only=False)
    last["epoch"] = 3
    torch.save(last, ns["RUN_DIR"] / "last.pt")
    progress = (ns["RUN_DIR"] / "last.pt").read_bytes()
    exec(source, ns)
    assert (ns["RUN_DIR"] / "last.pt").read_bytes() == progress


def test_plots_use_latest_checkpoint_history_after_interruption(notebook, tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    _, _, cells = notebook
    history = pd.DataFrame({"epoch": [1, 2, 3], "train_loss": [2., 1., .5], "val_loss": [3., 2., 1.]})
    history.iloc[:2].to_csv(tmp_path / "history.csv", index=False)
    torch.save({"history": history.to_dict("records")}, tmp_path / "last.pt")
    ns = {"RUN_DIR": tmp_path, "display": Mock()}
    for cell_id in ("pilot-losses", "full-losses"):
        exec(cells[cell_id].source, ns)
        pd.testing.assert_frame_equal(ns["history"], history)
        assert ns["ax"].lines[1].get_xdata()[-1] == 3
        plt.close("all")


def test_command_exit_checks_detect_failures(notebook, backup_run):
    _, _, cells = notebook
    _, ns, _ = backup_run
    (ns["RUN_DIR"] / "summary.json").write_text("{}")
    for cell_id in ("pilot", "full", "evaluation"):
        for status in (0, 9):
            ns["_exit_code"] = -1
            shell = SimpleNamespace(system=Mock(side_effect=lambda command: ns.update(_exit_code=status)))
            ns["get_ipython"] = lambda: shell
            source = TransformerManager().transform_cell(cells[cell_id].source)
            if status:
                with pytest.raises(AssertionError, match="command failed"):
                    exec(source, ns)
            else:
                exec(source, ns)
            shell.system.assert_called_once()


def test_ipython_exposes_real_command_exit_status(tmp_path, monkeypatch):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    shell = InteractiveShell()
    for status in (0, 9):
        command = (f"!PYTHONUNBUFFERED=1 {shlex.quote(sys.executable)} -u "
                   f"-c 'raise SystemExit({status})'\nassert _exit_code == 0")
        with capture_output():
            result = shell.run_cell(command, store_history=False)
        assert shell.user_ns["_exit_code"] == status
        if status:
            assert isinstance(result.error_in_exec, AssertionError)
        else:
            assert result.error_in_exec is None


def test_checkpoint_check_rejects_incomplete_or_inconsistent_pairs(notebook, tmp_path):
    _, _, cells = notebook
    ns = {}
    exec(cells["checkpoint-check"].source, ns)
    for case in ("best-only", "last-only", "epoch", "loss", "missing-metadata"):
        run = tmp_path / case
        run.mkdir()
        write_checkpoint_pair(run)
        if case in ("best-only", "last-only"):
            (run / ("last.pt" if case == "best-only" else "best.pt")).unlink()
        elif case == "epoch":
            # The actual failure being guarded against: last=16, recorded best=15, best.pt=1.
            torch.save({"epoch": 16, "best_epoch": 15, "best_val_loss": .2}, run / "last.pt")
        elif case == "loss":
            torch.save({"epoch": 1, "val_loss": .9}, run / "best.pt")
        else:
            torch.save({"epoch": 1}, run / "best.pt")
        before = {p.name: p.read_bytes() for p in run.iterdir()}
        with pytest.raises((RuntimeError, AssertionError), match="checkpoint"):
            ns["check_checkpoints"](run)
        assert {p.name: p.read_bytes() for p in run.iterdir()} == before


def test_fresh_training_without_checkpoints_is_allowed(notebook, tmp_path):
    _, _, cells = notebook
    run = tmp_path / "fresh"
    ns = {"RUN_DIR": run}
    exec(cells["checkpoint-check"].source, ns)
    for existing_directory in (False, True):
        if existing_directory:
            run.mkdir()
        shell = SimpleNamespace(system=Mock(side_effect=lambda command: ns.update(_exit_code=0)))
        ns["get_ipython"] = lambda: shell
        for cell_id in ("pilot", "full"):
            exec(TransformerManager().transform_cell(cells[cell_id].source), ns)
        assert shell.system.call_count == 2
    assert list(run.iterdir()) == []


def test_restore_rejects_inconsistent_backup_before_copy(notebook, tmp_path, monkeypatch):
    path, _, cells = notebook
    remote = tmp_path / "remote"
    saved = remote / "high_dimensional" / RUN_NAMES[path.stem]
    saved.mkdir(parents=True)
    write_checkpoint_pair(saved)
    torch.save({"epoch": 16, "best_epoch": 15, "best_val_loss": .2}, saved / "last.pt")
    hub = SimpleNamespace(dataset_download=Mock(return_value=str(remote)))
    monkeypatch.setitem(sys.modules, "kagglehub", hub)
    source = cells["restore"].source.replace("/kaggle/working/", f"{tmp_path}/")
    ns = {}
    exec(cells["checkpoint-check"].source, ns)
    with pytest.raises(AssertionError, match="Inconsistent checkpoints"):
        exec(source, ns)
    assert not ns["RUN_DIR"].exists()
