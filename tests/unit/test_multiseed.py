"""Exercise multiseed orchestration without training or reading real returns."""
from argparse import Namespace
import json
from pathlib import Path

import pytest
import yaml

from diffusion_portfolio.training import multiseed


def write_result(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "best_epoch": 2, "best_val_loss": .25,
    }))
    (run_dir / "validation_probabilistic_metrics.json").write_text(json.dumps({
        "crps_mean": .01, "energy_score": .02,
        "calibration": [{"ace": -.1}, {"ace": .2}],
    }))


@pytest.mark.parametrize("relative_paths", [False, True])
def test_multiseed_uses_supplied_repository_and_caller_paths(tmp_path, monkeypatch, relative_paths):
    repo = tmp_path / "repo"
    caller = tmp_path / "caller"
    caller.mkdir()
    (repo / "configs").mkdir(parents=True)
    for filename in ("mvp.yaml", "cde.yaml"):
        (repo / "configs" / filename).write_text("seed: 42\n")
    write_result(caller / "vanilla42")
    write_result(caller / "cde42")
    monkeypatch.chdir(caller)
    calls = []

    def run_command(command, *, cwd):
        assert cwd == repo.resolve()
        calls.append(command)
        if command[1] == "scripts/train_diffusion.py":
            config_path = Path(command[command.index("--config") + 1])
            assert config_path.is_absolute()
            assert yaml.safe_load(config_path.read_text())["seed"] == 43
            output = Path(command[command.index("--output-dir") + 1])
            assert output.is_absolute()
            write_result(output)
            # Force the independent validation command to execute too.
            (output / "validation_probabilistic_metrics.json").unlink()
        else:
            assert command[1] == "scripts/evaluate_diffusion.py"
            output = Path(command[command.index("--run-dir") + 1])
            assert output.is_absolute()
            write_result(output)

    monkeypatch.setattr(multiseed, "run_command", run_command)
    path = lambda name: name if relative_paths else str(caller / name)
    args = Namespace(
        seed42_vanilla_dir=path("vanilla42"), seed42_cde_dir=path("cde42"),
        output_root=path("results"), seeds=[43], device="cpu",
    )
    multiseed.run_multiseed(args, repo_root=str(repo))
    assert len(calls) == 4
    assert (caller / "results/per_seed_results.csv").is_file()
    summary = json.loads((caller / "results/summary.json").read_text())
    assert summary["summary"]["paired"]["crps"]["n_seeds"] == 2
