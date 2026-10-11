"""Run the probe/router entrypoints against tiny saved CDE experiments."""
from argparse import Namespace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import ReturnWindowDataset, TrainStandardizer, WindowedReturns, collate_return_batch
from diffusion_portfolio.data.preparation import save_standardizer
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.training.diffusion import evaluate_diffusion_loss
from diffusion_portfolio.training import probes, routing


@pytest.fixture
def retained_cde(tmp_path):
    raw_config = {
        "seed": 41, "data": {"lookback": 4},
        "model": {
            "history_encoder": "cde", "condition_dim": 8, "history_hidden_dim": 16,
            "diffusion_steps": 5, "channels": [8], "time_embed_dim": 8, "n_res_blocks": 1,
            "cde_hidden_dim": 8, "cde_drift_hidden_dim": 16, "cde_sensitivity_hidden_dim": 16,
            "cde_solver": "rk4", "cde_use_adjoint": False, "cde_fixed_steps_per_interval": 2,
        },
        "training": {"batch_size": 4}, "evaluation": {"n_scenarios": 4},
    }
    run_dir = tmp_path / "cde"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(raw_config))
    cfg = load_config(run_dir / "config.yaml")
    rng = np.random.default_rng(19)

    def dataset(start, count):
        windows = WindowedReturns(
            rng.normal(0, .1, (count, 4, 3)).astype(np.float32),
            rng.normal(0, .1, (count, 1, 3)).astype(np.float32),
            pd.bdate_range(start, periods=count),
        )
        return ReturnWindowDataset(windows, windows)

    class DevelopmentSplits:
        train = dataset("1999-12-01", 12)
        val = dataset("2000-01-03", 6)

        @property
        def test(self):
            raise AssertionError("Probe/router accessed the held-out split")

    splits = DevelopmentSplits()
    scaler = TrainStandardizer.identity(("a", "b", "c"))
    save_standardizer(run_dir / "standardizer.npz", scaler)
    torch.manual_seed(cfg.seed)
    model = build_diffusion(cfg, n_assets=3)
    loader = torch.utils.data.DataLoader(splits.val, batch_size=4, collate_fn=collate_return_batch)
    val_loss = evaluate_diffusion_loss(
        model, loader, device=torch.device("cpu"), validation_seed=cfg.training.validation_seed,
    )
    torch.save({"epoch": 1, "val_loss": val_loss, "model_state_dict": model.state_dict()}, run_dir / "best.pt")
    (run_dir / "summary.json").write_text(json.dumps({"best_epoch": 1, "best_val_loss": val_loss}))
    (run_dir / "validation_probabilistic_metrics.json").write_text(json.dumps({
        "crps_mean": .01, "energy_score": .02, "calibration": [{"ace": .1}],
    }))
    config = yaml.safe_load((Path(__file__).parents[2] / "configs/evidential.yaml").read_text())
    config["model"]["input_dim"] = 8
    config["training"].update(epochs=1, batch_size=4)
    config["regimes"]["recent_days"] = 2
    probe_config = tmp_path / "probe.yaml"
    probe_config.write_text(yaml.safe_dump(config))
    return Namespace(run_dir=run_dir, cfg=cfg, splits=splits, scaler=scaler, probe_config=probe_config)


@pytest.mark.parametrize("target", ["next_day", "current"])
def test_probe_entrypoint_restores_cde_and_preserves_reproducibility(retained_cde, tmp_path, monkeypatch, target):
    run = retained_cde
    # Patch source acquisition only: checkpoint loading, CDE extraction,
    # thresholds, optimization and artifact serialization all execute.
    monkeypatch.setattr(probes, "prepare_return_data", lambda cfg: (run.splits, run.scaler), raising=False)
    first, second = tmp_path / "probe1", tmp_path / "probe2"
    for output in (first, second):
        probes.run_probe(Namespace(
            cde_run_dir=str(run.run_dir), config=str(run.probe_config), output_dir=str(output),
            device="cpu", regime_target=target,
        ))
    for filename in ("regime_thresholds.json", "validation_metrics.json", "evidential_history.csv", "softmax_history.csv"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    with np.load(first / "validation_predictions.npz") as a, np.load(second / "validation_predictions.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    for filename in ("evidential_best.pt", "softmax_best.pt"):
        a = torch.load(first / filename, weights_only=False)["model_state_dict"]
        b = torch.load(second / filename, weights_only=False)["model_state_dict"]
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)


@pytest.mark.parametrize("validation_stream", ["separate", "legacy", "invalid"])
def test_router_entrypoint_reuses_saved_scaler_and_restores_checkpoints(retained_cde, tmp_path, monkeypatch, validation_stream):
    run = retained_cde
    checkpoint = torch.load(run.run_dir / "best.pt", weights_only=False)
    model = build_diffusion(run.cfg, n_assets=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    conditions = routing.extract_conditions(model.history_encoder, run.splits.val, batch_size=4, device=torch.device("cpu"))
    legacy_loss = routing.evaluate_router_loss(
        model, torch.nn.Identity(), conditions, routing.model_targets(run.splits.val),
        batch_size=4, validation_seed=run.cfg.training.validation_seed, device=torch.device("cpu"),
    )
    if validation_stream != "separate":
        checkpoint["val_loss"] = legacy_loss if validation_stream == "legacy" else legacy_loss + 100
        torch.save(checkpoint, run.run_dir / "best.pt")
        (run.run_dir / "summary.json").write_text(json.dumps({"best_epoch": 1, "best_val_loss": checkpoint["val_loss"]}))
    monkeypatch.setattr(probes, "prepare_return_data", lambda cfg: (run.splits, run.scaler))
    probe_dir = tmp_path / "probe"
    probes.run_probe(Namespace(
        cde_run_dir=str(run.run_dir), config=str(run.probe_config), output_dir=str(probe_dir),
        device="cpu", regime_target="current",
    ))

    def prepare(cfg, *, scaler):
        assert scaler.columns == run.scaler.columns
        np.testing.assert_array_equal(scaler.mean, run.scaler.mean)
        np.testing.assert_array_equal(scaler.std, run.scaler.std)
        return run.splits

    monkeypatch.setattr(routing, "prepare_datasets", prepare)
    config = yaml.safe_load((Path(__file__).parents[2] / "configs/router.yaml").read_text())
    config["training"].update(epochs=1, batch_size=4)
    config["router"]["expert_hidden_dim"] = 8
    config["evaluation"]["sampling_batch_size"] = 2
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.safe_dump(config))
    outputs = (tmp_path / "router1", tmp_path / "router2")
    for output in outputs:
        args = Namespace(
            cde_run_dir=str(run.run_dir), probe_run_dir=str(probe_dir),
            config=str(config_path), output_dir=str(output), device="cpu",
        )
        if validation_stream == "invalid":
            with pytest.raises(RuntimeError, match="neither known corruption stream"):
                routing.run_router(args)
            assert not (output / "comparison.json").exists()
            return
        routing.run_router(args)
    assert (outputs[0] / "comparison.json").read_bytes() == (outputs[1] / "comparison.json").read_bytes()
    comparison = json.loads((outputs[0] / "comparison.json").read_text())
    assert comparison["CDE"]["validation_loss"] == legacy_loss
    if validation_stream == "separate":
        assert comparison["checkpoint_validation_loss"] == checkpoint["val_loss"]
        assert comparison["baseline_validation_stream"] == "router_legacy"
    else:
        assert "checkpoint_validation_loss" not in comparison
    for variant in ("softmax_moe", "evidential_moe"):
        first, second = (output / variant for output in outputs)
        for filename in ("history.csv", "validation_probabilistic_metrics.json"):
            assert (first / filename).read_bytes() == (second / filename).read_bytes()
        with np.load(first / "validation_scenarios.npz") as a, np.load(second / "validation_scenarios.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        a = torch.load(first / "best.pt", weights_only=False)["router_state_dict"]
        b = torch.load(second / "best.pt", weights_only=False)["router_state_dict"]
        assert all(torch.equal(a[key], b[key]) for key in a)
