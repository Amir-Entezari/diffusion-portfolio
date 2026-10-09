"""Small regressions for refactor-specific contracts and archived layouts."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from diffusion_portfolio.config import DataConfig, ModelConfig, MVPConfig, TrainingConfig, PortfolioConfig, EvaluationConfig
from diffusion_portfolio.data.features import standardize_features
from diffusion_portfolio.models.diffusion import PrecomputedConditionDiffusion
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.baselines.diffolio import DiffolioObjective


def small_config():
    return MVPConfig(
        seed=41, data=DataConfig(lookback=4),
        model=ModelConfig(
            condition_dim=8, history_hidden_dim=16, diffusion_steps=5,
            channels=(8,), time_embed_dim=8, n_res_blocks=1,
            cde_hidden_dim=8, cde_drift_hidden_dim=16, cde_sensitivity_hidden_dim=16,
            cde_solver="rk4", cde_rtol=1e-6, cde_atol=1e-7,
            cde_use_adjoint=False, cde_fixed_steps_per_interval=4,
        ),
        training=TrainingConfig(), portfolio=PortfolioConfig(), evaluation=EvaluationConfig(),
    )


@pytest.mark.parametrize("kind", ["gaussian", "cde", "levy2", "levy18", "precomputed", "diffolio"])
def test_state_dict_layout_matches_pre_refactor_fixture(kind, tmp_path):
    # These hashes were recorded from the original implementations before the
    # moves, using this small configuration and five assets. Keys, order and
    # shapes must remain identical: no key migration is needed.
    cfg = small_config()
    if kind == "diffolio":
        model = DiffolioObjective(
            training_covariance=torch.eye(5) * .001, n_assets=5,
            n_asset_characteristics=3, n_systematic=2, lookback=4,
            hidden_dim=8, num_heads=2, mlp_dim=16, time_embedding_dim=8,
            diffusion_steps=5, beta_start=.0001, beta_end=.02, lambda_corr=.05,
        )
    else:
        if kind == "cde":
            cfg = replace(cfg, model=replace(cfg.model, history_encoder="cde"))
        alpha = {"levy2": 2., "levy18": 1.8}.get(kind)
        if alpha is not None:
            cfg = replace(cfg, model=replace(cfg.model, prediction_type="epsilon"))
        model = build_diffusion(cfg, n_assets=5, alpha=alpha)
        if kind == "precomputed":
            model = PrecomputedConditionDiffusion(model, extra_dim=3)
    layout = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    fixtures = json.loads((Path(__file__).parents[1] / "fixtures/checkpoint_layouts.json").read_text())
    assert hashlib.sha256(json.dumps(layout).encode()).hexdigest() == fixtures[kind]
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert not model.load_state_dict(checkpoint["model_state_dict"], strict=True).missing_keys


@pytest.mark.parametrize("drop_constant", [False, True])
def test_feature_standardization_preserves_both_existing_constant_policies(drop_constant):
    train = np.array([[1, 5, -3], [3, 5, -1]], dtype=np.float32)
    validation = np.array([[5, 9, 1]], dtype=np.float32)
    train_z, val_z, mean, std, constant = standardize_features(
        train, validation, drop_constant=drop_constant
    )
    np.testing.assert_array_equal(mean, [2, 5, -2])
    np.testing.assert_array_equal(std, [1, 0, 1])
    np.testing.assert_array_equal(constant, [False, True, False])
    expected_train = [[-1, -1], [1, 1]] if drop_constant else [[-1, 0, -1], [1, 0, 1]]
    expected_val = [[3, 3]] if drop_constant else [[3, 4, 3]]
    np.testing.assert_array_equal(train_z, expected_train)
    np.testing.assert_array_equal(val_z, expected_val)
    changed = standardize_features(train, validation * 1000, drop_constant=drop_constant)
    np.testing.assert_array_equal(changed[0], train_z)
    np.testing.assert_array_equal(changed[2], mean)


def test_mlp_import_does_not_require_optional_cde():
    code = '''
import builtins
original = builtins.__import__
def no_cde(name, *args, **kwargs):
    if name == "torchcde":
        raise ImportError("optional dependency deliberately blocked")
    return original(name, *args, **kwargs)
builtins.__import__ = no_cde
from diffusion_portfolio.models.diffusion import ConditionalDiffusionModel, HistoryEncoder
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_cde_import_is_a_pure_reexport():
    from diffusion_portfolio.models.cde import NeuralCDE as legacy
    from diffusion_portfolio.models.encoders.cde import NeuralCDE
    assert legacy is NeuralCDE


def test_validation_evaluator_preserves_sampler_stream_and_avoids_test(tmp_path, monkeypatch):
    from diffusion_portfolio.data import ReturnWindowDataset, TrainStandardizer, WindowedReturns, collate_return_batch
    from diffusion_portfolio.evaluation import forecasts
    from torch.utils.data import DataLoader
    import pandas as pd

    cfg = replace(small_config(), evaluation=EvaluationConfig(n_scenarios=4))
    history = np.linspace(-.3, .3, 60).astype(np.float32).reshape(3, 4, 5)
    target = np.linspace(-.1, .1, 15).astype(np.float32).reshape(3, 1, 5)
    windows = WindowedReturns(history, target, pd.bdate_range("2000-01-03", periods=3))
    dataset = ReturnWindowDataset(windows, windows)
    scaler = TrainStandardizer.identity(tuple("abcde"))

    class ValidationOnly:
        val = dataset

        @property
        def test(self):
            raise AssertionError("Validation evaluation accessed the held-out split")

    monkeypatch.setattr(forecasts, "load_config", lambda path: cfg)
    monkeypatch.setattr(forecasts, "prepare_return_data", lambda *a, **k: (ValidationOnly(), scaler))
    model = build_diffusion(cfg, n_assets=5).eval()
    torch.save({"epoch": 1, "val_loss": .25, "model_state_dict": model.state_dict()}, tmp_path / "best.pt")

    # Reference the original validation loop: seed, iterate batches, sample,
    # then inverse-transform. DataLoader iteration itself consumes RNG.
    torch.manual_seed(cfg.training.validation_seed)
    np.random.seed(cfg.training.validation_seed)
    expected = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_return_batch):
            expected.append(scaler.inverse_transform(model.sample(batch["history"], n_scenarios=4).numpy()))
    expected_rng = torch.get_rng_state()
    result = forecasts.evaluate_diffusion(tmp_path, batch_size=2, device="cpu")
    with np.load(tmp_path / "validation_scenarios.npz") as saved:
        np.testing.assert_array_equal(saved["scenarios"], np.concatenate(expected))
        np.testing.assert_array_equal(saved["observed"], target[:, 0])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert result["sampling_seed"] == cfg.training.validation_seed
    assert result["split"] == "validation"
