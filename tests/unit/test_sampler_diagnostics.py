"""Keep diagnostics faithful to current prediction targets and transitions."""
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from diffusion_portfolio.config import MVPConfig, DataConfig, ModelConfig, TrainingConfig, PortfolioConfig, EvaluationConfig
from diffusion_portfolio.evaluation.sampler_diagnostics import diagnose_reverse_sampler
from diffusion_portfolio.models.diffusion.build import build_diffusion


def make_model(n_assets, prediction_type):
    cfg = MVPConfig(
        seed=41, data=DataConfig(lookback=4),
        model=ModelConfig(condition_dim=8, history_hidden_dim=16, diffusion_steps=5,
                          channels=(8,), time_embed_dim=8, n_res_blocks=1,
                          prediction_type=prediction_type),
        training=TrainingConfig(), portfolio=PortfolioConfig(), evaluation=EvaluationConfig(),
    )
    return build_diffusion(cfg, n_assets=n_assets)


class ZeroPrediction(nn.Module):
    def forward(self, x, timesteps, condition):
        return torch.zeros_like(x)


@pytest.mark.parametrize("n_assets", [3, 17])
@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_diagnostics_preserve_prediction_semantics_and_reverse_chain(n_assets, prediction_type):
    model = make_model(n_assets, prediction_type)
    model.score_network = ZeroPrediction()
    histories = torch.linspace(-.2, .2, 8 * n_assets).reshape(2, 4, n_assets)
    targets = torch.linspace(-.1, .1, 2 * n_assets).reshape(2, n_assets)
    rng_before = torch.get_rng_state()
    report = diagnose_reverse_sampler(model, histories, targets, n_repeats=2, n_scenarios=4)
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert report == diagnose_reverse_sampler(model, histories, targets, n_repeats=2, n_scenarios=4)
    generator = torch.Generator().manual_seed(777)
    for row in report["timestep_errors"]:
        step = row["step"]
        x0 = targets.repeat_interleave(2, dim=0)
        epsilon = torch.randn(x0.shape, generator=generator)
        t = torch.full((len(x0),), step, dtype=torch.long)
        a = model.noise_schedule.sqrt_alphas_cumprod[step]
        s = model.noise_schedule.sqrt_one_minus_alphas_cumprod[step]
        epsilon_hat = torch.zeros_like(epsilon) if prediction_type == "epsilon" else s * (a * x0 + s * epsilon)
        assert row["epsilon_mse"] == ((epsilon_hat - epsilon) ** 2).mean().item()
        training_target = epsilon if prediction_type == "epsilon" else a * epsilon - s * x0
        assert row["prediction_mse"] == (training_target ** 2).mean().item()

    # Replay the original diagnostic's explicit-noise chain through the public
    # reverse transition, rather than trusting a second sampler implementation.
    model.eval()
    with torch.no_grad():
        condition = model._encode_condition(histories[:1]).repeat_interleave(4, dim=0)
        initial = torch.randn((4, n_assets), generator=torch.Generator().manual_seed(1234))
        for stochastic, name in [(True, "stochastic"), (False, "zero_posterior_noise")]:
            x = initial.clone()
            generator = torch.Generator().manual_seed(5678)
            expected = []
            watched = {row["step"] for row in report["timestep_errors"]}
            for step in reversed(range(model.diffusion_steps)):
                noise = torch.randn(x.shape, generator=generator) if stochastic and step else torch.zeros_like(x)
                x = model.reverse_step(x, torch.full((4,), step, dtype=torch.long), condition, noise=noise)
                if step in watched:
                    expected.append(dict(step=step, mean=x.mean().item(), std=x.std().item(),
                                         max_abs=x.abs().max().item(), finite=True))
            assert report["reverse_chains"][name] == expected


def test_diagnostics_reject_out_of_schedule_timesteps():
    model = make_model(3, "epsilon")
    with pytest.raises(ValueError, match="timesteps"):
        diagnose_reverse_sampler(model, torch.zeros(2, 4, 3), torch.zeros(2, 3), timesteps=[5])


def test_diagnostic_workflow_selects_training_only(tmp_path, monkeypatch):
    from diffusion_portfolio import config
    from diffusion_portfolio.data import preparation, TrainStandardizer
    from diffusion_portfolio.models.diffusion import build
    from diffusion_portfolio.evaluation.sampler_diagnostics import run_sampler_diagnostics

    class TrainingOnly:
        train = [{"history": torch.zeros(4, 3), "target": torch.zeros(1, 3)} for _ in range(3)]

        @property
        def val(self):
            raise AssertionError("Training diagnostic accessed validation")

        @property
        def test(self):
            raise AssertionError("Training diagnostic accessed the held-out split")

    model = make_model(3, "v_prediction")
    checkpoint = tmp_path / "best.pt"
    torch.save({"epoch": 2, "val_loss": .1, "model_state_dict": model.state_dict()}, checkpoint)
    monkeypatch.setattr(config, "load_config", lambda path: SimpleNamespace(seed=41))
    monkeypatch.setattr(preparation, "prepare_return_data", lambda *a, **k: (TrainingOnly(), TrainStandardizer.identity(("a", "b", "c"))))
    monkeypatch.setattr(build, "build_diffusion", lambda cfg, *, n_assets: model if n_assets == 3 else pytest.fail("Wrong asset count"))
    report = run_sampler_diagnostics("unused.yaml", checkpoint, device="cpu", n_overfit_samples=2, n_repeats=1, n_scenarios=4)
    assert report["split"] == "train"
    assert report["checkpoint_epoch"] == 2
