import pytest
import torch

from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)
from diffusion_portfolio.models.levy import (
    ConditionalLevyDiffusionModel,
)


def make_gaussian_epsilon_model():
    return ConditionalDiffusionModel(
        lookback=5,
        n_assets=4,
        condition_dim=8,
        history_hidden_dim=16,
        diffusion_steps=10,
        schedule_type="cosine",
        channels=[
            8,
            16,
        ],
        prediction_type="epsilon",
        time_embed_dim=8,
        n_res_blocks=1,
    )


def make_levy_model(
    *,
    alpha: float,
):
    return ConditionalLevyDiffusionModel(
        alpha=alpha,
        lookback=5,
        n_assets=4,
        condition_dim=8,
        history_hidden_dim=16,
        diffusion_steps=10,
        schedule_type="cosine",
        channels=[
            8,
            16,
        ],
        time_embed_dim=8,
        n_res_blocks=1,
    )


def test_alpha_two_has_same_parameter_structure_as_gaussian():
    gaussian = (
        make_gaussian_epsilon_model()
    )

    levy = make_levy_model(
        alpha=2.0
    )


    # Schedule buffers differ by name, so compare
    # trainable parameter structure only.
    gaussian_parameters = {
        name
        for name, _
        in gaussian.named_parameters()
    }

    levy_parameters = {
        name
        for name, _
        in levy.named_parameters()
    }

    assert gaussian_parameters == (
        levy_parameters
    )


def test_alpha_two_training_loss_matches_gaussian_epsilon():
    torch.manual_seed(
        42
    )

    gaussian = (
        make_gaussian_epsilon_model()
    )

    levy = make_levy_model(
        alpha=2.0
    )

    # Copy all learned modules exactly.
    levy.history_encoder.load_state_dict(
        gaussian.history_encoder.state_dict()
    )

    levy.score_network.load_state_dict(
        gaussian.score_network.state_dict()
    )

    history = torch.randn(
        6,
        5,
        4,
    )

    target = torch.randn(
        6,
        1,
        4,
    )

    timesteps = torch.tensor(
        [
            0,
            1,
            3,
            5,
            7,
            9,
        ]
    )

    noise = torch.randn(
        6,
        4,
    )

    gaussian_output = (
        gaussian.training_loss(
            history,
            target,
            noise=noise,
            timesteps=timesteps,
        )
    )

    levy_output = (
        levy.training_loss(
            history,
            target,
            noise=noise,
            timesteps=timesteps,
        )
    )

    torch.testing.assert_close(
        levy_output.noisy_target,
        gaussian_output.noisy_target,
        atol=1e-7,
        rtol=1e-6,
    )

    torch.testing.assert_close(
        levy_output.training_target,
        gaussian_output.training_target,
        atol=0.0,
        rtol=0.0,
    )

    torch.testing.assert_close(
        levy_output.prediction,
        gaussian_output.prediction,
        atol=1e-7,
        rtol=1e-6,
    )

    torch.testing.assert_close(
        levy_output.loss,
        gaussian_output.loss,
        atol=1e-7,
        rtol=1e-6,
    )


def test_heavy_tail_training_loss_is_finite():
    torch.manual_seed(
        123
    )

    model = make_levy_model(
        alpha=1.7
    )

    history = torch.randn(
        16,
        5,
        4,
    )

    target = torch.randn(
        16,
        1,
        4,
    )

    output = model.training_loss(
        history,
        target,
    )

    assert torch.isfinite(
        output.loss
    )

    assert torch.isfinite(
        output.noisy_target
    ).all()

    assert torch.isfinite(
        output.training_target
    ).all()

    assert output.prediction.shape == (
        16,
        4,
    )


def test_levy_sampling_is_blocked_until_reverse_is_implemented():
    model = make_levy_model(
        alpha=1.7
    )

    condition = torch.randn(
        2,
        8,
    )

    with pytest.raises(
        NotImplementedError,
        match="reverse sampling",
    ):
        model.sample_from_condition(
            condition,
            n_scenarios=4,
        )