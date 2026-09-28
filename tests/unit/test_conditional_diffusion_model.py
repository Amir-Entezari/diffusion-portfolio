import torch
import pytest

from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
    HistoryEncoder,
)


def make_model():
    return ConditionalDiffusionModel(
        lookback=60,
        n_assets=12,
        condition_dim=16,
        history_hidden_dim=32,
        diffusion_steps=20,
        schedule_type="cosine",
        channels=[
            16,
            32,
        ],
        time_embed_dim=16,
        n_res_blocks=1,
    )


def test_history_encoder_output_shape():
    encoder = HistoryEncoder(
        lookback=60,
        n_assets=12,
        condition_dim=16,
        hidden_dim=32,
    )

    history = torch.randn(
        8,
        60,
        12,
    )

    condition = encoder(
        history
    )

    assert condition.shape == (
        8,
        16,
    )

    assert torch.isfinite(
        condition
    ).all()


def test_training_loss_is_scalar_and_finite():
    model = make_model()

    history = torch.randn(
        8,
        60,
        12,
    )

    target = torch.randn(
        8,
        1,
        12,
    )

    output = model.training_loss(
        history,
        target,
    )

    assert output.loss.ndim == 0

    assert torch.isfinite(
        output.loss
    )

    assert output.predicted_noise.shape == (
        8,
        12,
    )

    assert output.target_noise.shape == (
        8,
        12,
    )


def test_fixed_noise_and_timesteps_are_deterministic():
    torch.manual_seed(
        42
    )

    model = make_model()

    history = torch.randn(
        4,
        60,
        12,
    )

    target = torch.randn(
        4,
        1,
        12,
    )

    noise = torch.randn(
        4,
        12,
    )

    timesteps = torch.tensor(
        [
            0,
            3,
            7,
            19,
        ]
    )

    output_a = model.training_loss(
        history,
        target,
        noise=noise,
        timesteps=timesteps,
    )

    output_b = model.training_loss(
        history,
        target,
        noise=noise,
        timesteps=timesteps,
    )

    torch.testing.assert_close(
        output_a.loss,
        output_b.loss,
    )

    torch.testing.assert_close(
        output_a.predicted_noise,
        output_b.predicted_noise,
    )

    torch.testing.assert_close(
        output_a.noisy_target,
        output_b.noisy_target,
    )


def test_gradients_reach_condition_encoder_and_denoiser():
    model = make_model()

    history = torch.randn(
        8,
        60,
        12,
    )

    target = torch.randn(
        8,
        1,
        12,
    )

    output = model.training_loss(
        history,
        target,
    )

    output.loss.backward()

    encoder_grads = [
        parameter.grad
        for parameter
        in model.history_encoder.parameters()
        if parameter.grad is not None
    ]

    denoiser_grads = [
        parameter.grad
        for parameter
        in model.score_network.parameters()
        if parameter.grad is not None
    ]

    assert encoder_grads

    assert denoiser_grads

    assert all(
        torch.isfinite(grad).all()
        for grad in encoder_grads
    )

    assert all(
        torch.isfinite(grad).all()
        for grad in denoiser_grads
    )


def test_flat_and_horizon_one_targets_are_equivalent():
    model = make_model()

    history = torch.randn(
        4,
        60,
        12,
    )

    flat_target = torch.randn(
        4,
        12,
    )

    window_target = (
        flat_target.unsqueeze(1)
    )

    noise = torch.randn(
        4,
        12,
    )

    timesteps = torch.tensor(
        [
            1,
            2,
            3,
            4,
        ]
    )

    flat = model.training_loss(
        history,
        flat_target,
        noise=noise,
        timesteps=timesteps,
    )

    window = model.training_loss(
        history,
        window_target,
        noise=noise,
        timesteps=timesteps,
    )

    torch.testing.assert_close(
        flat.loss,
        window.loss,
    )

    torch.testing.assert_close(
        flat.predicted_noise,
        window.predicted_noise,
    )


def test_multistep_target_is_rejected():
    model = make_model()

    history = torch.randn(
        4,
        60,
        12,
    )

    target = torch.randn(
        4,
        2,
        12,
    )

    with pytest.raises(
        ValueError,
        match="horizon=1",
    ):
        model.training_loss(
            history,
            target,
        )