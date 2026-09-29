import torch
import pytest

from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
    HistoryEncoder,
)


def make_model(
    prediction_type="v_prediction",
    ):
    return ConditionalDiffusionModel(
        lookback=60,
        n_assets=12,
        condition_dim=16,
        history_hidden_dim=32,
        diffusion_steps=20,
        schedule_type="cosine",
        prediction_type=prediction_type,
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

    assert output.prediction.shape == (
        8,
        12,
    )

    assert output.training_target.shape == (
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
        output_a.prediction,
        output_b.prediction,
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
        flat.prediction,
        window.prediction,
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
        
        
def test_sampler_returns_expected_scenario_shape():
    model = make_model()

    history = torch.randn(
        3,
        60,
        12,
    )

    samples = model.sample(
        history,
        n_scenarios=5,
    )

    assert samples.shape == (
        3,
        5,
        12,
    )

    assert torch.isfinite(
        samples
    ).all()
    
    
def test_sampler_is_reproducible_with_fixed_seed():
    model = make_model()

    history = torch.randn(
        2,
        60,
        12,
    )

    torch.manual_seed(
        123
    )

    samples_a = model.sample(
        history,
        n_scenarios=3,
    )

    torch.manual_seed(
        123
    )

    samples_b = model.sample(
        history,
        n_scenarios=3,
    )

    torch.testing.assert_close(
        samples_a,
        samples_b,
    )
    
    
def test_sampler_rejects_wrong_initial_noise_shape():
    model = make_model()

    history = torch.randn(
        2,
        60,
        12,
    )

    wrong_noise = torch.randn(
        2,
        4,
        11,
    )

    with pytest.raises(
        ValueError,
        match="initial_noise",
    ):
        model.sample(
            history,
            n_scenarios=4,
            initial_noise=wrong_noise,
        )
        
        
def test_reverse_step_at_zero_is_deterministic():
    model = make_model()

    history = torch.randn(
        4,
        60,
        12,
    )

    condition = (
        model.history_encoder(
            history
        )
    )

    x_t = torch.randn(
        4,
        12,
    )

    timesteps = torch.zeros(
        4,
        dtype=torch.long,
    )

    noise_a = torch.randn(
        4,
        12,
    )

    noise_b = torch.randn(
        4,
        12,
    )

    result_a = model.reverse_step(
        x_t,
        timesteps,
        condition,
        noise=noise_a,
    )

    result_b = model.reverse_step(
        x_t,
        timesteps,
        condition,
        noise=noise_b,
    )

    torch.testing.assert_close(
        result_a,
        result_b,
    )
    
def test_sampler_depends_on_history_condition():
    torch.manual_seed(
        42
    )

    model = make_model()

    history_a = torch.zeros(
        1,
        60,
        12,
    )

    history_b = torch.ones(
        1,
        60,
        12,
    )

    initial_noise = torch.randn(
        1,
        2,
        12,
    )

    torch.manual_seed(
        123
    )

    samples_a = model.sample(
        history_a,
        n_scenarios=2,
        initial_noise=initial_noise,
    )

    torch.manual_seed(
        123
    )

    samples_b = model.sample(
        history_b,
        n_scenarios=2,
        initial_noise=initial_noise,
    )

    assert not torch.allclose(
        samples_a,
        samples_b,
    )
    
    
def test_v_prediction_training_target_matches_definition():
    model = make_model(
        prediction_type="v_prediction"
    )

    history = torch.randn(
        4,
        60,
        12,
    )

    target = torch.randn(
        4,
        12,
    )

    noise = torch.randn(
        4,
        12,
    )

    timesteps = torch.tensor(
        [
            0,
            5,
            10,
            19,
        ]
    )

    output = model.training_loss(
        history,
        target,
        noise=noise,
        timesteps=timesteps,
    )

    coefficients = (
        model.noise_schedule
        .get_coefficients(
            timesteps
        )
    )

    sqrt_alpha_bar = (
        coefficients[
            "sqrt_alpha_cumprod"
        ].unsqueeze(-1)
    )

    sqrt_one_minus = (
        coefficients[
            "sqrt_one_minus_alpha_cumprod"
        ].unsqueeze(-1)
    )

    expected = (
        sqrt_alpha_bar
        * noise
        - sqrt_one_minus
        * target
    )

    torch.testing.assert_close(
        output.training_target,
        expected,
    )


def test_epsilon_prediction_remains_supported():
    model = make_model(
        prediction_type="epsilon"
    )

    history = torch.randn(
        4,
        60,
        12,
    )

    target = torch.randn(
        4,
        12,
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

    output = model.training_loss(
        history,
        target,
        noise=noise,
        timesteps=timesteps,
    )

    torch.testing.assert_close(
        output.training_target,
        noise,
    )


def test_invalid_prediction_type_is_rejected():
    with pytest.raises(
        ValueError,
        match="prediction_type",
    ):
        ConditionalDiffusionModel(
            lookback=60,
            n_assets=12,
            condition_dim=16,
            history_hidden_dim=32,
            diffusion_steps=20,
            schedule_type="cosine",
            prediction_type="magic",
            channels=[
                16,
                32,
            ],
            time_embed_dim=16,
            n_res_blocks=1,
        )