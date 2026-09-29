import torch
import torch.nn.functional as F

from diffusion_portfolio.models.diffolio import (
    DiffolioObjective,
)


def make_objective(
    *,
    lambda_corr: float = 0.05,
):
    torch.manual_seed(
        123
    )

    training_returns = torch.randn(
        200,
        3,
    )

    centered = (
        training_returns
        - training_returns.mean(
            dim=0,
            keepdim=True,
        )
    )

    training_covariance = (
        centered.T
        @ centered
    ) / training_returns.shape[0]

    return DiffolioObjective(
        training_covariance=(
            training_covariance
        ),
        n_assets=3,
        n_asset_characteristics=2,
        n_systematic=2,
        lookback=8,
        hidden_dim=16,
        num_heads=4,
        mlp_dim=32,
        time_embedding_dim=8,
        diffusion_steps=10,
        beta_start=1e-4,
        beta_end=0.02,
        lambda_corr=lambda_corr,
    )


def make_batch(
    batch_size: int = 4,
):
    torch.manual_seed(
        456
    )

    target = torch.randn(
        batch_size,
        3,
    )

    return_history = torch.randn(
        batch_size,
        8,
        3,
    )

    # Raw history is deliberately on a much smaller
    # financial-return-like scale.
    return_history_raw = (
        torch.randn(
            batch_size,
            8,
            3,
        )
        * 0.01
    )

    asset_covariates = torch.randn(
        batch_size,
        8,
        3,
        2,
    )

    systematic_covariates = torch.randn(
        batch_size,
        8,
        2,
    )

    return (
        target,
        return_history,
        return_history_raw,
        asset_covariates,
        systematic_covariates,
    )


def test_objective_returns_expected_shapes_and_finite_losses():
    model = make_objective()

    output = model.training_loss(
        *make_batch()
    )

    assert output.prediction.shape == (
        4,
        3,
    )

    assert output.noise.shape == (
        4,
        3,
    )

    assert output.noisy_target.shape == (
        4,
        3,
    )

    assert output.target_correlation.shape == (
        4,
        3,
        3,
    )

    assert output.shrinkage.shape == (
        4,
    )

    assert torch.isfinite(
        output.loss
    )

    assert torch.isfinite(
        output.ddpm_loss
    )

    assert torch.isfinite(
        output.correlation_loss
    )


def test_total_loss_is_exact_weighted_sum():
    model = make_objective(
        lambda_corr=0.05
    )

    output = model.training_loss(
        *make_batch()
    )

    expected = (
        output.ddpm_loss
        + 0.05
        * output.correlation_loss
    )

    torch.testing.assert_close(
        output.loss,
        expected,
    )


def test_fixed_noise_and_timesteps_use_standard_forward_diffusion():
    model = make_objective()

    (
        target,
        return_history,
        return_history_raw,
        asset_covariates,
        systematic_covariates,
    ) = make_batch(
        batch_size=2
    )

    noise = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [-0.4, 0.5, -0.6],
        ],
        dtype=target.dtype,
    )

    timesteps = torch.tensor(
        [
            0,
            9,
        ]
    )

    output = model.training_loss(
        target,
        return_history,
        return_history_raw,
        asset_covariates,
        systematic_covariates,
        noise=noise,
        timesteps=timesteps,
    )

    expected_noisy = (
        model.noise_schedule.q_sample(
            target,
            timesteps,
            noise,
        )
    )

    torch.testing.assert_close(
        output.noisy_target,
        expected_noisy,
    )

    torch.testing.assert_close(
        output.noise,
        noise,
    )

    # Internal indices 0 and 9 correspond to
    # paper timesteps 1 and 10.
    torch.testing.assert_close(
        output.diffusion_timesteps,
        torch.tensor(
            [
                1,
                10,
            ]
        ),
    )

    expected_mse = F.mse_loss(
        output.prediction,
        noise,
    )

    torch.testing.assert_close(
        output.ddpm_loss,
        expected_mse,
    )


def test_zero_correlation_weight_reduces_to_ddpm_loss():
    model = make_objective(
        lambda_corr=0.0
    )

    output = model.training_loss(
        *make_batch()
    )

    torch.testing.assert_close(
        output.loss,
        output.ddpm_loss,
    )


def test_complete_objective_backpropagates_through_denoiser():
    model = make_objective()

    output = model.training_loss(
        *make_batch(
            batch_size=2
        )
    )

    output.loss.backward()

    decoder_grad = (
        model.denoiser.decoder.weight.grad
    )

    market_q_grad = (
        model.denoiser
        .hierarchy
        .market_attention
        .attention
        .in_proj_weight
        .grad
    )

    assert decoder_grad is not None
    assert market_q_grad is not None

    assert torch.isfinite(
        decoder_grad
    ).all()

    assert torch.isfinite(
        market_q_grad
    ).all()

    # Correlation targets themselves are detached supervision.
    assert not output.target_correlation.requires_grad