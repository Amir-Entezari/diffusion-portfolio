import torch

from diffusion_portfolio.baselines.diffolio import (
    DiffolioObjective,
    make_ddim_timesteps,
    sample_diffolio_ddim,
)


def make_model():
    torch.manual_seed(
        123
    )

    training_returns = torch.randn(
        100,
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
        lambda_corr=0.05,
    )


def make_conditioning(
    batch_size: int = 2,
):
    torch.manual_seed(
        456
    )

    return_history = torch.randn(
        batch_size,
        8,
        3,
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
        return_history,
        asset_covariates,
        systematic_covariates,
    )


def test_ddim_timestep_schedule_is_descending_and_complete():
    timesteps = make_ddim_timesteps(
        diffusion_steps=10,
        sampling_steps=5,
    )

    torch.testing.assert_close(
        timesteps,
        torch.tensor(
            [
                9,
                7,
                4,
                2,
                0,
            ]
        ),
    )


def test_ddim_sampler_output_shape_and_finiteness():
    model = make_model()

    conditioning = make_conditioning(
        batch_size=2
    )

    scenarios = sample_diffolio_ddim(
        model,
        *conditioning,
        n_scenarios=4,
        sampling_steps=5,
    )

    assert scenarios.shape == (
        2,
        4,
        3,
    )

    assert torch.isfinite(
        scenarios
    ).all()


def test_ddim_is_deterministic_given_initial_noise():
    model = make_model()

    conditioning = make_conditioning(
        batch_size=2
    )

    torch.manual_seed(
        789
    )

    initial_noise = torch.randn(
        2,
        4,
        3,
    )

    samples_a = sample_diffolio_ddim(
        model,
        *conditioning,
        n_scenarios=4,
        sampling_steps=5,
        initial_noise=initial_noise,
    )

    samples_b = sample_diffolio_ddim(
        model,
        *conditioning,
        n_scenarios=4,
        sampling_steps=5,
        initial_noise=initial_noise,
    )

    torch.testing.assert_close(
        samples_a,
        samples_b,
        atol=0.0,
        rtol=0.0,
    )


def test_different_initial_noise_produces_different_scenarios():
    model = make_model()

    conditioning = make_conditioning(
        batch_size=1
    )

    initial_a = torch.zeros(
        1,
        2,
        3,
    )

    initial_b = torch.ones(
        1,
        2,
        3,
    )

    samples_a = sample_diffolio_ddim(
        model,
        *conditioning,
        n_scenarios=2,
        sampling_steps=5,
        initial_noise=initial_a,
    )

    samples_b = sample_diffolio_ddim(
        model,
        *conditioning,
        n_scenarios=2,
        sampling_steps=5,
        initial_noise=initial_b,
    )

    assert not torch.equal(
        samples_a,
        samples_b,
    )


def test_sampler_restores_training_mode():
    model = make_model()

    model.train()

    conditioning = make_conditioning(
        batch_size=1
    )

    sample_diffolio_ddim(
        model,
        *conditioning,
        n_scenarios=2,
        sampling_steps=5,
    )

    assert model.training


def test_full_schedule_sampling_is_supported():
    model = make_model()

    conditioning = make_conditioning(
        batch_size=1
    )

    samples = sample_diffolio_ddim(
        model,
        *conditioning,
        n_scenarios=2,
        sampling_steps=10,
    )

    assert samples.shape == (
        1,
        2,
        3,
    )

    assert torch.isfinite(
        samples
    ).all()