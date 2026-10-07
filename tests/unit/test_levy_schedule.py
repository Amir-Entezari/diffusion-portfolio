import torch

from diffusion_portfolio.models.diffusion.schedule import (
    NoiseSchedule,
)
from diffusion_portfolio.models.levy import (
    LevyNoiseSchedule,
    sample_ddpm_normalized_alpha_stable,
)


def test_ddpm_normalized_alpha_two_is_standard_gaussian():
    shape = (
        32,
        12,
    )

    generator_a = (
        torch.Generator()
    )

    generator_a.manual_seed(
        123
    )

    samples = (
        sample_ddpm_normalized_alpha_stable(
            alpha=2.0,
            shape=shape,
            dtype=torch.float64,
            generator=generator_a,
        )
    )

    generator_b = (
        torch.Generator()
    )

    generator_b.manual_seed(
        123
    )

    expected = torch.randn(
        shape,
        dtype=torch.float64,
        generator=generator_b,
    )

    torch.testing.assert_close(
        samples,
        expected,
    )


def test_levy_schedule_is_scale_preserving():
    alpha = 1.7

    schedule = LevyNoiseSchedule(
        alpha=alpha,
        n_steps=100,
        schedule_type="cosine",
    )

    step_total = (
        schedule.gammas.pow(
            alpha
        )
        + schedule.sigmas.pow(
            alpha
        )
    )

    cumulative_total = (
        schedule.bar_gammas.pow(
            alpha
        )
        + schedule.bar_sigmas.pow(
            alpha
        )
    )

    torch.testing.assert_close(
        step_total,
        torch.ones_like(
            step_total
        ),
        atol=1e-6,
        rtol=1e-6,
    )

    torch.testing.assert_close(
        cumulative_total,
        torch.ones_like(
            cumulative_total
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_alpha_two_schedule_matches_gaussian_ddpm():
    gaussian = NoiseSchedule(
        n_steps=100,
        schedule_type="cosine",
    )

    levy = LevyNoiseSchedule(
        alpha=2.0,
        n_steps=100,
        schedule_type="cosine",
    )

    torch.testing.assert_close(
        levy.bar_gammas,
        gaussian.sqrt_alphas_cumprod,
        atol=1e-7,
        rtol=1e-6,
    )

    torch.testing.assert_close(
        levy.bar_sigmas,
        gaussian.sqrt_one_minus_alphas_cumprod,
        atol=1e-7,
        rtol=1e-6,
    )


def test_alpha_two_forward_marginal_matches_gaussian_ddpm():
    torch.manual_seed(
        42
    )

    x_0 = torch.randn(
        8,
        12,
    )

    noise = torch.randn(
        8,
        12,
    )

    timesteps = torch.tensor(
        [
            0,
            3,
            7,
            12,
            25,
            50,
            75,
            99,
        ]
    )

    gaussian = NoiseSchedule(
        n_steps=100,
        schedule_type="cosine",
    )

    levy = LevyNoiseSchedule(
        alpha=2.0,
        n_steps=100,
        schedule_type="cosine",
    )

    gaussian_sample = (
        gaussian.q_sample(
            x_0,
            timesteps,
            noise,
        )
    )

    levy_sample = levy.q_sample(
        x_0,
        timesteps,
        noise,
    )

    torch.testing.assert_close(
        levy_sample,
        gaussian_sample,
        atol=1e-7,
        rtol=1e-6,
    )


def test_heavy_tailed_forward_marginal_is_finite():
    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        123
    )

    x_0 = torch.zeros(
        64,
        12,
        dtype=torch.float64,
    )

    noise = (
        sample_ddpm_normalized_alpha_stable(
            alpha=1.7,
            shape=tuple(
                x_0.shape
            ),
            dtype=torch.float64,
            generator=generator,
        )
    )

    timesteps = torch.full(
        (
            len(
                x_0
            ),
        ),
        50,
        dtype=torch.long,
    )

    schedule = LevyNoiseSchedule(
        alpha=1.7,
        n_steps=100,
        schedule_type="cosine",
    ).to(
        dtype=torch.float64
    )

    x_t = schedule.q_sample(
        x_0,
        timesteps,
        noise,
    )

    assert x_t.shape == (
        64,
        12,
    )

    assert torch.isfinite(
        x_t
    ).all()


def test_invalid_levy_alpha_is_rejected():
    try:
        LevyNoiseSchedule(
            alpha=1.0,
            n_steps=100,
        )

    except ValueError as error:
        assert (
            "1 < alpha <= 2"
            in str(
                error
            )
        )

    else:
        raise AssertionError(
            "invalid alpha was not rejected"
        )