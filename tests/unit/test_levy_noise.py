import math

import pytest
import torch

from diffusion_portfolio.models.levy import (
    sample_isotropic_alpha_stable,
    sample_positive_stable_mixer,
)


def test_positive_stable_mixer_is_positive_and_finite():
    generator = torch.Generator()
    generator.manual_seed(
        42
    )

    mixer = (
        sample_positive_stable_mixer(
            alpha=1.7,
            shape=(
                4096,
                1,
            ),
            dtype=torch.float64,
            generator=generator,
        )
    )

    assert mixer.shape == (
        4096,
        1,
    )

    assert torch.isfinite(
        mixer
    ).all()

    assert torch.all(
        mixer > 0
    )


def test_alpha_two_mixer_is_exactly_two():
    mixer = (
        sample_positive_stable_mixer(
            alpha=2.0,
            shape=(
                16,
                1,
            ),
            dtype=torch.float64,
        )
    )

    torch.testing.assert_close(
        mixer,
        torch.full_like(
            mixer,
            2.0,
        ),
    )


def test_alpha_two_matches_sqrt_two_gaussian():
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
        sample_isotropic_alpha_stable(
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

    expected = (
        math.sqrt(
            2.0
        )
        * torch.randn(
            shape,
            dtype=torch.float64,
            generator=generator_b,
        )
    )

    torch.testing.assert_close(
        samples,
        expected,
    )


def test_isotropic_stable_matches_characteristic_function():
    """Empirically verify the DLPM stable-law normalization."""

    generator = torch.Generator()

    generator.manual_seed(
        12345
    )

    alpha = 1.7

    samples = (
        sample_isotropic_alpha_stable(
            alpha=alpha,
            shape=(
                50_000,
                3,
            ),
            dtype=torch.float64,
            generator=generator,
        )
    )

    direction = torch.tensor(
        [
            0.7,
            -0.2,
            0.4,
        ],
        dtype=torch.float64,
    )

    projection = (
        samples
        @ direction
    )

    empirical_real = (
        torch.cos(
            projection
        )
        .mean()
        .item()
    )

    empirical_imaginary = (
        torch.sin(
            projection
        )
        .mean()
        .item()
    )

    expected_real = math.exp(
        -(
            torch.linalg.vector_norm(
                direction
            ).item()
            ** alpha
        )
    )

    assert empirical_real == (
        pytest.approx(
            expected_real,
            abs=0.015,
        )
    )

    assert empirical_imaginary == (
        pytest.approx(
            0.0,
            abs=0.015,
        )
    )


@pytest.mark.parametrize(
    "alpha",
    [
        0.0,
        1.0,
        2.1,
    ],
)
def test_invalid_alpha_is_rejected(
    alpha,
):
    with pytest.raises(
        ValueError,
        match="1 < alpha <= 2",
    ):
        sample_isotropic_alpha_stable(
            alpha=alpha,
            shape=(
                4,
                12,
            ),
        )