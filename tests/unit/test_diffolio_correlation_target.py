import torch

from diffusion_portfolio.models.diffolio.correlation_target import (
    compute_training_covariance,
    covariance_to_correlation,
    estimate_shrinkage_correlation,
)


def test_training_covariance_matches_manual_calculation():
    returns = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]
    )

    result = compute_training_covariance(
        returns
    )

    centered = (
        returns
        - returns.mean(
            dim=0,
            keepdim=True,
        )
    )

    expected = (
        centered.T
        @ centered
    ) / returns.shape[0]

    torch.testing.assert_close(
        result,
        expected,
    )


def test_covariance_to_correlation_has_unit_diagonal():
    covariance = torch.tensor(
        [
            [
                [4.0, 1.0],
                [1.0, 9.0],
            ],
            [
                [1.0, -0.5],
                [-0.5, 1.0],
            ],
        ]
    )

    correlation = (
        covariance_to_correlation(
            covariance
        )
    )

    assert correlation.shape == (
        2,
        2,
        2,
    )

    diagonal = torch.diagonal(
        correlation,
        dim1=-2,
        dim2=-1,
    )

    torch.testing.assert_close(
        diagonal,
        torch.ones_like(
            diagonal
        ),
    )

    assert torch.isfinite(
        correlation
    ).all()


def test_shrinkage_target_shapes_and_bounds():
    torch.manual_seed(
        123
    )

    training_returns = torch.randn(
        500,
        4,
    )

    training_covariance = (
        compute_training_covariance(
            training_returns
        )
    )

    recent_returns = torch.randn(
        8,
        63,
        4,
    )

    result = (
        estimate_shrinkage_correlation(
            recent_returns,
            training_covariance,
        )
    )

    assert result.covariance.shape == (
        8,
        4,
        4,
    )

    assert result.correlation.shape == (
        8,
        4,
        4,
    )

    assert result.shrinkage.shape == (
        8,
    )

    assert torch.all(
        result.shrinkage >= 0.0
    )

    assert torch.all(
        result.shrinkage <= 1.0
    )

    assert torch.isfinite(
        result.covariance
    ).all()

    assert torch.isfinite(
        result.correlation
    ).all()


def test_shrinkage_moves_covariance_toward_training_target():
    torch.manual_seed(
        123
    )

    training_returns = torch.randn(
        1000,
        3,
    )

    training_covariance = (
        compute_training_covariance(
            training_returns
        )
    )

    recent_returns = (
        torch.randn(
            5,
            63,
            3,
        )
        * torch.tensor(
            [
                1.0,
                3.0,
                0.5,
            ]
        )
    )

    result = (
        estimate_shrinkage_correlation(
            recent_returns,
            training_covariance,
        )
    )

    centered = (
        recent_returns
        - recent_returns.mean(
            dim=1,
            keepdim=True,
        )
    )

    sample_covariance = torch.einsum(
        "bti,btj->bij",
        centered,
        centered,
    ) / recent_returns.shape[1]

    target = training_covariance[
        None,
        :,
        :,
    ]

    sample_distance = (
        (
            sample_covariance
            - target
        )
        .square()
        .sum(
            dim=(-2, -1)
        )
    )

    shrunk_distance = (
        (
            result.covariance
            - target
        )
        .square()
        .sum(
            dim=(-2, -1)
        )
    )

    assert torch.all(
        shrunk_distance
        <= sample_distance
        + 1e-6
    )


def test_batch_samples_are_independent():
    torch.manual_seed(
        123
    )

    training_returns = torch.randn(
        500,
        3,
    )

    training_covariance = (
        compute_training_covariance(
            training_returns
        )
    )

    recent = torch.randn(
        2,
        63,
        3,
    )

    original = (
        estimate_shrinkage_correlation(
            recent,
            training_covariance,
        )
    )

    changed = recent.clone()

    # Alter only batch item 1.
    changed[
        1
    ] *= 100.0

    recomputed = (
        estimate_shrinkage_correlation(
            changed,
            training_covariance,
        )
    )

    torch.testing.assert_close(
        original.correlation[
            0
        ],
        recomputed.correlation[
            0
        ],
        atol=0.0,
        rtol=0.0,
    )

    torch.testing.assert_close(
        original.shrinkage[
            0
        ],
        recomputed.shrinkage[
            0
        ],
        atol=0.0,
        rtol=0.0,
    )