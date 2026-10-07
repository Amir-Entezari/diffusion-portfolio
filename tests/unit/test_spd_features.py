import math

import torch

from diffusion_portfolio.models.spd import (
    covariance_features,
    log_euclidean_spd_features,
    regularized_covariance,
    symmetric_vectorize,
)


def test_regularized_covariance_is_spd():
    torch.manual_seed(
        42
    )

    history = torch.randn(
        4,
        60,
        12,
    )

    covariance = regularized_covariance(
        history
    )

    assert covariance.shape == (
        4,
        12,
        12,
    )

    eigenvalues = torch.linalg.eigvalsh(
        covariance
    )

    assert torch.all(
        eigenvalues > 0
    )


def test_symmetric_vectorize_preserves_frobenius_norm():
    torch.manual_seed(
        42
    )

    raw = torch.randn(
        3,
        6,
        6,
    )

    symmetric = 0.5 * (
        raw
        + raw.transpose(
            1,
            2,
        )
    )

    vector = symmetric_vectorize(
        symmetric
    )

    matrix_norm = torch.linalg.matrix_norm(
        symmetric,
        ord="fro",
    )

    vector_norm = torch.linalg.vector_norm(
        vector,
        dim=1,
    )

    torch.testing.assert_close(
        matrix_norm,
        vector_norm,
    )


def test_spd_feature_shapes():
    torch.manual_seed(
        42
    )

    history = torch.randn(
        5,
        60,
        12,
    )

    covariance = covariance_features(
        history
    )

    log_spd = (
        log_euclidean_spd_features(
            history
        )
    )

    assert covariance.shape == (
        5,
        78,
    )

    assert log_spd.shape == (
        5,
        78,
    )

    assert torch.isfinite(
        covariance
    ).all()

    assert torch.isfinite(
        log_spd
    ).all()


def test_log_spd_features_are_deterministic():
    torch.manual_seed(
        42
    )

    history = torch.randn(
        3,
        60,
        12,
    )

    first = (
        log_euclidean_spd_features(
            history
        )
    )

    second = (
        log_euclidean_spd_features(
            history
        )
    )

    torch.testing.assert_close(
        first,
        second,
    )