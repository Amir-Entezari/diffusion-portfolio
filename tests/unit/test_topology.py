import pytest
import torch

from diffusion_portfolio.models.topology import (
    trajectory_geometry_features,
    trajectory_persistence_features,
)


def test_persistence_features_have_expected_shape():
    torch.manual_seed(
        42
    )

    trajectory = torch.randn(
        3,
        10,
        8,
    )

    features = (
        trajectory_persistence_features(
            trajectory,
            max_homology_dim=1,
            top_k=3,
        )
    )

    assert features.shape == (
        3,
        16,
    )

    assert torch.isfinite(
        features
    ).all()


def test_persistence_features_are_deterministic():
    trajectory = torch.randn(
        2,
        8,
        4,
    )

    first = trajectory_persistence_features(
        trajectory,
    )

    second = trajectory_persistence_features(
        trajectory,
    )

    torch.testing.assert_close(
        first,
        second,
    )


def test_wrong_trajectory_shape_is_rejected():
    trajectory = torch.randn(
        8,
        4,
    )

    with pytest.raises(
        ValueError,
        match="batch, time, hidden_dim",
    ):
        trajectory_persistence_features(
            trajectory
        )
        
        
def test_geometry_features_have_expected_shape():
    torch.manual_seed(
        42
    )

    trajectory = torch.randn(
        3,
        10,
        8,
    )

    features = trajectory_geometry_features(
        trajectory
    )

    assert features.shape == (
        3,
        7,
    )

    assert torch.isfinite(
        features
    ).all()