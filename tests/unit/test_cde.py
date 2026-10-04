import pytest
import torch

from diffusion_portfolio.models.cde import (
    CDEVectorField,
    NeuralCDE,
)


def make_cde(
    *,
    input_dim: int = 3,
    hidden_dim: int = 8,
    use_adjoint: bool = False,
) -> NeuralCDE:
    return NeuralCDE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        drift_hidden_dim=12,
        sensitivity_hidden_dim=12,
        solver="dopri5",
        rtol=1e-3,
        atol=1e-4,
        use_adjoint=use_adjoint,
    )


def test_vector_field_has_correct_augmented_shape():
    field = CDEVectorField(
        input_dim=3,
        hidden_dim=8,
        drift_hidden_dim=12,
        sensitivity_hidden_dim=12,
    )

    hidden = torch.randn(
        4,
        8,
    )

    result = field(
        torch.tensor(0.5),
        hidden,
    )

    # One channel for dt
    # plus three channels for dX.
    assert result.shape == (
        4,
        8,
        4,
    )

    assert torch.isfinite(
        result
    ).all()


def test_natural_cubic_control_recovers_observed_knots():
    torch.manual_seed(
        42
    )

    model = make_cde()

    path = torch.randn(
        2,
        8,
        3,
    )

    (
        control,
        times,
        flat_path,
    ) = model.build_control(
        path
    )

    reconstructed = control.evaluate(
        times
    )

    expected_time = (
        times
        .view(
            1,
            -1,
            1,
        )
        .expand(
            path.shape[0],
            -1,
            -1,
        )
    )

    torch.testing.assert_close(
        reconstructed[
            ...,
            :1,
        ],
        expected_time,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        reconstructed[
            ...,
            1:,
        ],
        flat_path,
        rtol=1e-5,
        atol=1e-5,
    )


def test_cde_returns_full_hidden_trajectory():
    torch.manual_seed(
        42
    )

    model = make_cde()

    path = torch.randn(
        2,
        8,
        3,
    )

    trajectory = model(
        path
    )

    assert trajectory.shape == (
        2,
        8,
        8,
    )

    assert torch.isfinite(
        trajectory
    ).all()


def test_cde_accepts_asset_feature_tensor():
    torch.manual_seed(
        42
    )

    model = make_cde(
        input_dim=6,
    )

    path = torch.randn(
        2,
        8,
        3,
        2,
    )

    trajectory = model(
        path
    )

    assert trajectory.shape == (
        2,
        8,
        8,
    )


def test_adjoint_gradients_reach_all_cde_components():
    torch.manual_seed(
        42
    )

    model = make_cde(
        use_adjoint=True,
    )

    path = torch.randn(
        2,
        6,
        3,
    )

    final_state = model.encode_final(
        path
    )

    loss = (
        final_state
        .pow(2)
        .mean()
    )

    loss.backward()

    initial_gradients = [
        parameter.grad
        for parameter
        in model.initial_projection.parameters()
    ]

    drift_gradients = [
        parameter.grad
        for parameter
        in model.vector_field.drift.parameters()
    ]

    sensitivity_gradients = [
        parameter.grad
        for parameter
        in model.vector_field.sensitivity.parameters()
    ]

    for group in (
        initial_gradients,
        drift_gradients,
        sensitivity_gradients,
    ):
        assert all(
            gradient is not None
            for gradient
            in group
        )

        assert all(
            torch.isfinite(
                gradient
            ).all()
            for gradient
            in group
        )


def test_batch_specific_timestamp_grids_are_rejected():
    model = make_cde()

    path = torch.randn(
        2,
        8,
        3,
    )

    timestamps = torch.arange(
        8,
        dtype=torch.float32,
    ).repeat(
        2,
        1,
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        model(
            path,
            timestamps=timestamps,
        )
    

def test_rk4_cde_returns_finite_trajectory_and_gradients():
    torch.manual_seed(
        42
    )

    model = NeuralCDE(
        input_dim=3,
        hidden_dim=8,
        drift_hidden_dim=12,
        sensitivity_hidden_dim=12,
        solver="rk4",
        rtol=1e-3,
        atol=1e-4,
        use_adjoint=False,
        fixed_steps_per_interval=4,
    )

    path = torch.randn(
        2,
        8,
        3,
    )

    trajectory = model(
        path
    )

    assert trajectory.shape == (
        2,
        8,
        8,
    )

    assert torch.isfinite(
        trajectory
    ).all()

    loss = (
        trajectory
        .pow(2)
        .mean()
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(
            gradient
        ).all()
        for gradient
        in gradients
    )


def test_fixed_steps_per_interval_must_be_positive():
    with pytest.raises(
        ValueError,
        match="fixed_steps_per_interval",
    ):
        NeuralCDE(
            input_dim=3,
            hidden_dim=8,
            solver="rk4",
            fixed_steps_per_interval=0,
        )