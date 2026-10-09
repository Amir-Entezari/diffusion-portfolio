import torch
import torch.nn as nn

from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)
from diffusion_portfolio.models.uncertainty.routing import (
    EvidentialResidualRouter,
    SoftmaxResidualRouter,
)


def test_softmax_router_starts_as_identity():
    router = SoftmaxResidualRouter(
        condition_dim=8,
        n_experts=3,
        expert_hidden_dim=12,
    )

    condition = torch.randn(
        5,
        8,
    )

    output = router.route(
        condition
    )

    torch.testing.assert_close(
        output.condition,
        condition,
    )

    torch.testing.assert_close(
        output.weights.sum(
            dim=-1
        ),
        torch.ones(5),
        atol=1e-6,
        rtol=1e-6,
    )


def test_evidential_router_starts_as_identity():
    router = EvidentialResidualRouter(
        condition_dim=8,
        n_experts=3,
        expert_hidden_dim=12,
        uncertainty_threshold=0.7,
        transition_steepness=10.0,
    )

    condition = torch.randn(
        5,
        8,
    )

    output = router.route(
        condition
    )

    torch.testing.assert_close(
        output.condition,
        condition,
    )

    torch.testing.assert_close(
        output.weights.sum(
            dim=-1
        ),
        torch.ones(5),
        atol=1e-6,
        rtol=1e-6,
    )

    assert output.vacuity is not None

    assert output.vacuity.shape == (
        5,
    )


def test_high_vacuity_moves_weights_toward_uniform():
    router = EvidentialResidualRouter(
        condition_dim=4,
        n_experts=3,
        expert_hidden_dim=8,
        uncertainty_threshold=0.5,
        transition_steepness=20.0,
    )

    with torch.no_grad():
        (
            router
            .evidential_head
            .evidence_layer
            .weight
            .zero_()
        )

        (
            router
            .evidential_head
            .evidence_layer
            .bias
            .fill_(
                -100.0
            )
        )

    condition = torch.zeros(
        2,
        4,
    )

    output = router.route(
        condition
    )

    uniform = torch.full(
        (
            2,
            3,
        ),
        1.0 / 3.0,
    )

    torch.testing.assert_close(
        output.weights,
        uniform,
        atol=1e-4,
        rtol=1e-4,
    )


def test_softmax_and_evidential_router_parameter_counts_match():
    softmax = SoftmaxResidualRouter(
        condition_dim=8,
        n_experts=3,
        expert_hidden_dim=12,
    )

    evidential = EvidentialResidualRouter(
        condition_dim=8,
        n_experts=3,
        expert_hidden_dim=12,
        uncertainty_threshold=0.7,
    )

    softmax_count = sum(
        parameter.numel()
        for parameter
        in softmax.parameters()
    )

    evidential_count = sum(
        parameter.numel()
        for parameter
        in evidential.parameters()
    )

    assert (
        softmax_count
        == evidential_count
    )


class AddOneAdapter(nn.Module):
    def forward(
        self,
        condition,
    ):
        return condition + 1.0


def test_diffusion_model_applies_condition_adapter():
    model = ConditionalDiffusionModel(
        lookback=6,
        n_assets=3,
        condition_dim=8,
        history_hidden_dim=12,
        diffusion_steps=10,
        schedule_type="cosine",
        prediction_type="v_prediction",
        channels=[
            8,
            16,
        ],
        time_embed_dim=8,
        n_res_blocks=1,
    )

    history = torch.randn(
        4,
        6,
        3,
    )

    base_condition = (
        model.history_encoder(
            history
        )
    )

    model.condition_adapter = (
        AddOneAdapter()
    )

    routed_condition = (
        model._encode_condition(
            history
        )
    )

    torch.testing.assert_close(
        routed_condition,
        base_condition + 1.0,
    )