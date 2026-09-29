import torch

from diffusion_portfolio.models.diffolio import (
    DiffolioHierarchy,
    SinusoidalDiffusionEmbedding,
)


def make_inputs(
    batch_size: int = 4,
):
    torch.manual_seed(
        123
    )

    noisy_target = torch.randn(
        batch_size,
        12,
    )

    timesteps = torch.randint(
        low=1,
        high=1001,
        size=(
            batch_size,
        ),
    )

    return_history = torch.randn(
        batch_size,
        63,
        12,
    )

    asset_covariates = torch.randn(
        batch_size,
        63,
        12,
        10,
    )

    systematic_covariates = torch.randn(
        batch_size,
        63,
        8,
    )

    return (
        noisy_target,
        timesteps,
        return_history,
        asset_covariates,
        systematic_covariates,
    )


def test_sinusoidal_embedding_shape_and_finiteness():
    embedding = (
        SinusoidalDiffusionEmbedding(
            embedding_dim=32
        )
    )

    timesteps = torch.tensor(
        [
            1,
            10,
            100,
            1000,
        ]
    )

    output = embedding(
        timesteps
    )

    assert output.shape == (
        4,
        32,
    )

    assert torch.isfinite(
        output
    ).all()


def test_hierarchy_output_shapes():
    model = DiffolioHierarchy()

    inputs = make_inputs(
        batch_size=4
    )

    output = model(
        *inputs
    )

    assert output.asset_latents.shape == (
        4,
        12,
        128,
    )

    assert (
        output.systematic_latents.shape
        == (
            4,
            8,
            128,
        )
    )

    assert output.asset_attention.shape == (
        4,
        12,
        4,
        1,
        63,
    )

    assert output.market_attention.shape == (
        4,
        4,
        20,
        20,
    )


def test_asset_cross_attention_is_independent_before_stage_two():
    torch.manual_seed(
        123
    )

    model = DiffolioHierarchy()

    (
        noisy_target,
        timesteps,
        return_history,
        asset_covariates,
        systematic_covariates,
    ) = make_inputs(
        batch_size=2
    )

    # Capture Stage-1 output directly.
    captured = {}

    def hook(
        module,
        args,
        output,
    ):
        captured[
            "hidden"
        ] = output.hidden.detach().clone()

    handle = (
        model.asset_attention.register_forward_hook(
            hook
        )
    )

    model(
        noisy_target,
        timesteps,
        return_history,
        asset_covariates,
        systematic_covariates,
    )

    original = captured[
        "hidden"
    ].reshape(
        2,
        12,
        128,
    )

    changed_history = (
        return_history.clone()
    )

    changed_covariates = (
        asset_covariates.clone()
    )

    # Change only asset 0.
    changed_history[
        :,
        :,
        0,
    ] += 100.0

    changed_covariates[
        :,
        :,
        0,
        :,
    ] -= 100.0

    model(
        noisy_target,
        timesteps,
        changed_history,
        changed_covariates,
        systematic_covariates,
    )

    changed = captured[
        "hidden"
    ].reshape(
        2,
        12,
        128,
    )

    handle.remove()

    # Other assets must be identical before market-level
    # self-attention.
    torch.testing.assert_close(
        original[
            :,
            1:,
            :,
        ],
        changed[
            :,
            1:,
            :,
        ],
        atol=0.0,
        rtol=0.0,
    )


def test_shared_asset_embeddings_have_single_parameter_set():
    model = DiffolioHierarchy()

    # There is one shared asset context embedding,
    # not one layer per asset.
    assert model.asset_context_embedding.in_features == 11
    assert model.asset_context_embedding.out_features == 128

    assert model.query_embedding.in_features == 33
    assert model.query_embedding.out_features == 128


def test_hierarchy_is_differentiable():
    model = DiffolioHierarchy()

    (
        noisy_target,
        timesteps,
        return_history,
        asset_covariates,
        systematic_covariates,
    ) = make_inputs(
        batch_size=2
    )

    noisy_target.requires_grad_()
    return_history.requires_grad_()
    asset_covariates.requires_grad_()
    systematic_covariates.requires_grad_()

    output = model(
        noisy_target,
        timesteps,
        return_history,
        asset_covariates,
        systematic_covariates,
    )

    loss = (
        output.asset_latents.square().mean()
        + output.systematic_latents.square().mean()
    )

    loss.backward()

    for tensor in (
        noisy_target,
        return_history,
        asset_covariates,
        systematic_covariates,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(
            tensor.grad
        ).all()