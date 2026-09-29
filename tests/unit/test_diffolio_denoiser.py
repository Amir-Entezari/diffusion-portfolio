import torch
from torch import nn

from diffusion_portfolio.models.diffolio import (
    DiffolioDenoiser,
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


def test_denoiser_output_shapes():
    model = DiffolioDenoiser()

    output = model(
        *make_inputs(
            batch_size=4
        )
    )

    assert output.noise.shape == (
        4,
        12,
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

    assert torch.isfinite(
        output.noise
    ).all()


def test_decoder_is_single_shared_linear_layer():
    model = DiffolioDenoiser()

    assert isinstance(
        model.decoder,
        nn.Linear,
    )

    assert model.decoder.in_features == 128
    assert model.decoder.out_features == 1

    # There must not be one decoder per asset.
    linear_children = [
        module
        for module in model.children()
        if isinstance(
            module,
            nn.Linear,
        )
    ]

    assert linear_children == [
        model.decoder
    ]


def test_denoiser_is_differentiable_end_to_end():
    model = DiffolioDenoiser()

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

    loss = output.noise.square().mean()

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

    assert (
        model.decoder.weight.grad
        is not None
    )

    assert torch.isfinite(
        model.decoder.weight.grad
    ).all()


def test_each_asset_receives_one_scalar_prediction():
    model = DiffolioDenoiser()

    captured = {}

    def hook(
        module,
        args,
        output,
    ):
        captured[
            "decoder_output"
        ] = output.detach().clone()

    handle = (
        model.decoder.register_forward_hook(
            hook
        )
    )

    model(
        *make_inputs(
            batch_size=3
        )
    )

    handle.remove()

    assert captured[
        "decoder_output"
    ].shape == (
        3,
        12,
        1,
    )