import torch

from diffusion_portfolio.models.diffolio import (
    CrossAttentionBlock,
    SelfAttentionBlock,
)


def test_cross_attention_shapes():
    torch.manual_seed(
        123
    )

    block = CrossAttentionBlock(
        hidden_dim=128,
        num_heads=4,
        mlp_dim=512,
    )

    batch_size = 8
    lookback = 63

    query = torch.randn(
        batch_size,
        128,
    )

    context = torch.randn(
        batch_size,
        lookback,
        128,
    )

    output = block(
        query,
        context,
        context,
    )

    assert output.hidden.shape == (
        batch_size,
        128,
    )

    assert output.attention.shape == (
        batch_size,
        4,
        1,
        lookback,
    )


def test_cross_attention_probabilities_sum_to_one():
    torch.manual_seed(
        123
    )

    block = CrossAttentionBlock(
        hidden_dim=128,
        num_heads=4,
        mlp_dim=512,
    )

    query = torch.randn(
        4,
        128,
    )

    context = torch.randn(
        4,
        63,
        128,
    )

    output = block(
        query,
        context,
        context,
    )

    sums = output.attention.sum(
        dim=-1
    )

    torch.testing.assert_close(
        sums,
        torch.ones_like(
            sums
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_self_attention_shapes():
    torch.manual_seed(
        123
    )

    block = SelfAttentionBlock(
        hidden_dim=128,
        num_heads=4,
        mlp_dim=512,
    )

    # Diffolio:
    #
    # 12 asset tokens
    # +
    # 8 systematic tokens
    # =
    # 20 market-level tokens
    x = torch.randn(
        8,
        20,
        128,
    )

    output = block(
        x
    )

    assert output.hidden.shape == (
        8,
        20,
        128,
    )

    assert output.attention.shape == (
        8,
        4,
        20,
        20,
    )


def test_self_attention_probabilities_sum_to_one():
    torch.manual_seed(
        123
    )

    block = SelfAttentionBlock(
        hidden_dim=128,
        num_heads=4,
        mlp_dim=512,
    )

    x = torch.randn(
        4,
        20,
        128,
    )

    output = block(
        x
    )

    sums = output.attention.sum(
        dim=-1
    )

    torch.testing.assert_close(
        sums,
        torch.ones_like(
            sums
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_attention_blocks_are_differentiable():
    torch.manual_seed(
        123
    )

    cross = CrossAttentionBlock(
        hidden_dim=128,
        num_heads=4,
        mlp_dim=512,
    )

    market = SelfAttentionBlock(
        hidden_dim=128,
        num_heads=4,
        mlp_dim=512,
    )

    query = torch.randn(
        2,
        128,
        requires_grad=True,
    )

    context = torch.randn(
        2,
        63,
        128,
        requires_grad=True,
    )

    asset_output = cross(
        query,
        context,
        context,
    ).hidden

    systematic_tokens = torch.randn(
        2,
        8,
        128,
        requires_grad=True,
    )

    tokens = torch.cat(
        (
            asset_output.unsqueeze(
                1
            ).expand(
                -1,
                12,
                -1,
            ),
            systematic_tokens,
        ),
        dim=1,
    )

    output = market(
        tokens
    )

    loss = output.hidden.square().mean()

    loss.backward()

    assert query.grad is not None
    assert context.grad is not None
    assert systematic_tokens.grad is not None

    assert torch.isfinite(
        query.grad
    ).all()

    assert torch.isfinite(
        context.grad
    ).all()

    assert torch.isfinite(
        systematic_tokens.grad
    ).all()