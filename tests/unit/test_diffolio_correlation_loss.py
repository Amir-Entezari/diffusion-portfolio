import torch

from diffusion_portfolio.baselines.diffolio.correlation_loss import (
    correlation_guided_loss,
    diffolio_correlation_regularizer,
    extract_asset_attention,
)


def test_extract_asset_attention_averages_heads():
    market_attention = torch.zeros(
        2,
        4,
        5,
        5,
    )

    for head in range(
        4
    ):
        market_attention[
            :,
            head,
            :,
            :,
        ] = float(
            head + 1
        )

    asset = extract_asset_attention(
        market_attention,
        n_assets=3,
    )

    assert asset.shape == (
        2,
        3,
        3,
    )

    # Mean of 1,2,3,4 = 2.5.
    torch.testing.assert_close(
        asset,
        torch.full_like(
            asset,
            2.5,
        ),
    )


def test_asset_block_is_not_renormalized():
    # One batch, two heads, four total tokens:
    #
    # assets = first 2
    # systematic = final 2
    market_attention = torch.tensor(
        [
            [
                [
                    [0.20, 0.30, 0.25, 0.25],
                    [0.10, 0.20, 0.30, 0.40],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
                [
                    [0.40, 0.10, 0.25, 0.25],
                    [0.20, 0.10, 0.30, 0.40],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
            ]
        ],
        dtype=torch.float32,
    )

    asset = extract_asset_attention(
        market_attention,
        n_assets=2,
    )

    expected = torch.tensor(
        [
            [
                [0.30, 0.20],
                [0.15, 0.15],
            ]
        ]
    )

    torch.testing.assert_close(
        asset,
        expected,
    )

    # The rows deliberately sum below one because some
    # attention remains assigned to macro tokens.
    row_sums = asset.sum(
        dim=-1
    )

    assert torch.all(
        row_sums < 1.0
    )


def test_perfect_row_alignment_has_loss_minus_one():
    attention = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )

    target = attention.clone()

    loss = correlation_guided_loss(
        attention,
        target,
    )

    torch.testing.assert_close(
        loss,
        torch.tensor(
            -1.0
        ),
    )


def test_rowwise_cosine_is_computed_before_averaging():
    attention = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ]
    )

    target = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ]
    )

    # Row 0 cosine = 1.
    # Row 1 cosine = 0.
    #
    # Mean similarity = 0.5
    # Loss = -0.5.
    loss = correlation_guided_loss(
        attention,
        target,
    )

    torch.testing.assert_close(
        loss,
        torch.tensor(
            -0.5
        ),
    )


def test_target_is_detached_but_attention_receives_gradient():
    attention = torch.tensor(
        [
            [
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        ],
        requires_grad=True,
    )

    target = torch.tensor(
        [
            [
                [1.0, 0.5],
                [0.5, 1.0],
            ]
        ],
        requires_grad=True,
    )

    loss = correlation_guided_loss(
        attention,
        target,
    )

    loss.backward()

    assert attention.grad is not None

    assert torch.isfinite(
        attention.grad
    ).all()

    # Correlation estimate is supervision,
    # not a learnable computation graph.
    assert target.grad is None


def test_convenience_regularizer_is_differentiable():
    torch.manual_seed(
        123
    )

    logits = torch.randn(
        3,
        4,
        20,
        20,
        requires_grad=True,
    )

    market_attention = torch.softmax(
        logits,
        dim=-1,
    )

    target = torch.eye(
        12
    ).unsqueeze(
        0
    ).expand(
        3,
        -1,
        -1,
    )

    loss = diffolio_correlation_regularizer(
        market_attention,
        target,
        n_assets=12,
    )

    assert loss.ndim == 0

    assert torch.isfinite(
        loss
    )

    loss.backward()

    assert logits.grad is not None

    assert torch.isfinite(
        logits.grad
    ).all()