import torch

from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)
from diffusion_portfolio.models.diffusion import PrecomputedConditionDiffusion


def make_diffusion():
    return ConditionalDiffusionModel(
        lookback=8,
        n_assets=3,
        condition_dim=6,
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


def test_cde_only_condition_is_unchanged():
    model = PrecomputedConditionDiffusion(
        make_diffusion(),
        extra_dim=0,
    )

    condition = torch.randn(
        4,
        6,
    )

    torch.testing.assert_close(
        model.make_condition(
            condition
        ),
        condition,
    )


def test_zero_initialized_extra_features_do_not_change_condition():
    model = PrecomputedConditionDiffusion(
        make_diffusion(),
        extra_dim=5,
    )

    condition = torch.randn(
        4,
        6,
    )

    extra = torch.randn(
        4,
        5,
    )

    features = torch.cat(
        [
            condition,
            extra,
        ],
        dim=1,
    )

    torch.testing.assert_close(
        model.make_condition(
            features
        ),
        condition,
    )