"""Diffolio reproduction components."""

from diffusion_portfolio.models.diffolio.attention import (
    AttentionBlockOutput,
    CrossAttentionBlock,
    FeedForward,
    SelfAttentionBlock,
)
from diffusion_portfolio.models.diffolio.hierarchy import (
    DiffolioHierarchy,
    DiffolioHierarchyOutput,
    SinusoidalDiffusionEmbedding,
)
from diffusion_portfolio.models.diffolio.denoiser import (
    DiffolioDenoiser,
    DiffolioDenoiserOutput,
)

from diffusion_portfolio.models.diffolio.correlation_target import (
    ShrinkageCorrelationTarget,
    compute_training_covariance,
    covariance_to_correlation,
    estimate_shrinkage_correlation,
)

__all__ = [
    "AttentionBlockOutput",
    "CrossAttentionBlock",
    "FeedForward",
    "SelfAttentionBlock",
    "DiffolioHierarchy",
    "DiffolioHierarchyOutput",
    "SinusoidalDiffusionEmbedding",
    "DiffolioDenoiser",
    "DiffolioDenoiserOutput",
    "ShrinkageCorrelationTarget",
    "compute_training_covariance",
    "covariance_to_correlation",
    "estimate_shrinkage_correlation",
]