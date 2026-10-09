from diffusion_portfolio.models.encoders.mlp import HistoryEncoder
from .model import (
    ConditionalDiffusionModel,
    DiffusionTrainingOutput,
)
from .schedule import NoiseSchedule
from .score_network import (
    ScoreNetwork,
    SinusoidalTimeEmbedding,
)

__all__ = [
    "PrecomputedConditionDiffusion",
    "ConditionalDiffusionModel",
    "DiffusionTrainingOutput",
    "HistoryEncoder",
    "NoiseSchedule",
    "ScoreNetwork",
    "SinusoidalTimeEmbedding",
    "CDEHistoryEncoder",
]
from diffusion_portfolio.models.diffusion.precomputed import PrecomputedConditionDiffusion


def __getattr__(name):
    if name == "CDEHistoryEncoder":
        from diffusion_portfolio.models.encoders.cde import CDEHistoryEncoder
        return CDEHistoryEncoder
    raise AttributeError(name)
