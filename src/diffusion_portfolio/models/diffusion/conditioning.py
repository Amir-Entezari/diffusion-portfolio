"""Compatibility imports for history encoders; implementations live in encoders."""
from diffusion_portfolio.models.encoders.mlp import HistoryEncoder


def __getattr__(name):
    if name == "CDEHistoryEncoder":
        from diffusion_portfolio.models.encoders.cde import CDEHistoryEncoder
        return CDEHistoryEncoder
    raise AttributeError(name)
