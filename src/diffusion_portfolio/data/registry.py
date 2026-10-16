"""Small dispatch for currently available return-panel sources."""
from functools import partial

from diffusion_portfolio.data.core import ReturnTable
from diffusion_portfolio.data.sources.kenneth_french import load_kf12_daily
from diffusion_portfolio.data.sources.french_portfolios import (
    load_kf49_daily, load_kf100_size_bm_daily,
)
from diffusion_portfolio.data.sources.synthetic_market import load_synthetic_market


def load_return_table(dataset_name: str, **kwargs) -> ReturnTable:
    """Load arithmetic decimal asset returns in source-defined asset order.

    Keyword arguments are forwarded to the source adapter. Future sources must
    convert their units explicitly to this contract before returning a panel.
    Controlled synthetic sources expose observations only, never hidden states.
    """
    loaders = {
        "ken_french_12": load_kf12_daily,
        "ken_french_49": load_kf49_daily,
        "ken_french_100_size_bm": load_kf100_size_bm_daily,
        "synthetic_gaussian": partial(load_synthetic_market, "synthetic_gaussian"),
        "synthetic_regime": partial(load_synthetic_market, "synthetic_regime"),
        "synthetic_regime_jump": partial(load_synthetic_market, "synthetic_regime_jump"),
    }
    try:
        loader = loaders[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown return dataset: {dataset_name}") from None
    return loader(**kwargs)
