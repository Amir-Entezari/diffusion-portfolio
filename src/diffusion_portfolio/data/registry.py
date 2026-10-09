"""Small dispatch for currently available return-panel sources."""
from diffusion_portfolio.data.core import ReturnTable
from diffusion_portfolio.data.sources.kenneth_french import load_kf12_daily


def load_return_table(dataset_name: str, **kwargs) -> ReturnTable:
    """Load decimal returns in source-defined asset order.

    Keyword arguments are forwarded to the source loader. Synthetic paths keep
    their existing generator API and log-return convention.
    """
    loaders = {"ken_french_12": load_kf12_daily}
    try:
        loader = loaders[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown return dataset: {dataset_name}") from None
    return loader(**kwargs)
