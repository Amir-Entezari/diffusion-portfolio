import numpy as np

from diffusion_portfolio.data.sources.synthetic import SyntheticConfig, SyntheticJumpDiffusionGenerator


def test_synthetic_generator_shapes_and_finiteness():
    data = SyntheticJumpDiffusionGenerator(
        SyntheticConfig(n_assets=5, n_timesteps=64, lob_depth=3, seed=7)
    ).generate()
    assert data["returns"].shape == (64, 5)
    assert data["prices"].shape == (64, 5)
    assert data["lob"].shape == (64, 5, 12)
    assert np.isfinite(data["returns"]).all()
    assert np.isfinite(data["prices"]).all()
