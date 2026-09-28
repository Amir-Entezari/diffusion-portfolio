"""Cheap local/Kaggle smoke test for the refactored package."""
from __future__ import annotations

import torch

from diffusion_portfolio.data.synthetic import SyntheticConfig, SyntheticJumpDiffusionGenerator
from diffusion_portfolio.models.diffusion import NoiseSchedule, ScoreNetwork
from diffusion_portfolio.utils.seed import set_global_seed


def main() -> None:
    set_global_seed(42)

    gen = SyntheticJumpDiffusionGenerator(
        SyntheticConfig(n_assets=12, n_timesteps=128, seed=42)
    )
    data = gen.generate()
    assert data["returns"].shape == (128, 12)

    batch = torch.tensor(data["returns"][1:9], dtype=torch.float32)
    schedule = NoiseSchedule(n_steps=20, schedule_type="cosine")
    t = torch.randint(0, 20, (batch.shape[0],))
    noise = torch.randn_like(batch)
    x_t = schedule.q_sample(batch, t, noise)

    cond = torch.randn(batch.shape[0], 16)
    net = ScoreNetwork(
        data_dim=12,
        channels=[16, 32],
        time_embed_dim=16,
        condition_dim=16,
        n_res_blocks=1,
    )
    pred = net(x_t, t, cond)
    assert pred.shape == batch.shape
    assert torch.isfinite(pred).all()

    print("Smoke test passed")
    print(f"returns: {tuple(data['returns'].shape)}")
    print(f"score output: {tuple(pred.shape)}")


if __name__ == "__main__":
    main()
