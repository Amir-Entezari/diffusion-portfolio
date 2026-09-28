"""Cheap local/Kaggle smoke test for the refactored package."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data.synthetic import SyntheticConfig, SyntheticJumpDiffusionGenerator
from diffusion_portfolio.models.diffusion import NoiseSchedule, ScoreNetwork
from diffusion_portfolio.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    set_global_seed(cfg.seed)

    n_assets = 12  # Stage 2 will derive this from the real dataset contract.
    gen = SyntheticJumpDiffusionGenerator(
        SyntheticConfig(n_assets=n_assets, n_timesteps=128, seed=cfg.seed)
    )
    data = gen.generate()
    assert data["returns"].shape == (128, n_assets)

    batch = torch.tensor(data["returns"][1:9], dtype=torch.float32)
    schedule = NoiseSchedule(
        n_steps=cfg.model.diffusion_steps,
        schedule_type=cfg.model.schedule,
    )
    t = torch.randint(0, cfg.model.diffusion_steps, (batch.shape[0],))
    noise = torch.randn_like(batch)
    x_t = schedule.q_sample(batch, t, noise)

    cond = torch.randn(batch.shape[0], cfg.model.condition_dim)
    net = ScoreNetwork(
        data_dim=n_assets,
        channels=list(cfg.model.channels),
        time_embed_dim=min(128, cfg.model.condition_dim),
        condition_dim=cfg.model.condition_dim,
        n_res_blocks=1,
    )
    pred = net(x_t, t, cond)
    assert pred.shape == batch.shape
    assert torch.isfinite(pred).all()

    print("Smoke test passed")
    print(f"config: {args.config}")
    print(f"diffusion_steps: {schedule.n_steps}")
    print(f"condition_dim: {cfg.model.condition_dim}")
    print(f"channels: {cfg.model.channels}")
    print(f"returns: {tuple(data['returns'].shape)}")
    print(f"score output: {tuple(pred.shape)}")


if __name__ == "__main__":
    main()
