from pathlib import Path

import yaml

from diffusion_portfolio.config import load_config
from diffusion_portfolio.models.diffusion import NoiseSchedule, ScoreNetwork


def test_config_values_reach_instantiated_components(tmp_path: Path):
    raw = {
        "seed": 7,
        "data": {
            "lookback": 40,
            "horizon": 1,
            "sample_start": "1960-01-01",
            "sample_end": "2020-12-31",
            "train_end": "1999-12-31",
            "val_end": "2005-12-31",
        },
        "model": {
            "condition_dim": 24,
            "diffusion_steps": 37,
            "schedule": "linear",
            "channels": [16, 32],
        },
        "training": {},
        "portfolio": {"transaction_cost_bps": 23.0},
        "evaluation": {},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    cfg = load_config(path)
    schedule = NoiseSchedule(
        n_steps=cfg.model.diffusion_steps,
        schedule_type=cfg.model.schedule,
    )
    net = ScoreNetwork(
        data_dim=12,
        channels=list(cfg.model.channels),
        time_embed_dim=16,
        condition_dim=cfg.model.condition_dim,
        n_res_blocks=1,
    )

    assert cfg.seed == 7
    assert cfg.data.lookback == 40
    assert cfg.portfolio.transaction_cost_bps == 23.0
    assert schedule.n_steps == 37
    assert net.time_embed[1].out_features == 16
    # AdaGN receives time embedding + model condition.
    assert net.encoders[0][0].norm1.proj.in_features == 16 + 24
