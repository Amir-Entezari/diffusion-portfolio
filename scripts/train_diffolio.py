"""CLI for baselines diffolio experiment experiments."""
import argparse
from pathlib import Path

from diffusion_portfolio.baselines.diffolio.experiment import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Diffolio reproduction "
            "on KF12 real data."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/diffolio.yaml"
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Override training.total_steps. "
            "Useful for smoke runs."
        ),
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help=(
            "Override training.warmup_steps."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "'auto', 'cpu', 'cuda', "
            "or another torch device string."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Override checkpoint path."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume model/optimizer state from "
            "a Diffolio checkpoint."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
