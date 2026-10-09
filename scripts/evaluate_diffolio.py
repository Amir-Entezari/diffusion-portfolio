"""CLI for baselines diffolio evaluation experiments."""
import argparse
from pathlib import Path

from diffusion_portfolio.baselines.diffolio.evaluation import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/diffolio/"
            "diffolio_100k.pt"
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "val",
            "test",
        ],
        default="val",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional limit for a sampling smoke test."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
