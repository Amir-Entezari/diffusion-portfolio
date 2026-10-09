"""CLI for baselines diffolio scenario_diagnostics experiments."""
import argparse
from pathlib import Path

from diffusion_portfolio.baselines.diffolio.scenario_diagnostics import run_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(
            "artifacts/diffolio/val_scenarios.npz"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostics(parse_args())
