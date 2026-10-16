"""CLI for training multiseed experiments."""
import argparse
from pathlib import Path

from diffusion_portfolio.training.multiseed import run_multiseed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-config", default=None,
                        help="Matching Vanilla base config; defaults to the historical KF12 config.")
    parser.add_argument("--cde-config", default=None,
                        help="Matching CDE base config; defaults to the historical KF12 config.")

    parser.add_argument(
        "--seed42-vanilla-dir",
        required=True,
    )

    parser.add_argument(
        "--seed42-cde-dir",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        default="/kaggle/working/phase0_multiseed",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[43, 44],
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_multiseed(parse_args(), repo_root=Path(__file__).resolve().parents[1])
