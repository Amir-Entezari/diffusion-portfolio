"""CLI for evaluation tail_diagnostics experiments."""
import argparse

from diffusion_portfolio.evaluation.tail_diagnostics import run_tail_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cde-run-dir",
        default="/kaggle/working/phase0_cde",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_tail_diagnostics(parse_args())
