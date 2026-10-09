"""CLI for data diagnostics experiments."""
import argparse

from diffusion_portfolio.data.diagnostics import run_data_check


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/mvp.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_data_check(parse_args())
