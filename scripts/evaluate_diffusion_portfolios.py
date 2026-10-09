"""CLI for evaluation portfolio_experiment experiments."""
import argparse

from diffusion_portfolio.evaluation.portfolio_experiment import run_portfolio_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_portfolio_evaluation(parse_args())
