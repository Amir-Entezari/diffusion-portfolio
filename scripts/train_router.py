"""CLI for training routing experiments."""
import argparse

from diffusion_portfolio.training.routing import run_router


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cde-run-dir",
        required=True,
    )
    parser.add_argument(
        "--probe-run-dir",
        required=True,
    )
    parser.add_argument(
        "--config",
        default="configs/router.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/phase0_router",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_router(parse_args())
