"""CLI for data diagnostics experiments."""
import argparse

from diffusion_portfolio.data.diagnostics import run_data_check


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/mvp.yaml",
    )
    parser.add_argument(
        "--model-smoke", action="store_true",
        help="Also check one training batch's forward/loss/backward; no optimizer update.",
    )
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run_data_check(parse_args())
