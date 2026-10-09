"""CLI for training probes experiments."""
import argparse

from diffusion_portfolio.training.probes import run_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cde-run-dir",
        default="/kaggle/working/phase0_cde",
    )

    parser.add_argument(
        "--config",
        default="configs/evidential.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/phase0_evidential",
    )
    parser.add_argument(
        "--regime-target",
        choices=[
            "next_day",
            "current",
        ],
        default="current",
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
    run_probe(parse_args())
