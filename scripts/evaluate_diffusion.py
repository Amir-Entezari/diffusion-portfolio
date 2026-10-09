"""Evaluate current diffusion models; defaults to validation only."""
import argparse
from diffusion_portfolio.data.features import FEATURE_VARIANTS
from diffusion_portfolio.evaluation.forecasts import evaluate_diffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--variant", choices=FEATURE_VARIANTS, default=None)
    parser.add_argument("--cde-run-dir", default=None)
    parser.add_argument("--feature-dir", default=None)
    args = parser.parse_args()
    if args.variant is not None and (args.cde_run_dir is None or args.feature_dir is None):
        parser.error("--variant requires --cde-run-dir and --feature-dir")
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate_diffusion(
        args.run_dir, batch_size=args.batch_size, device=args.device, split=args.split,
        variant=args.variant, cde_run_dir=args.cde_run_dir, feature_dir=args.feature_dir,
    )
