"""Train frozen-CDE topology, geometry or covariance conditioning ablations."""
import argparse
from diffusion_portfolio.data.features import FEATURE_VARIANTS
from diffusion_portfolio.training.runs import train_feature_ablation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=FEATURE_VARIANTS, required=True)
    parser.add_argument("--cde-run-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_feature_ablation(
        args.variant, args.cde_run_dir, args.feature_dir, args.output_dir,
        device=args.device, resume=args.resume,
    )
