"""Compare strictly matched saved Vanilla/statistical validation forecasts."""
import argparse

from diffusion_portfolio.evaluation.forecast_comparison import compare_synthetic_forecasts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--diffusion-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_synthetic_forecasts(args.baseline_dir, args.diffusion_dir, output_dir=args.output_dir,
                                limit=args.limit, block_length=args.block_length,
                                n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
