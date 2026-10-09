"""Extract current topology or covariance/SPD features without test evaluation."""
import argparse
from diffusion_portfolio.data.precompute import precompute_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["topology", "spd"], required=True)
    parser.add_argument("--cde-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eigenvalue-floor", type=float, default=1e-6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precompute_features(
        args.kind, args.cde_run_dir, args.output_dir, batch_size=args.batch_size,
        device=args.device, eigenvalue_floor=args.eigenvalue_floor,
    )
