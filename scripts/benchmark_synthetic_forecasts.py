"""Fit statistical baselines and evaluate a deterministic validation subset."""
import argparse

from diffusion_portfolio.evaluation.synthetic_benchmarks import benchmark_synthetic_forecasts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="gaussian,ar1,ewma", help="Comma-separated baseline names")
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument("--n-scenarios", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="First N available validation targets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ewma-decay", type=float, default=.94)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_synthetic_forecasts(
        args.config, output_dir=args.output_dir, models=tuple(name.strip() for name in args.models.split(",")),
        split=args.split, n_scenarios=args.n_scenarios, limit=args.limit, seed=args.seed, ewma_decay=args.ewma_decay,
    )
