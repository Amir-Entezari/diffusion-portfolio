"""Diagnose Gaussian reverse sampling on training examples only."""
import argparse
import json
from pathlib import Path

from diffusion_portfolio.evaluation.sampler_diagnostics import run_sampler_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/overfit_debug_best.pt")
    parser.add_argument("--standardizer", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-overfit-samples", type=int, default=32)
    parser.add_argument("--n-repeats", type=int, default=8)
    parser.add_argument("--n-scenarios", type=int, default=128)
    parser.add_argument("--timesteps", nargs="+", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_sampler_diagnostics(
        args.config, args.checkpoint, standardizer_path=args.standardizer,
        device=args.device, n_overfit_samples=args.n_overfit_samples,
        n_repeats=args.n_repeats, n_scenarios=args.n_scenarios, timesteps=args.timesteps,
    )
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
