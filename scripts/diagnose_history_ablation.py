"""Diagnose strictly matched saved context-ablation validation forecasts."""
import argparse

from diffusion_portfolio.evaluation.history_ablation import diagnose_history_ablation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--diffusion-dirs", nargs="+", required=True,
                        help="K1=path K60=path (or all K0/K1/K5/K60 archives)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    pairs = [value.split("=", 1) for value in args.diffusion_dirs]
    if any(len(pair) != 2 or not pair[1] for pair in pairs):
        parser.error("Each diffusion directory must be LABEL=path")
    if len({pair[0] for pair in pairs}) != len(pairs):
        parser.error("Duplicate ablation labels")
    args.diffusion_dirs = dict(pairs)
    return args


if __name__ == "__main__":
    args = parse_args()
    diagnose_history_ablation(args.baseline_dir, args.diffusion_dirs, output_dir=args.output_dir,
                              block_length=args.block_length, n_bootstrap=args.n_bootstrap,
                              bootstrap_seed=args.bootstrap_seed)
