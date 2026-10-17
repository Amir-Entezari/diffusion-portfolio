"""Compare four completed context/capacity arms or two Small execution pilots."""
import argparse

from diffusion_portfolio.evaluation.capacity_ablation import diagnose_capacity_ablation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--diffusion-dirs", nargs="+", required=True, help="K1-Large=path K60-Large=path K1-Small=path K60-Small=path")
    parser.add_argument("--run-dirs", nargs="+", default=[], help="LABEL=original-run-dir, for hash-verified last.pt/history.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot", action="store_true", help="Compare exactly K1-Small/K60-Small two-epoch checkpoints only")
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    for field in ("diffusion_dirs", "run_dirs"):
        pairs = [value.split("=", 1) for value in getattr(args, field)]
        if any(len(pair) != 2 or not pair[0] or not pair[1] for pair in pairs):
            parser.error(f"{field} entries must be LABEL=path")
        if len({pair[0] for pair in pairs}) != len(pairs):
            parser.error(f"Duplicate {field} labels")
        setattr(args, field, dict(pairs))
    return args


if __name__ == "__main__":
    args = parse_args()
    diagnose_capacity_ablation(args.baseline_dir, args.diffusion_dirs, run_dirs=args.run_dirs,
                               output_dir=args.output_dir, pilot=args.pilot, block_length=args.block_length,
                               n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
