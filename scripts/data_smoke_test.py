"""CLI for data diagnostics experiments."""
import argparse
import json
from pathlib import Path

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
    parser.add_argument("--report-json", type=Path, default=None,
                        help="Save the successful smoke report for notebook provenance.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_data_check(args)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
