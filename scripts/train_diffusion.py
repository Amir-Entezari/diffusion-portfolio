"""Train current Gaussian or Lévy diffusion, using explicit saved preprocessing."""
import argparse
from diffusion_portfolio.training.runs import train_diffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs/vanilla_diffusion")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--reference-run-dir", default="/kaggle/working/phase0_cde")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_diffusion(
        args.config or ("configs/levy.yaml" if args.alpha is not None else "configs/mvp.yaml"),
        args.output_dir, device=args.device, resume=args.resume,
        stop_after_epoch=args.stop_after_epoch, alpha=args.alpha,
        reference_run_dir=args.reference_run_dir,
    )
