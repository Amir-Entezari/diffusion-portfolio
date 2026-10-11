"""Training-sample Gaussian DDPM error and reverse-chain scale diagnostics.

These checks diagnose a sampler; they are not forecast performance metrics.
The original diagnostic's CPU noise streams (777, 1234, 5678) are retained.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


DIAGNOSTIC_TIMESTEPS = (0, 1, 10, 25, 50, 75, 90, 95, 98, 99)


@torch.no_grad()
def diagnose_reverse_sampler(model, histories, targets, *, timesteps=None,
                             n_repeats=8, n_scenarios=128):
    """Return epsilon errors and stochastic/zero-noise chain scale summaries.

    Both epsilon and v-prediction are converted to epsilon for the error
    amplification diagnostic. Reverse transitions use the model's canonical
    implementation; this function does not implement another sampler.
    """
    if hasattr(model, "alpha"):
        raise ValueError("This diagnostic uses Gaussian DDPM error amplification")
    if n_repeats <= 0 or n_scenarios <= 0:
        raise ValueError("Repeat and scenario counts must be positive")
    targets = model._prepare_target(targets)
    if not len(targets) or len(histories) != len(targets):
        raise ValueError("Histories and targets must have matching nonempty batches")
    if timesteps is None:
        timesteps = sorted({
            step for step in DIAGNOSTIC_TIMESTEPS if step < model.diffusion_steps
        } | {model.diffusion_steps - 1})
    else:
        timesteps = list(timesteps)
    if not timesteps or any(step < 0 or step >= model.diffusion_steps for step in timesteps):
        raise ValueError("Diagnostic timesteps must be within the diffusion schedule")

    device, dtype = targets.device, targets.dtype
    was_training = model.training
    model.eval()
    try:
        history = histories.repeat_interleave(n_repeats, dim=0)
        target = targets.repeat_interleave(n_repeats, dim=0)
        betas = model.noise_schedule.effective_betas.detach().to(device=device, dtype=dtype)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        errors = []
        generator = torch.Generator(device="cpu").manual_seed(777)
        for step in timesteps:
            noise = torch.randn(target.shape, generator=generator, dtype=dtype).to(device)
            steps = torch.full((len(target),), step, dtype=torch.long, device=device)
            noisy = model.noise_schedule.q_sample(target, steps, noise)
            prediction = model.predict_model_output(noisy, steps, history)
            if model.prediction_type == "epsilon":
                epsilon = prediction
            else:
                a, s = model._diffusion_coefficients(steps, noisy)
                epsilon = a * prediction + s * noisy
            mse = float(F.mse_loss(epsilon, noise).cpu())
            rmse = math.sqrt(mse)
            amplification = float(betas[step]) / math.sqrt(
                float(alphas[step]) * (1.0 - float(alpha_bar[step]))
            )
            errors.append({
                "step": step, "beta": float(betas[step]),
                "alpha_bar": float(alpha_bar[step]),
                "prediction_mse": float(F.mse_loss(
                    prediction, model._make_training_target(target, noise, steps)
                ).cpu()),
                "epsilon_mse": mse, "epsilon_rmse": rmse,
                "error_amplification": amplification,
                "amplified_rmse": amplification * rmse,
            })

        condition = model._encode_condition(histories[:1]).repeat_interleave(n_scenarios, dim=0)
        initial_generator = torch.Generator(device="cpu").manual_seed(1234)
        initial = torch.randn(
            (n_scenarios, model.n_assets), generator=initial_generator, dtype=dtype
        ).to(device)
        chains = {}
        for stochastic in (True, False):
            x = initial.clone()
            generator = torch.Generator(device="cpu").manual_seed(5678)
            rows = []
            for step in reversed(range(model.diffusion_steps)):
                steps = torch.full((n_scenarios,), step, dtype=torch.long, device=device)
                noise = (
                    torch.randn(x.shape, generator=generator, dtype=dtype).to(device)
                    if stochastic and step > 0 else torch.zeros_like(x)
                )
                x = model.reverse_step(x, steps, condition, noise=noise)
                if step in timesteps:
                    rows.append({
                        "step": step, "mean": x.mean().item(), "std": x.std().item(),
                        "max_abs": x.abs().max().item(), "finite": bool(torch.isfinite(x).all()),
                    })
            chains["stochastic" if stochastic else "zero_posterior_noise"] = rows
        return {
            "prediction_type": model.prediction_type,
            "timestep_errors": errors, "reverse_chains": chains,
        }
    finally:
        model.train(was_training)


def run_sampler_diagnostics(config_path, checkpoint_path, *, standardizer_path=None,
                            device="auto", n_overfit_samples=32, n_repeats=8,
                            n_scenarios=128, timesteps=None):
    """Restore a Gaussian checkpoint and select training examples only."""
    from diffusion_portfolio.config import load_config
    from diffusion_portfolio.data.preparation import prepare_return_data
    from diffusion_portfolio.models.diffusion.build import build_diffusion
    from diffusion_portfolio.utils.device import resolve_device

    cfg = load_config(config_path)
    device = resolve_device(device)
    datasets, scaler = prepare_return_data(cfg, standardizer_path=standardizer_path)
    if not 0 < n_overfit_samples <= len(datasets.train):
        raise ValueError("Training sample count must be positive and no larger than the split")
    indices = np.sort(np.random.default_rng(cfg.seed).choice(
        len(datasets.train), size=n_overfit_samples, replace=False,
    ))
    histories = torch.stack([datasets.train[int(i)]["history"] for i in indices]).to(device)
    targets = torch.stack([datasets.train[int(i)]["target"] for i in indices]).to(device)
    model = build_diffusion(cfg, n_assets=len(scaler.columns)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    report = diagnose_reverse_sampler(
        model, histories, targets, timesteps=timesteps,
        n_repeats=n_repeats, n_scenarios=n_scenarios,
    )
    report.update(split="train", checkpoint_epoch=int(checkpoint["epoch"]),
                  checkpoint_val_loss=float(checkpoint["val_loss"]))
    return report
