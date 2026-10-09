"""Local real-data diagnostic for the Diffolio reproduction.

This is deliberately NOT a benchmark experiment.

It checks:

1. Can the full hierarchical denoiser overfit a tiny set of
   real KF12 examples under fixed diffusion corruption?

2. After short stochastic denoising training, does the
   50-step deterministic DDIM sampler remain finite and
   non-collapsed?

No result from this script should be reported as model
performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from diffusion_portfolio.baselines.diffolio import (
    DiffolioObjective,
    compute_training_covariance,
    sample_diffolio_ddim,
)


# ============================================================
# Configuration
# ============================================================

GOYAL_PATH = "data/raw/Data2025.xlsx"

SAMPLE_START = "1958-01-01"
SAMPLE_END = "2023-12-31"

TRAIN_END = "1999-12-31"
VAL_END = "2004-12-31"

LOOKBACK = 63

# Keep this deliberately tiny.
N_OVERFIT_SAMPLES = 16

# Phase A:
# exact same examples + exact same corruption every step.
FIXED_OVERFIT_STEPS = 500

# Phase B:
# random timestep + random epsilon every step.
STOCHASTIC_STEPS = 500

LEARNING_RATE = 3e-4

N_DIAGNOSTIC_SCENARIOS = 64
DDIM_STEPS = 50

SEED = 42


from diffusion_portfolio.baselines.diffolio.preparation import prepare_diffolio_data

def seed_everything(
    seed: int,
) -> None:
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def prepare_real_data():
    splits, raw_returns, return_scaler, _ = prepare_diffolio_data({
        "dataset": "ken_french_12", "goyal_path": GOYAL_PATH,
        "sample_start": SAMPLE_START, "sample_end": SAMPLE_END,
        "train_end": TRAIN_END, "val_end": VAL_END, "lookback": LOOKBACK,
    })
    train_mask = raw_returns.dates <= pd.Timestamp(TRAIN_END)
    training_covariance = compute_training_covariance(
        torch.from_numpy(raw_returns.returns[train_mask]).float()
    )
    return splits.train, training_covariance, return_scaler


def select_tiny_batch(
    train_windows,
    *,
    device: torch.device,
):
    """Select a deterministic spread of training examples."""

    n_total = (
        train_windows.target.shape[
            0
        ]
    )

    indices = np.linspace(
        0,
        n_total - 1,
        num=N_OVERFIT_SAMPLES,
        dtype=np.int64,
    )

    target = torch.from_numpy(
        train_windows.target[
            indices
        ]
    ).float().to(
        device
    )

    return_history = torch.from_numpy(
        train_windows.return_history[
            indices
        ]
    ).float().to(
        device
    )

    return_history_raw = torch.from_numpy(
        train_windows.return_history_raw[
            indices
        ]
    ).float().to(
        device
    )

    asset_covariates = torch.from_numpy(
        train_windows.asset_covariates[
            indices
        ]
    ).float().to(
        device
    )

    systematic_covariates = torch.from_numpy(
        train_windows.systematic_covariates[
            indices
        ]
    ).float().to(
        device
    )

    return (
        target,
        return_history,
        return_history_raw,
        asset_covariates,
        systematic_covariates,
    )


def run_diagnostic() -> None:
    seed_everything(
        SEED
    )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )
    else:
        device = torch.device(
            "cpu"
        )

    print(
        "Device:",
        device
    )

    (
        train_windows,
        training_covariance,
        return_scaler,
    ) = prepare_real_data()

    batch = select_tiny_batch(
        train_windows,
        device=device,
    )

    (
        target,
        return_history,
        return_history_raw,
        asset_covariates,
        systematic_covariates,
    ) = batch

    print()
    print(
        "Tiny diagnostic batch:"
    )
    print(
        " target:",
        tuple(
            target.shape
        ),
    )
    print(
        " return history:",
        tuple(
            return_history.shape
        ),
    )
    print(
        " asset covariates:",
        tuple(
            asset_covariates.shape
        ),
    )
    print(
        " systematic:",
        tuple(
            systematic_covariates.shape
        ),
    )

    # ========================================================
    # Full Diffolio architecture.
    # ========================================================

    model = DiffolioObjective(
        training_covariance=(
            training_covariance
        ),
        n_assets=12,
        n_asset_characteristics=10,
        n_systematic=8,
        lookback=63,
        hidden_dim=128,
        num_heads=4,
        mlp_dim=512,
        time_embedding_dim=32,
        diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        lambda_corr=0.05,
    ).to(
        device
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    # ========================================================
    # Phase A — exact fixed-corruption memorization.
    # ========================================================

    print()
    print(
        "=" * 72
    )
    print(
        "PHASE A — FIXED-CORRUPTION MEMORIZATION"
    )
    print(
        "=" * 72
    )

    probe_generator = torch.Generator(
        device="cpu"
    )

    probe_generator.manual_seed(
        12345
    )

    probe_noise = torch.randn(
        target.shape,
        generator=probe_generator,
        dtype=target.dtype,
        device="cpu",
    ).to(
        device
    )

    # Cover the complete diffusion horizon.
    probe_timesteps = torch.linspace(
        0,
        model.diffusion_steps - 1,
        steps=N_OVERFIT_SAMPLES,
        device=device,
    ).round().long()

    model.eval()

    with torch.no_grad():
        initial_probe = (
            model.training_loss(
                target,
                return_history,
                return_history_raw,
                asset_covariates,
                systematic_covariates,
                noise=probe_noise,
                timesteps=probe_timesteps,
            )
        )

    initial_ddpm = float(
        initial_probe.ddpm_loss.cpu()
    )

    print(
        f"Initial fixed DDPM MSE: "
        f"{initial_ddpm:.6f}"
    )

    model.train()

    for step in range(
        1,
        FIXED_OVERFIT_STEPS + 1,
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        output = model.training_loss(
            target,
            return_history,
            return_history_raw,
            asset_covariates,
            systematic_covariates,
            noise=probe_noise,
            timesteps=probe_timesteps,
        )

        if not torch.isfinite(
            output.loss
        ):
            raise RuntimeError(
                "Non-finite loss during "
                "fixed-corruption phase"
            )

        output.loss.backward()

        optimizer.step()

        if (
            step == 1
            or step % 100 == 0
            or step
            == FIXED_OVERFIT_STEPS
        ):
            print(
                f"Fixed step "
                f"{step:04d}/"
                f"{FIXED_OVERFIT_STEPS:04d} | "
                f"total={float(output.loss.detach()):.6f} | "
                f"ddpm={float(output.ddpm_loss.detach()):.6f} | "
                f"corr={float(output.correlation_loss.detach()):.6f}"
            )

    model.eval()

    with torch.no_grad():
        final_probe = (
            model.training_loss(
                target,
                return_history,
                return_history_raw,
                asset_covariates,
                systematic_covariates,
                noise=probe_noise,
                timesteps=probe_timesteps,
            )
        )

    final_ddpm = float(
        final_probe.ddpm_loss.cpu()
    )

    reduction = (
        1.0
        - final_ddpm
        / initial_ddpm
    )

    print()
    print(
        f"Final fixed DDPM MSE:   "
        f"{final_ddpm:.6f}"
    )

    print(
        f"Fixed-probe reduction:  "
        f"{100.0 * reduction:.2f}%"
    )

    if not (
        final_ddpm
        < initial_ddpm
    ):
        raise RuntimeError(
            "FAIL: fixed-corruption DDPM MSE "
            "did not improve"
        )

    if reduction < 0.25:
        raise RuntimeError(
            "FAIL: fixed-corruption memorization "
            "was weak (<25% DDPM MSE reduction)"
        )

    print(
        "PASS: full denoiser can memorize "
        "the fixed real-data probe"
    )

    # ========================================================
    # Phase B — short stochastic denoising training.
    #
    # Purpose:
    # expose the model to many different diffusion
    # timesteps/noise draws before exercising DDIM.
    # ========================================================

    print()
    print(
        "=" * 72
    )
    print(
        "PHASE B — STOCHASTIC DIFFUSION TRAINING"
    )
    print(
        "=" * 72
    )

    model.train()

    for step in range(
        1,
        STOCHASTIC_STEPS + 1,
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        output = model.training_loss(
            target,
            return_history,
            return_history_raw,
            asset_covariates,
            systematic_covariates,
        )

        if not torch.isfinite(
            output.loss
        ):
            raise RuntimeError(
                "Non-finite loss during "
                "stochastic phase"
            )

        output.loss.backward()

        optimizer.step()

        if (
            step == 1
            or step % 100 == 0
            or step
            == STOCHASTIC_STEPS
        ):
            print(
                f"Random step "
                f"{step:04d}/"
                f"{STOCHASTIC_STEPS:04d} | "
                f"total={float(output.loss.detach()):.6f} | "
                f"ddpm={float(output.ddpm_loss.detach()):.6f} | "
                f"corr={float(output.correlation_loss.detach()):.6f}"
            )

    # ========================================================
    # Phase B2 — fresh-noise denoising skill by timestep.
    #
    # This diagnoses the high-noise end of the diffusion
    # process separately. A zero predictor has MSE ~= 1.
    # ========================================================

    print()
    print(
        "=" * 72
    )
    print(
        "PHASE B2 — FRESH-NOISE SKILL BY TIMESTEP"
    )
    print(
        "=" * 72
    )

    probe_steps = (
        0,
        249,
        499,
        749,
        999,
    )

    n_probe_draws = 8

    fresh_generator = torch.Generator(
        device="cpu"
    )

    fresh_generator.manual_seed(
        20260929
    )

    model.eval()

    for timestep in probe_steps:
        model_squared_error = 0.0
        zero_squared_error = 0.0

        with torch.no_grad():
            for _ in range(
                n_probe_draws
            ):
                fresh_noise = torch.randn(
                    target.shape,
                    generator=fresh_generator,
                    dtype=target.dtype,
                    device="cpu",
                ).to(
                    device
                )

                schedule_timesteps = torch.full(
                    (
                        target.shape[0],
                    ),
                    timestep,
                    device=device,
                    dtype=torch.long,
                )

                noisy_target = (
                    model.noise_schedule.q_sample(
                        target,
                        schedule_timesteps,
                        fresh_noise,
                    )
                )

                denoiser_output = (
                    model.denoiser(
                        noisy_target,
                        schedule_timesteps + 1,
                        return_history,
                        asset_covariates,
                        systematic_covariates,
                    )
                )

                model_squared_error += float(
                    (
                        denoiser_output.noise
                        - fresh_noise
                    )
                    .square()
                    .mean()
                    .cpu()
                )

                # epsilon_hat = 0 baseline.
                zero_squared_error += float(
                    fresh_noise
                    .square()
                    .mean()
                    .cpu()
                )

        model_mse = (
            model_squared_error
            / n_probe_draws
        )

        zero_mse = (
            zero_squared_error
            / n_probe_draws
        )

        skill = (
            1.0
            - model_mse
            / zero_mse
        )

        alpha_bar = float(
            model.noise_schedule
            .alphas_cumprod[
                timestep
            ]
            .detach()
            .cpu()
        )

        amplification = (
            1.0
            / np.sqrt(
                alpha_bar
            )
        )

        print(
            f"t={timestep:03d} | "
            f"model MSE={model_mse:.6f} | "
            f"zero MSE={zero_mse:.6f} | "
            f"skill={100.0 * skill:+7.2f}% | "
            f"1/sqrt(a_bar)={amplification:8.2f}"
        )

    # ========================================================
    # Phase C — actual 50-step DDIM path.
    # ========================================================

    print()
    print(
        "=" * 72
    )
    print(
        "PHASE C — 50-STEP DDIM STABILITY"
    )
    print(
        "=" * 72
    )

    # Only two conditioning windows are needed here.
    sampler_history = (
        return_history[
            :2
        ]
    )

    sampler_asset = (
        asset_covariates[
            :2
        ]
    )

    sampler_systematic = (
        systematic_covariates[
            :2
        ]
    )

    initial_generator = torch.Generator(
        device="cpu"
    )

    initial_generator.manual_seed(
        999
    )

    initial_noise = torch.randn(
        (
            2,
            N_DIAGNOSTIC_SCENARIOS,
            12,
        ),
        generator=initial_generator,
        dtype=target.dtype,
        device="cpu",
    ).to(
        device
    )

    scenarios = sample_diffolio_ddim(
        model,
        sampler_history,
        sampler_asset,
        sampler_systematic,
        n_scenarios=(
            N_DIAGNOSTIC_SCENARIOS
        ),
        sampling_steps=DDIM_STEPS,
        initial_noise=initial_noise,
    )

    finite = bool(
        torch.isfinite(
            scenarios
        ).all()
    )

    scenario_std = float(
        scenarios.std(
            dim=1,
            unbiased=False,
        )
        .mean()
        .cpu()
    )

    model_abs_max = float(
        scenarios.abs()
        .max()
        .cpu()
    )

    raw_scenarios = (
        return_scaler.inverse_transform(
            scenarios
            .detach()
            .cpu()
            .numpy()
        )
    )

    raw_abs_max = float(
        np.abs(
            raw_scenarios
        ).max()
    )

    print(
        "Finite scenarios:",
        finite,
    )

    print(
        "Mean scenario std "
        f"(model space): {scenario_std:.6f}"
    )

    print(
        "Max |sample| "
        f"(model space): {model_abs_max:.6f}"
    )

    print(
        "Max |sample| "
        f"(raw excess return): {raw_abs_max:.6f}"
    )

    if not finite:
        raise RuntimeError(
            "FAIL: DDIM generated non-finite values"
        )

    if scenario_std <= 1e-6:
        raise RuntimeError(
            "FAIL: DDIM scenarios collapsed"
        )

    if model_abs_max >= 1e4:
        raise RuntimeError(
            "FAIL: DDIM sampler numerically exploded"
        )

    print(
        "PASS: 50-step DDIM is finite "
        "and non-collapsed"
    )

    print()
    print(
        "=" * 72
    )
    print(
        "LOCAL DIAGNOSTIC PASSED"
    )
    print(
        "=" * 72
    )

    print(
        "This validates implementation/optimization "
        "only — not forecasting quality."
    )
