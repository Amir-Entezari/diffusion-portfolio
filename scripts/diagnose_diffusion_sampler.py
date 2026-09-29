"""Diagnose timestep-specific DDPM error and reverse-chain stability."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import (
    TrainStandardizer,
    build_window_datasets,
    load_daily_risk_free,
    load_kf12_daily,
    slice_return_table,
    to_excess_returns,
)
from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)


N_OVERFIT_SAMPLES = 32
N_REPEATS = 8
N_TRAJECTORY_SCENARIOS = 128

TIMESTEPS_TO_CHECK = [
    0,
    1,
    10,
    25,
    50,
    75,
    90,
    95,
    98,
    99,
]


def main():
    cfg = load_config(
        "configs/mvp.yaml"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Rebuild exact data used by overfit diagnostic
    # ---------------------------------------------------------
    assets = slice_return_table(
        load_kf12_daily(),
        start=cfg.data.sample_start,
        end=cfg.data.sample_end,
    )

    risk_free = load_daily_risk_free()

    excess = to_excess_returns(
        assets,
        risk_free,
    )

    scaler = TrainStandardizer.fit(
        excess,
        train_end=cfg.data.train_end,
    )

    scaled = scaler.transform(
        excess
    )

    datasets = build_window_datasets(
        scaled,
        excess,
        lookback=cfg.data.lookback,
        horizon=cfg.data.horizon,
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
        test_end=cfg.data.sample_end,
    )

    rng = np.random.default_rng(
        cfg.seed
    )

    indices = np.sort(
        rng.choice(
            len(datasets.train),
            size=N_OVERFIT_SAMPLES,
            replace=False,
        )
    )

    histories = torch.stack(
        [
            datasets.train[int(i)][
                "history"
            ]
            for i in indices
        ]
    ).to(device)

    targets = torch.stack(
        [
            datasets.train[int(i)][
                "target"
            ][0]
            for i in indices
        ]
    ).to(device)

    # ---------------------------------------------------------
    # Load best overfit model
    # ---------------------------------------------------------
    model = ConditionalDiffusionModel(
        lookback=cfg.data.lookback,
        n_assets=12,
        condition_dim=(
            cfg.model.condition_dim
        ),
        history_hidden_dim=(
            cfg.model.history_hidden_dim
        ),
        diffusion_steps=(
            cfg.model.diffusion_steps
        ),
        schedule_type=(
            cfg.model.schedule
        ),
        channels=list(
            cfg.model.channels
        ),
        time_embed_dim=(
            cfg.model.time_embed_dim
        ),
        n_res_blocks=(
            cfg.model.n_res_blocks
        ),
    ).to(device)

    checkpoint = torch.load(
        "checkpoints/overfit_debug_best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint val loss:",
        checkpoint["val_loss"],
    )

    # ---------------------------------------------------------
    # Actual schedule
    # ---------------------------------------------------------
    betas = (
        model.noise_schedule
        .effective_betas
        .detach()
        .to(device)
    )

    alphas = 1.0 - betas

    alpha_bar = torch.cumprod(
        alphas,
        dim=0,
    )

    # ---------------------------------------------------------
    # Timestep-specific epsilon MSE
    # ---------------------------------------------------------
    print()
    print("=" * 100)
    print("TIMESTEP-SPECIFIC EPSILON ERROR")
    print("=" * 100)

    print(
        f"{'t':>4} "
        f"{'beta':>12} "
        f"{'alpha_bar':>14} "
        f"{'MSE':>12} "
        f"{'RMSE':>12} "
        f"{'err_amp':>12} "
        f"{'amp*RMSE':>12}"
    )

    repeated_history = (
        histories.repeat_interleave(
            N_REPEATS,
            dim=0,
        )
    )

    repeated_target = (
        targets.repeat_interleave(
            N_REPEATS,
            dim=0,
        )
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        777
    )

    with torch.no_grad():
        for step in TIMESTEPS_TO_CHECK:
            batch_size = (
                repeated_target.shape[0]
            )

            noise = torch.randn(
                repeated_target.shape,
                generator=generator,
                dtype=repeated_target.dtype,
                device="cpu",
            ).to(device)

            timesteps = torch.full(
                (batch_size,),
                step,
                dtype=torch.long,
                device=device,
            )

            x_t = (
                model.noise_schedule.q_sample(
                    repeated_target,
                    timesteps,
                    noise,
                )
            )

            predicted = (
                model.predict_noise(
                    x_t,
                    timesteps,
                    repeated_history,
                )
            )

            mse = float(
                F.mse_loss(
                    predicted,
                    noise,
                ).cpu()
            )

            rmse = math.sqrt(
                mse
            )

            beta = float(
                betas[step].cpu()
            )

            alpha = float(
                alphas[step].cpu()
            )

            abar = float(
                alpha_bar[step].cpu()
            )

            # Magnitude multiplying epsilon-prediction
            # error in the DDPM reverse mean.
            error_amplification = (
                beta
                / math.sqrt(
                    alpha
                    * (1.0 - abar)
                )
            )

            propagated_rmse = (
                error_amplification
                * rmse
            )

            print(
                f"{step:4d} "
                f"{beta:12.6f} "
                f"{abar:14.8f} "
                f"{mse:12.6f} "
                f"{rmse:12.6f} "
                f"{error_amplification:12.4f} "
                f"{propagated_rmse:12.4f}"
            )

    # ---------------------------------------------------------
    # Free reverse trajectory
    # ---------------------------------------------------------
    history = histories[:1]

    with torch.no_grad():
        condition = (
            model.history_encoder(
                history
            )
        )

        condition = (
            condition.repeat_interleave(
                N_TRAJECTORY_SCENARIOS,
                dim=0,
            )
        )

    initial_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    initial_generator.manual_seed(
        1234
    )

    initial_x = torch.randn(
        (
            N_TRAJECTORY_SCENARIOS,
            model.n_assets,
        ),
        generator=initial_generator,
        device="cpu",
    ).to(device)

    watch_steps = {
        99,
        98,
        95,
        90,
        75,
        50,
        25,
        10,
        1,
        0,
    }

    def run_trajectory(
        *,
        stochastic: bool,
    ):
        x = initial_x.clone()

        noise_generator = (
            torch.Generator(
                device="cpu"
            )
        )

        noise_generator.manual_seed(
            5678
        )

        print()
        print(
            "STOCHASTIC"
            if stochastic
            else "ZERO POSTERIOR NOISE"
        )

        print(
            f"{'after t':>8} "
            f"{'mean':>14} "
            f"{'std':>14} "
            f"{'max_abs':>14}"
        )

        with torch.no_grad():
            for step in reversed(
                range(
                    model.diffusion_steps
                )
            ):
                timesteps = torch.full(
                    (
                        N_TRAJECTORY_SCENARIOS,
                    ),
                    step,
                    dtype=torch.long,
                    device=device,
                )

                if (
                    stochastic
                    and step > 0
                ):
                    step_noise = torch.randn(
                        x.shape,
                        generator=(
                            noise_generator
                        ),
                        device="cpu",
                    ).to(device)

                else:
                    step_noise = (
                        torch.zeros_like(
                            x
                        )
                    )

                x = model.reverse_step(
                    x,
                    timesteps,
                    condition,
                    noise=step_noise,
                )

                if step in watch_steps:
                    print(
                        f"{step:8d} "
                        f"{x.mean().item():14.6f} "
                        f"{x.std().item():14.6f} "
                        f"{x.abs().max().item():14.6f}"
                    )

    print()
    print("=" * 100)
    print("REVERSE-CHAIN SCALE")
    print("=" * 100)

    run_trajectory(
        stochastic=True
    )

    run_trajectory(
        stochastic=False
    )


if __name__ == "__main__":
    main()