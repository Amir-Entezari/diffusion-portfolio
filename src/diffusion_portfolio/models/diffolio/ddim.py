"""Deterministic DDIM inference for Diffolio."""

from __future__ import annotations

import torch

from diffusion_portfolio.models.diffolio.objective import (
    DiffolioObjective,
)


def make_ddim_timesteps(
    *,
    diffusion_steps: int,
    sampling_steps: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create uniformly spaced reverse DDIM timestep indices.

    Returns zero-based schedule indices in DESCENDING order.

    Example
    -------
    diffusion_steps=10, sampling_steps=5

        [9, 7, 4, 2, 0]
    """

    if diffusion_steps <= 1:
        raise ValueError(
            "diffusion_steps must be > 1"
        )

    if sampling_steps <= 0:
        raise ValueError(
            "sampling_steps must be positive"
        )

    if sampling_steps > diffusion_steps:
        raise ValueError(
            "sampling_steps cannot exceed diffusion_steps"
        )

    if sampling_steps == 1:
        return torch.tensor(
            [
                diffusion_steps - 1
            ],
            dtype=torch.long,
            device=device,
        )

    ascending = torch.linspace(
        0,
        diffusion_steps - 1,
        steps=sampling_steps,
        device=device,
    )

    indices = torch.round(
        ascending
    ).to(
        dtype=torch.long
    )

    # Rounding should remain unique when S <= T.
    if torch.unique(
        indices
    ).numel() != sampling_steps:
        raise RuntimeError(
            "DDIM timestep construction produced duplicates"
        )

    return torch.flip(
        indices,
        dims=[
            0
        ],
    )


@torch.no_grad()
def sample_diffolio_ddim(
    model: DiffolioObjective,
    return_history: torch.Tensor,
    asset_covariates: torch.Tensor,
    systematic_covariates: torch.Tensor,
    *,
    n_scenarios: int,
    sampling_steps: int = 50,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate Diffolio scenarios with deterministic DDIM.

    Parameters
    ----------
    model:
        Trained Diffolio objective/model.

    return_history:
        Standardized return histories:

            [batch, lookback, assets]

    asset_covariates:
        Standardized asset-specific covariates:

            [batch, lookback, assets, characteristics]

    systematic_covariates:
        Standardized systematic covariates:

            [batch, lookback, systematic_features]

    n_scenarios:
        Number of independent scenarios per conditioning window.

    sampling_steps:
        Number of DDIM reverse steps. Diffolio uses 50.

    initial_noise:
        Optional Gaussian x_T initialization:

            [batch, n_scenarios, assets]

        Because eta=0, sampling is deterministic conditional
        on this initial noise.

    Returns
    -------
    torch.Tensor
        Generated next-day returns in MODEL SPACE:

            [batch, n_scenarios, assets]

        These must be inverse-transformed before financial
        evaluation.
    """

    if n_scenarios <= 0:
        raise ValueError(
            "n_scenarios must be positive"
        )

    if return_history.ndim != 3:
        raise ValueError(
            "return_history must have shape "
            "[batch, lookback, assets]"
        )

    batch_size = return_history.shape[
        0
    ]

    if return_history.shape != (
        batch_size,
        model.lookback,
        model.n_assets,
    ):
        raise ValueError(
            "return_history shape does not match model"
        )

    if asset_covariates.ndim != 4:
        raise ValueError(
            "asset_covariates must have shape "
            "[batch, lookback, assets, characteristics]"
        )

    if systematic_covariates.ndim != 3:
        raise ValueError(
            "systematic_covariates must have shape "
            "[batch, lookback, systematic_features]"
        )

    if (
        asset_covariates.shape[0]
        != batch_size
    ):
        raise ValueError(
            "asset_covariates batch size differs"
        )

    if (
        systematic_covariates.shape[0]
        != batch_size
    ):
        raise ValueError(
            "systematic_covariates batch size differs"
        )

    device = return_history.device
    dtype = return_history.dtype

    timesteps = make_ddim_timesteps(
        diffusion_steps=(
            model.diffusion_steps
        ),
        sampling_steps=(
            sampling_steps
        ),
        device=device,
    )

    repeated_history = (
        return_history.repeat_interleave(
            n_scenarios,
            dim=0,
        )
    )

    repeated_asset_covariates = (
        asset_covariates.repeat_interleave(
            n_scenarios,
            dim=0,
        )
    )

    repeated_systematic_covariates = (
        systematic_covariates.repeat_interleave(
            n_scenarios,
            dim=0,
        )
    )

    if initial_noise is None:
        x = torch.randn(
            (
                batch_size,
                n_scenarios,
                model.n_assets,
            ),
            device=device,
            dtype=dtype,
        )

    else:
        expected_shape = (
            batch_size,
            n_scenarios,
            model.n_assets,
        )

        if initial_noise.shape != expected_shape:
            raise ValueError(
                "initial_noise must have shape "
                "[batch, n_scenarios, assets]"
            )

        x = initial_noise.to(
            device=device,
            dtype=dtype,
        )

    x = x.reshape(
        batch_size
        * n_scenarios,
        model.n_assets,
    )

    alphas_cumprod = (
        model.noise_schedule
        .alphas_cumprod
        .to(
            device=device,
            dtype=dtype,
        )
    )

    was_training = model.training

    model.eval()

    try:
        for position, timestep in enumerate(
            timesteps
        ):
            t = int(
                timestep.item()
            )

            schedule_t = torch.full(
                (
                    batch_size
                    * n_scenarios,
                ),
                t,
                device=device,
                dtype=torch.long,
            )

            # Paper notation uses 1,...,T.
            diffusion_t = (
                schedule_t
                + 1
            )

            denoiser_output = model.denoiser(
                x,
                diffusion_t,
                repeated_history,
                repeated_asset_covariates,
                repeated_systematic_covariates,
            )

            predicted_noise = (
                denoiser_output.noise
            )

            alpha_bar_t = (
                alphas_cumprod[
                    t
                ]
            )

            sqrt_alpha_bar_t = torch.sqrt(
                alpha_bar_t
            )

            sqrt_one_minus_t = torch.sqrt(
                1.0
                - alpha_bar_t
            )

            # DDIM estimate of x_0:
            #
            # x0_hat =
            #   (x_t - sqrt(1-alpha_bar_t) * eps)
            #   / sqrt(alpha_bar_t)
            predicted_x0 = (
                x
                - sqrt_one_minus_t
                * predicted_noise
            ) / sqrt_alpha_bar_t

            # Final transition returns x0_hat directly.
            if position == (
                len(
                    timesteps
                )
                - 1
            ):
                x = predicted_x0
                continue

            previous_t = int(
                timesteps[
                    position + 1
                ].item()
            )

            alpha_bar_previous = (
                alphas_cumprod[
                    previous_t
                ]
            )

            # Deterministic DDIM: eta = 0.
            #
            # x_s =
            # sqrt(alpha_bar_s) * x0_hat
            # +
            # sqrt(1-alpha_bar_s) * eps_theta
            x = (
                torch.sqrt(
                    alpha_bar_previous
                )
                * predicted_x0
                + torch.sqrt(
                    1.0
                    - alpha_bar_previous
                )
                * predicted_noise
            )

    finally:
        model.train(
            was_training
        )

    return x.reshape(
        batch_size,
        n_scenarios,
        model.n_assets,
    )