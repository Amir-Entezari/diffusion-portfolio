"""Training-side conditional alpha-stable diffusion model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion_portfolio.models.diffusion.model import (
    ConditionalDiffusionModel,
    DiffusionTrainingOutput,
)
from diffusion_portfolio.models.levy.noise import (
    sample_ddpm_normalized_alpha_stable,
    sample_ddpm_normalized_positive_stable_mixer,
)
from diffusion_portfolio.models.levy.schedule import (
    LevyNoiseSchedule,
)


class ConditionalLevyDiffusionModel(
    ConditionalDiffusionModel
):
    """Conditional epsilon-prediction alpha-stable diffusion.

    The model implements the alpha-stable forward process,
    epsilon-prediction training objective, and the DLPM-style
    reverse process conditioned on a positive-stable mixer path.

    Reverse sampling is deliberately not implemented yet.
    """

    def __init__(
        self,
        *,
        alpha: float,
        lookback: int,
        n_assets: int,
        condition_dim: int,
        history_hidden_dim: int,
        diffusion_steps: int,
        schedule_type: str,
        channels: list[int],
        time_embed_dim: int,
        n_res_blocks: int,
        history_encoder_type: str = "mlp",
        cde_hidden_dim: int = 128,
        cde_drift_hidden_dim: int = 256,
        cde_sensitivity_hidden_dim: int = 256,
        cde_solver: str = "dopri5",
        cde_rtol: float = 1e-4,
        cde_atol: float = 1e-5,
        cde_use_adjoint: bool = True,
        cde_fixed_steps_per_interval: int = 4,
    ) -> None:
        super().__init__(
            lookback=lookback,
            n_assets=n_assets,
            condition_dim=condition_dim,
            history_hidden_dim=history_hidden_dim,
            diffusion_steps=diffusion_steps,
            schedule_type=schedule_type,
            channels=channels,
            prediction_type="epsilon",
            time_embed_dim=time_embed_dim,
            n_res_blocks=n_res_blocks,
            history_encoder_type=(
                history_encoder_type
            ),
            cde_hidden_dim=cde_hidden_dim,
            cde_drift_hidden_dim=(
                cde_drift_hidden_dim
            ),
            cde_sensitivity_hidden_dim=(
                cde_sensitivity_hidden_dim
            ),
            cde_solver=cde_solver,
            cde_rtol=cde_rtol,
            cde_atol=cde_atol,
            cde_use_adjoint=(
                cde_use_adjoint
            ),
            cde_fixed_steps_per_interval=(
                cde_fixed_steps_per_interval
            ),
        )

        self.alpha = float(
            alpha
        )

        # Replace the Gaussian schedule created by the parent.
        self.noise_schedule = (
            LevyNoiseSchedule(
                alpha=self.alpha,
                n_steps=diffusion_steps,
                schedule_type=schedule_type,
            )
        )

    def sample_training_noise(
        self,
        *,
        shape: tuple[int, ...],
        device: torch.device | str,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample DDPM-normalized alpha-stable training corruption."""

        return sample_ddpm_normalized_alpha_stable(
            alpha=self.alpha,
            shape=shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def training_loss_from_condition(
        self,
        condition: Tensor,
        target: Tensor,
        *,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> DiffusionTrainingOutput:
        """Compute the DLPM-style epsilon-prediction loss."""

        x_0 = self._prepare_target(
            target
        )

        if condition.shape != (
            x_0.shape[0],
            self.condition_dim,
        ):
            raise ValueError(
                "condition must have shape "
                "[batch, condition_dim]"
            )

        condition = condition.to(
            device=x_0.device,
            dtype=x_0.dtype,
        )

        batch_size = (
            x_0.shape[0]
        )

        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.diffusion_steps,
                size=(
                    batch_size,
                ),
                device=x_0.device,
            )

        else:
            timesteps = timesteps.to(
                device=x_0.device,
                dtype=torch.long,
            )

            if timesteps.shape != (
                batch_size,
            ):
                raise ValueError(
                    "timesteps must have "
                    "shape [batch]"
                )

            if (
                torch.any(
                    timesteps < 0
                )
                or torch.any(
                    timesteps
                    >= self.diffusion_steps
                )
            ):
                raise ValueError(
                    "timesteps are outside "
                    "the diffusion schedule"
                )

        if noise is None:
            noise = self.sample_training_noise(
                shape=tuple(
                    x_0.shape
                ),
                device=x_0.device,
                dtype=x_0.dtype,
            )

        else:
            noise = noise.to(
                device=x_0.device,
                dtype=x_0.dtype,
            )

            if noise.shape != (
                x_0.shape
            ):
                raise ValueError(
                    "noise must match "
                    "target shape"
                )

        x_t = (
            self.noise_schedule.q_sample(
                x_0,
                timesteps,
                noise,
            )
        )

        prediction = (
            self.score_network(
                x_t,
                timesteps,
                condition,
            )
        )

        # DLPM epsilon target.
        #
        # The loss below follows the authors' lploss=2
        # implementation: per-sample RMSE, then batch mean.
        training_target = noise

        squared_error = F.mse_loss(
            prediction,
            training_target,
            reduction="none",
        )

        per_sample_loss = torch.sqrt(
            squared_error.mean(
                dim=tuple(
                    range(
                        1,
                        squared_error.ndim,
                    )
                )
            )
        )

        loss = per_sample_loss.mean()

        return DiffusionTrainingOutput(
            loss=loss,
            prediction=prediction,
            training_target=(
                training_target
            ),
            noisy_target=x_t,
            timesteps=timesteps,
        )

    @torch.no_grad()
    def _conditional_variance_path(
        self,
        mixer_path: Tensor,
    ) -> Tensor:
        """Compute Gaussian conditional variances given a mixer path.

        For each diffusion transition,

            Sigma_t =
                sigma_t^2 A_t
                + gamma_t^2 Sigma_{t-1}.

        The returned tensor has shape

            [diffusion_steps, batch, 1].
        """

        if mixer_path.ndim != 3:
            raise ValueError(
                "mixer_path must have shape "
                "[steps, batch, 1]"
            )

        if mixer_path.shape[
            0
        ] != self.diffusion_steps:
            raise ValueError(
                "mixer_path step dimension "
                "does not match diffusion_steps"
            )

        if mixer_path.shape[
            2
        ] != 1:
            raise ValueError(
                "isotropic mixer_path must have "
                "last dimension 1"
            )

        if not torch.isfinite(
            mixer_path
        ).all():
            raise ValueError(
                "mixer_path contains "
                "non-finite values"
            )

        if torch.any(
            mixer_path <= 0
        ):
            raise ValueError(
                "mixer_path must be positive"
            )

        dtype = (
            mixer_path.dtype
        )

        device = (
            mixer_path.device
        )

        gammas = (
            self.noise_schedule
            .gammas
            .to(
                device=device,
                dtype=dtype,
            )
        )

        sigmas = (
            self.noise_schedule
            .sigmas
            .to(
                device=device,
                dtype=dtype,
            )
        )

        batch_size = (
            mixer_path.shape[1]
        )

        previous = torch.zeros(
            (
                batch_size,
                1,
            ),
            device=device,
            dtype=dtype,
        )

        variances = []

        for step in range(
            self.diffusion_steps
        ):
            current = (
                sigmas[
                    step
                ].square()
                * mixer_path[
                    step
                ]
                + gammas[
                    step
                ].square()
                * previous
            )

            variances.append(
                current
            )

            previous = current

        variance_path = torch.stack(
            variances,
            dim=0,
        )

        if not torch.isfinite(
            variance_path
        ).all():
            raise RuntimeError(
                "conditional variance path "
                "contains non-finite values"
            )

        return variance_path


    def _levy_reverse_step(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        condition: Tensor,
        *,
        variance_previous: Tensor,
        variance_current: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample one DLPM reverse transition conditional on mixers."""

        if x_t.ndim != 2:
            raise ValueError(
                "x_t must have shape "
                "[batch, assets]"
            )

        batch_size = (
            x_t.shape[0]
        )

        if timesteps.shape != (
            batch_size,
        ):
            raise ValueError(
                "timesteps must have "
                "shape [batch]"
            )

        if condition.shape != (
            batch_size,
            self.condition_dim,
        ):
            raise ValueError(
                "condition has incorrect shape"
            )

        expected_variance_shape = (
            batch_size,
            1,
        )

        if variance_previous.shape != (
            expected_variance_shape
        ):
            raise ValueError(
                "variance_previous must have "
                "shape [batch, 1]"
            )

        if variance_current.shape != (
            expected_variance_shape
        ):
            raise ValueError(
                "variance_current must have "
                "shape [batch, 1]"
            )

        timesteps = timesteps.to(
            device=x_t.device,
            dtype=torch.long,
        )

        condition = condition.to(
            device=x_t.device,
            dtype=x_t.dtype,
        )

        variance_previous = (
            variance_previous.to(
                device=x_t.device,
                dtype=x_t.dtype,
            )
        )

        variance_current = (
            variance_current.to(
                device=x_t.device,
                dtype=x_t.dtype,
            )
        )

        coefficients = (
            self.noise_schedule
            .get_coefficients(
                timesteps
            )
        )

        gamma = (
            coefficients[
                "gamma"
            ]
            .to(
                device=x_t.device,
                dtype=x_t.dtype,
            )
            .unsqueeze(
                -1
            )
        )

        bar_sigma = (
            coefficients[
                "bar_sigma"
            ]
            .to(
                device=x_t.device,
                dtype=x_t.dtype,
            )
            .unsqueeze(
                -1
            )
        )

        eps = torch.finfo(
            x_t.dtype
        ).tiny

        denominator = (
            variance_current
            .clamp_min(
                eps
            )
        )

        posterior_fraction = (
            1.0
            - (
                gamma.square()
                * variance_previous
                / denominator
            )
        )

        # Numerical protection only.
        posterior_fraction = (
            posterior_fraction.clamp(
                min=0.0,
                max=1.0,
            )
        )

        prediction = (
            self.score_network(
                x_t,
                timesteps,
                condition,
            )
        )

        mean = (
            x_t
            - bar_sigma
            * posterior_fraction
            * prediction
        ) / gamma

        posterior_variance = (
            posterior_fraction
            * variance_previous
        ).clamp_min(
            0.0
        )

        if noise is None:
            noise = torch.randn_like(
                x_t
            )

        else:
            noise = noise.to(
                device=x_t.device,
                dtype=x_t.dtype,
            )

            if noise.shape != (
                x_t.shape
            ):
                raise ValueError(
                    "noise must match "
                    "x_t shape"
                )

        nonzero = (
            timesteps > 0
        ).to(
            dtype=x_t.dtype
        ).unsqueeze(
            -1
        )

        sample = (
            mean
            + nonzero
            * torch.sqrt(
                posterior_variance
            )
            * noise
        )

        if not torch.isfinite(
            sample
        ).all():
            raise RuntimeError(
                "Levy reverse step produced "
                "non-finite values"
            )

        return sample


    def reverse_step(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        condition: Tensor,
        *,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Expose ordinary DDPM reverse_step only at alpha=2.

        For alpha < 2 the correct transition requires the latent
        positive-stable mixer path and therefore must be performed
        through the full Levy sampler.
        """

        if self.alpha != 2.0:
            raise RuntimeError(
                "alpha < 2 reverse transitions "
                "require a latent mixer path; "
                "use sample_from_condition()"
            )

        return super().reverse_step(
            x_t,
            timesteps,
            condition,
            noise=noise,
        )


    @torch.no_grad()
    def sample_from_condition(
        self,
        condition: Tensor,
        *,
        n_scenarios: int,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Generate scenarios using the DLPM reverse chain."""

        # Exact Gaussian control.
        if self.alpha == 2.0:
            return super().sample_from_condition(
                condition,
                n_scenarios=n_scenarios,
                initial_noise=initial_noise,
            )

        if condition.ndim != 2:
            raise ValueError(
                "condition must have shape "
                "[batch, condition_dim]"
            )

        if condition.shape[
            1
        ] != self.condition_dim:
            raise ValueError(
                "condition dimension does "
                "not match model"
            )

        if n_scenarios <= 0:
            raise ValueError(
                "n_scenarios must be positive"
            )

        batch_size = (
            condition.shape[0]
        )

        total_samples = (
            batch_size
            * n_scenarios
        )

        repeated_condition = (
            condition.repeat_interleave(
                n_scenarios,
                dim=0,
            )
        )

        if initial_noise is None:
            initial = (
                sample_ddpm_normalized_alpha_stable(
                    alpha=self.alpha,
                    shape=(
                        batch_size,
                        n_scenarios,
                        self.n_assets,
                    ),
                    device=condition.device,
                    dtype=condition.dtype,
                )
            )

            terminal_scale = (
                self.noise_schedule
                .bar_sigmas[
                    -1
                ]
                .to(
                    device=condition.device,
                    dtype=condition.dtype,
                )
            )

            initial = (
                terminal_scale
                * initial
            )

        else:
            expected_shape = (
                batch_size,
                n_scenarios,
                self.n_assets,
            )

            if initial_noise.shape != (
                expected_shape
            ):
                raise ValueError(
                    "initial_noise must have shape "
                    "[batch, n_scenarios, assets]"
                )

            initial = initial_noise.to(
                device=condition.device,
                dtype=condition.dtype,
            )

        x = initial.reshape(
            total_samples,
            self.n_assets,
        )

        mixer_path = (
            sample_ddpm_normalized_positive_stable_mixer(
                alpha=self.alpha,
                shape=(
                    self.diffusion_steps,
                    total_samples,
                    1,
                ),
                device=x.device,
                dtype=x.dtype,
            )
        )

        variance_path = (
            self._conditional_variance_path(
                mixer_path
            )
        )

        for step in reversed(
            range(
                self.diffusion_steps
            )
        ):
            timesteps = torch.full(
                (
                    total_samples,
                ),
                step,
                device=x.device,
                dtype=torch.long,
            )

            variance_current = (
                variance_path[
                    step
                ]
            )

            if step == 0:
                variance_previous = (
                    torch.zeros_like(
                        variance_current
                    )
                )

                step_noise = (
                    torch.zeros_like(
                        x
                    )
                )

            else:
                variance_previous = (
                    variance_path[
                        step
                        - 1
                    ]
                )

                step_noise = (
                    torch.randn_like(
                        x
                    )
                )

            x = self._levy_reverse_step(
                x,
                timesteps,
                repeated_condition,
                variance_previous=(
                    variance_previous
                ),
                variance_current=(
                    variance_current
                ),
                noise=step_noise,
            )

        return x.reshape(
            batch_size,
            n_scenarios,
            self.n_assets,
        )