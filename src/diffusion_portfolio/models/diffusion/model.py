"""Vanilla conditional DDPM for next-day multivariate excess returns."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusion_portfolio.models.diffusion.conditioning import (
    CDEHistoryEncoder,
    HistoryEncoder,
)
from diffusion_portfolio.models.diffusion.schedule import (
    NoiseSchedule,
)
from diffusion_portfolio.models.diffusion.score_network import (
    ScoreNetwork,
)


@dataclass(frozen=True)
class DiffusionTrainingOutput:
    """Intermediate values from one diffusion training step."""

    loss: Tensor
    prediction: Tensor
    training_target: Tensor
    noisy_target: Tensor
    timesteps: Tensor


class ConditionalDiffusionModel(nn.Module):
    """Conditional epsilon-prediction DDPM.

    The model learns:

        epsilon_theta(x_t, t, history)

    where:

        history:
            previous L standardized excess returns

        x_0:
            standardized next-day multivariate excess return

        x_t:
            noisy version of x_0 at diffusion timestep t
    """

    def __init__(
        self,
        *,
        lookback: int,
        n_assets: int,
        condition_dim: int,
        history_hidden_dim: int,
        diffusion_steps: int,
        schedule_type: str,
        channels: list[int],
        prediction_type: str,
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
        super().__init__()

        if diffusion_steps <= 1:
            raise ValueError(
                "diffusion_steps must be > 1"
            )
        if prediction_type not in {
            "epsilon",
            "v_prediction",
        }:
            raise ValueError(
                "prediction_type must be "
                "'epsilon' or 'v_prediction'"
            )
        self.prediction_type = (
            prediction_type
        )
        self.lookback = lookback
        self.n_assets = n_assets
        self.condition_dim = (
            condition_dim
        )
        self.diffusion_steps = (
            diffusion_steps
        )

        self.history_encoder_type = (
            history_encoder_type
        )

        if history_encoder_type == "mlp":
            self.history_encoder = (
                HistoryEncoder(
                    lookback=lookback,
                    n_assets=n_assets,
                    condition_dim=condition_dim,
                    hidden_dim=(
                        history_hidden_dim
                    ),
                )
            )

        elif history_encoder_type == "cde":
            self.history_encoder = (
                CDEHistoryEncoder(
                    lookback=lookback,
                    n_assets=n_assets,
                    condition_dim=condition_dim,
                    hidden_dim=(
                        cde_hidden_dim
                    ),
                    drift_hidden_dim=(
                        cde_drift_hidden_dim
                    ),
                    sensitivity_hidden_dim=(
                        cde_sensitivity_hidden_dim
                    ),
                    solver=cde_solver,
                    rtol=cde_rtol,
                    atol=cde_atol,
                    use_adjoint=(
                        cde_use_adjoint
                    ),
                    fixed_steps_per_interval=(
                        cde_fixed_steps_per_interval
                    ),
                )
            )

        else:
            raise ValueError(
                "history_encoder_type must be "
                "'mlp' or 'cde'"
            )
        # Optional Phase-0B routing layer.
        #
        # Identity preserves all existing baseline behavior and
        # introduces no checkpoint parameters.
        self.condition_adapter = (
            nn.Identity()
        )
        
        self.noise_schedule = (
            NoiseSchedule(
                n_steps=diffusion_steps,
                schedule_type=(
                    schedule_type
                ),
            )
        )

        self.score_network = (
            ScoreNetwork(
                data_dim=n_assets,
                channels=channels,
                time_embed_dim=(
                    time_embed_dim
                ),
                condition_dim=(
                    condition_dim
                ),
                n_res_blocks=(
                    n_res_blocks
                ),
            )
        )

    def _prepare_target(
        self,
        target: Tensor,
    ) -> Tensor:
        """Convert [B,1,N] or [B,N] target to [B,N]."""

        if target.ndim == 3:
            if target.shape[1] != 1:
                raise ValueError(
                    "Vanilla MVP currently supports "
                    "horizon=1 only"
                )

            target = target[
                :,
                0,
                :,
            ]

        elif target.ndim != 2:
            raise ValueError(
                "target must have shape "
                "[batch, assets] or "
                "[batch, 1, assets]"
            )

        if target.shape[1] != self.n_assets:
            raise ValueError(
                "target asset dimension "
                "does not match model"
            )

        return target
    def _encode_condition(
        self,
        history: Tensor,
    ) -> Tensor:
        """Encode history and apply an optional condition adapter."""

        condition = (
            self.history_encoder(
                history
            )
        )

        condition = (
            self.condition_adapter(
                condition
            )
        )

        expected_shape = (
            history.shape[0],
            self.condition_dim,
        )

        if condition.shape != (
            expected_shape
        ):
            raise RuntimeError(
                "Condition adapter produced "
                f"{tuple(condition.shape)}; "
                f"expected {expected_shape}"
            )

        return condition
    def predict_model_output(
        self,
        noisy_target: Tensor,
        timesteps: Tensor,
        history: Tensor,
    ) -> Tensor:
        """Predict the configured diffusion parameterization."""

        if noisy_target.ndim != 2:
            raise ValueError(
                "noisy_target must have "
                "shape [batch, assets]"
            )

        if noisy_target.shape[1] != self.n_assets:
            raise ValueError(
                "noisy_target asset dimension "
                "does not match model"
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [batch]"
            )

        if (
            timesteps.shape[0]
            != noisy_target.shape[0]
        ):
            raise ValueError(
                "timestep batch dimension mismatch"
            )

        condition = (
            self._encode_condition(
                history
            )
        )

        return self.score_network(
            noisy_target,
            timesteps,
            condition,
        )
        
    def _diffusion_coefficients(
        self,
        timesteps: Tensor,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return broadcastable sqrt(alpha_bar) coefficients."""

        coefficients = (
            self.noise_schedule
            .get_coefficients(
                timesteps
            )
        )

        shape = (
            reference.shape[0],
            *(
                [1]
                * (
                    reference.ndim
                    - 1
                )
            ),
        )

        sqrt_alpha_bar = (
            coefficients[
                "sqrt_alpha_cumprod"
            ]
            .to(
                device=reference.device,
                dtype=reference.dtype,
            )
            .reshape(shape)
        )

        sqrt_one_minus = (
            coefficients[
                "sqrt_one_minus_alpha_cumprod"
            ]
            .to(
                device=reference.device,
                dtype=reference.dtype,
            )
            .reshape(shape)
        )

        return (
            sqrt_alpha_bar,
            sqrt_one_minus,
        )


    def _make_training_target(
        self,
        x_0: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Construct epsilon or v training target."""

        if self.prediction_type == "epsilon":
            return noise

        (
            sqrt_alpha_bar,
            sqrt_one_minus,
        ) = self._diffusion_coefficients(
            timesteps,
            x_0,
        )

        # v = sqrt(alpha_bar) * epsilon
        #     - sqrt(1-alpha_bar) * x_0
        return (
            sqrt_alpha_bar
            * noise
            - sqrt_one_minus
            * x_0
        )


    def _prediction_to_x0(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        prediction: Tensor,
    ) -> Tensor:
        """Convert model output to a prediction of clean x_0."""

        (
            sqrt_alpha_bar,
            sqrt_one_minus,
        ) = self._diffusion_coefficients(
            timesteps,
            x_t,
        )

        if self.prediction_type == "epsilon":
            denominator = (
                sqrt_alpha_bar.clamp_min(
                    torch.finfo(
                        x_t.dtype
                    ).eps
                )
            )

            return (
                x_t
                - sqrt_one_minus
                * prediction
            ) / denominator

        # v-prediction:
        #
        # x_0 =
        # sqrt(alpha_bar) * x_t
        # - sqrt(1-alpha_bar) * v
        return (
            sqrt_alpha_bar
            * x_t
            - sqrt_one_minus
            * prediction
        )
    def reverse_step(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        condition: Tensor,
        *,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample one DDPM reverse transition.

        Computes:

            p_theta(x_{t-1} | x_t, condition)

        using epsilon prediction.

        Parameters
        ----------
        x_t:
            Current noisy sample, shape [batch, assets].

        timesteps:
            Integer diffusion timesteps, shape [batch].

        condition:
            Precomputed history condition, shape
            [batch, condition_dim].

        noise:
            Optional Gaussian noise for the posterior transition.
            Ignored automatically when t = 0.
        """

        if x_t.ndim != 2:
            raise ValueError(
                "x_t must have shape "
                "[batch, assets]"
            )

        if x_t.shape[1] != self.n_assets:
            raise ValueError(
                "x_t asset dimension "
                "does not match model"
            )

        if timesteps.shape != (
            x_t.shape[0],
        ):
            raise ValueError(
                "timesteps must have shape [batch]"
            )

        if condition.shape != (
            x_t.shape[0],
            self.condition_dim,
        ):
            raise ValueError(
                "condition must have shape "
                "[batch, condition_dim]"
            )

        timesteps = timesteps.to(
            device=x_t.device,
            dtype=torch.long,
        )

        if (
            torch.any(timesteps < 0)
            or torch.any(
                timesteps
                >= self.diffusion_steps
            )
        ):
            raise ValueError(
                "timesteps are outside "
                "the diffusion schedule"
            )

        # Support both the current fixed schedules and the
        # optional learned-beta implementation.
        betas = (
            self.noise_schedule
            .effective_betas
            .to(
                device=x_t.device,
                dtype=x_t.dtype,
            )
        )

        alphas = (
            1.0
            - betas
        )

        alphas_cumprod = torch.cumprod(
            alphas,
            dim=0,
        )

        beta_t = betas[
            timesteps
        ]

        alpha_t = alphas[
            timesteps
        ]

        alpha_bar_t = (
            alphas_cumprod[
                timesteps
            ]
        )

        # alpha_bar_{t-1}; by definition alpha_bar_{-1} = 1
        alpha_bar_prev = torch.ones_like(
            alpha_bar_t
        )

        nonzero_mask = (
            timesteps > 0
        )

        if torch.any(
            nonzero_mask
        ):
            alpha_bar_prev[
                nonzero_mask
            ] = alphas_cumprod[
                timesteps[
                    nonzero_mask
                ]
                - 1
            ]

        prediction = (
            self.score_network(
                x_t,
                timesteps,
                condition,
            )
        )

        predicted_x0 = (
            self._prediction_to_x0(
                x_t,
                timesteps,
                prediction,
            )
        )

        # q(x_{t-1} | x_t, x_0) posterior mean
        denominator = (
            1.0
            - alpha_bar_t
        )

        posterior_mean_coef_x0 = (
            beta_t
            * torch.sqrt(
                alpha_bar_prev
            )
            / denominator
        )

        posterior_mean_coef_xt = (
            torch.sqrt(
                alpha_t
            )
            * (
                1.0
                - alpha_bar_prev
            )
            / denominator
        )

        mean = (
            posterior_mean_coef_x0.unsqueeze(
                -1
            )
            * predicted_x0
            + posterior_mean_coef_xt.unsqueeze(
                -1
            )
            * x_t
        )

        posterior_variance = (
            beta_t
            * (
                1.0
                - alpha_bar_prev
            )
            / (
                1.0
                - alpha_bar_t
            )
        )

        posterior_variance = (
            posterior_variance.clamp(
                min=0.0
            )
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

            if noise.shape != x_t.shape:
                raise ValueError(
                    "noise must match x_t shape"
                )

        # At t=0 there is no additional stochastic term.
        stochastic_mask = (
            timesteps > 0
        ).to(
            dtype=x_t.dtype
        ).unsqueeze(-1)

        return (
            mean
            + stochastic_mask
            * torch.sqrt(
                posterior_variance
            ).unsqueeze(-1)
            * noise
        )

    @torch.no_grad()
    def sample(
        self,
        history: Tensor,
        *,
        n_scenarios: int,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Generate conditional next-day return scenarios.

        Parameters
        ----------
        history:
            Standardized historical excess returns,
            shape [batch, lookback, assets].

        n_scenarios:
            Number of independent scenarios per historical window.

        initial_noise:
            Optional x_T initialization with shape
            [batch, n_scenarios, assets].

        Returns
        -------
        Tensor
            Generated standardized next-day returns with shape:

                [batch, n_scenarios, assets]

        Notes
        -----
        The returned samples remain in MODEL SPACE.

        For our current MVP this means standardized excess returns.
        They must be inverse-transformed before financial evaluation.
        """

        if history.ndim != 3:
            raise ValueError(
                "history must have shape "
                "[batch, lookback, assets]"
            )

        if history.shape[1] != self.lookback:
            raise ValueError(
                "history lookback dimension "
                "does not match model"
            )

        if history.shape[2] != self.n_assets:
            raise ValueError(
                "history asset dimension "
                "does not match model"
            )

        if n_scenarios <= 0:
            raise ValueError(
                "n_scenarios must be positive"
            )

        batch_size = history.shape[0]

        # Encode and route each historical window once.
        condition = (
            self._encode_condition(
                history
            )
        )

        # Each history gets M independently denoised scenarios.
        repeated_condition = (
            condition.repeat_interleave(
                n_scenarios,
                dim=0,
            )
        )

        if initial_noise is None:
            x = torch.randn(
                (
                    batch_size,
                    n_scenarios,
                    self.n_assets,
                ),
                device=history.device,
                dtype=history.dtype,
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

            x = initial_noise.to(
                device=history.device,
                dtype=history.dtype,
            )

        x = x.reshape(
            batch_size
            * n_scenarios,
            self.n_assets,
        )

        for step in reversed(
            range(
                self.diffusion_steps
            )
        ):
            timesteps = torch.full(
                (
                    batch_size
                    * n_scenarios,
                ),
                step,
                device=history.device,
                dtype=torch.long,
            )

            if step > 0:
                step_noise = torch.randn_like(
                    x
                )
            else:
                step_noise = torch.zeros_like(
                    x
                )

            x = self.reverse_step(
                x,
                timesteps,
                repeated_condition,
                noise=step_noise,
            )

        return x.reshape(
            batch_size,
            n_scenarios,
            self.n_assets,
        )
    def training_loss(
        self,
        history: Tensor,
        target: Tensor,
        *,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> DiffusionTrainingOutput:
        """Calculate the standard DDPM epsilon-prediction objective."""

        x_0 = self._prepare_target(
            target
        )

        if history.shape[0] != x_0.shape[0]:
            raise ValueError(
                "history and target batch sizes differ"
            )

        batch_size = x_0.shape[0]

        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.diffusion_steps,
                size=(batch_size,),
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
                    "timesteps must have shape [batch]"
                )

            if (
                torch.any(timesteps < 0)
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
            noise = torch.randn_like(
                x_0
            )

        else:
            noise = noise.to(
                device=x_0.device,
                dtype=x_0.dtype,
            )

            if noise.shape != x_0.shape:
                raise ValueError(
                    "noise must match target shape"
                )

        x_t = self.noise_schedule.q_sample(
            x_0,
            timesteps,
            noise,
        )

        prediction = (
            self.predict_model_output(
                x_t,
                timesteps,
                history,
            )
        )

        training_target = (
            self._make_training_target(
                x_0,
                noise,
                timesteps,
            )
        )

        loss = F.mse_loss(
            prediction,
            training_target,
        )

        return DiffusionTrainingOutput(
            loss=loss,
            prediction=prediction,
            training_target=(
                training_target
            ),
            noisy_target=x_t,
            timesteps=timesteps,
        )