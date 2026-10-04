"""Neural Controlled Differential Equation for proposal Phase 0.

Implements the proposal equation

    dH(t) = f_theta(H(t)) dt
            + g_theta(H(t)) dX(t)

using an augmented control

    X_tilde(t) = [t, X(t)].

The corresponding standard Neural CDE is

    dH(t) = F_theta(H(t)) dX_tilde(t)

with

    F_theta(H) = [f_theta(H), g_theta(H)].

Discrete observations are interpolated with natural cubic splines,
as specified by the proposal.

This module contains reusable model code. "Phase 0" remains an
experimental/research-stage name rather than a source-code folder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    import torchcde
except ImportError as exc:
    raise ImportError(
        "NeuralCDE requires torchcde. "
        "Install with: pip install -e '.[cde]'"
    ) from exc


class CDEVectorField(nn.Module):
    """Vector field implementing [f(H), g(H)].

    For an input path with D channels:

        f(H): R^h -> R^h
        g(H): R^h -> R^(h x D)

    The augmented time channel makes the complete output

        F(H): R^h -> R^(h x (D + 1))

    where the first column multiplies dt.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        drift_hidden_dim: int = 256,
        sensitivity_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        if drift_hidden_dim <= 0:
            raise ValueError(
                "drift_hidden_dim must be positive"
            )

        if sensitivity_hidden_dim <= 0:
            raise ValueError(
                "sensitivity_hidden_dim must be positive"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.drift = nn.Sequential(
            nn.Linear(
                hidden_dim,
                drift_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                drift_hidden_dim,
                hidden_dim,
            ),
        )

        self.sensitivity = nn.Sequential(
            nn.Linear(
                hidden_dim,
                sensitivity_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                sensitivity_hidden_dim,
                hidden_dim * input_dim,
            ),
        )

    def forward(
        self,
        t: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        # This version is autonomous:
        #
        # f_theta(H), g_theta(H)
        #
        # rather than explicitly depending on t.
        del t

        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                "hidden-state dimension does not "
                "match the vector field"
            )

        drift = (
            self.drift(
                hidden
            )
            .unsqueeze(-1)
        )

        sensitivity = (
            self.sensitivity(
                hidden
            )
            .reshape(
                *hidden.shape[:-1],
                self.hidden_dim,
                self.input_dim,
            )
        )

        # First control channel = time.
        #
        # F(H) d[t, X]
        #
        # = f(H) dt + g(H) dX
        return torch.cat(
            (
                drift,
                sensitivity,
            ),
            dim=-1,
        )


class NeuralCDE(nn.Module):
    """Continuous-time encoder for multivariate market paths.

    Input
    -----
    path:
        Either

            [batch, time, input_dim]

        or

            [batch, time, assets, features]

        where assets * features = input_dim.

    Output
    ------
    forward():
        Hidden trajectory

            [batch, time, hidden_dim]

    encode_final():
        Final hidden state

            [batch, hidden_dim]

    Notes
    -----
    A time channel is appended to the observed market path. Natural
    cubic interpolation is then applied to the complete augmented
    control.

    No fallback solver is provided intentionally. If the required
    CDE package is unavailable, the experiment should fail loudly
    rather than silently change its mathematical model.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        drift_hidden_dim: int = 256,
        sensitivity_hidden_dim: int = 256,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-5,
        use_adjoint: bool = True,
        fixed_steps_per_interval: int = 4,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        if rtol <= 0:
            raise ValueError(
                "rtol must be positive"
            )

        if atol <= 0:
            raise ValueError(
                "atol must be positive"
            )

        if fixed_steps_per_interval <= 0:
            raise ValueError(
                "fixed_steps_per_interval must be positive"
            )

        self.input_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.solver = solver
        self.rtol = float(
            rtol
        )
        self.atol = float(
            atol
        )
        self.use_adjoint = bool(
            use_adjoint
        )
        self.fixed_steps_per_interval = int(
            fixed_steps_per_interval
        )

        self.initial_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.vector_field = CDEVectorField(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            drift_hidden_dim=(
                drift_hidden_dim
            ),
            sensitivity_hidden_dim=(
                sensitivity_hidden_dim
            ),
        )

    def _flatten_path(
        self,
        path: Tensor,
    ) -> Tensor:
        if path.ndim == 4:
            batch, length = path.shape[:2]

            path = path.reshape(
                batch,
                length,
                -1,
            )

        elif path.ndim != 3:
            raise ValueError(
                "path must have shape "
                "[batch, time, channels] or "
                "[batch, time, assets, features]"
            )

        if path.shape[-1] != self.input_dim:
            raise ValueError(
                "flattened path dimension does not "
                "match input_dim"
            )

        if path.shape[1] < 2:
            raise ValueError(
                "Neural CDE requires at least "
                "two observations"
            )

        if not torch.isfinite(
            path
        ).all():
            raise ValueError(
                "path contains NaN or infinite values"
            )

        return path

    def _prepare_times(
        self,
        path: Tensor,
        timestamps: Tensor | None,
    ) -> Tensor:
        length = path.shape[1]

        if timestamps is None:
            return torch.linspace(
                0.0,
                1.0,
                steps=length,
                device=path.device,
                dtype=path.dtype,
            )

        if timestamps.ndim != 1:
            raise ValueError(
                "timestamps must be one-dimensional. "
                "Batch-specific timestamp grids are "
                "not supported by this implementation."
            )

        if len(timestamps) != length:
            raise ValueError(
                "timestamps length does not "
                "match path length"
            )

        timestamps = timestamps.to(
            device=path.device,
            dtype=path.dtype,
        )

        if not torch.isfinite(
            timestamps
        ).all():
            raise ValueError(
                "timestamps contain non-finite values"
            )

        differences = (
            timestamps[1:]
            - timestamps[:-1]
        )

        if torch.any(
            differences <= 0
        ):
            raise ValueError(
                "timestamps must be strictly increasing"
            )

        # Preserve relative irregular spacing whilst putting every
        # path on a numerically stable [0, 1] integration interval.
        span = (
            timestamps[-1]
            - timestamps[0]
        )

        return (
            timestamps
            - timestamps[0]
        ) / span

    def build_control(
        self,
        path: Tensor,
        timestamps: Tensor | None = None,
    ):
        """Construct the natural-cubic augmented control.

        Returns
        -------
        control:
            torchcde.CubicSpline instance.

        times:
            Normalized knot times.

        flat_path:
            Flattened original path [B, T, D].
        """

        flat_path = self._flatten_path(
            path
        )

        times = self._prepare_times(
            flat_path,
            timestamps,
        )

        batch_size = (
            flat_path.shape[0]
        )

        time_channel = (
            times
            .view(
                1,
                -1,
                1,
            )
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        augmented_path = torch.cat(
            (
                time_channel,
                flat_path,
            ),
            dim=-1,
        )

        coefficients = (
            torchcde.natural_cubic_coeffs(
                augmented_path,
                t=times,
            )
        )

        control = torchcde.CubicSpline(
            coefficients,
            t=times,
        )

        return (
            control,
            times,
            flat_path,
        )

    def _initial_state(
        self,
        flat_path: Tensor,
    ) -> Tensor:
        first_observation = (
            flat_path[
                :,
                0,
                :,
            ]
        )

        return torch.tanh(
            self.initial_projection(
                first_observation
            )
        )

    def _integrate(
        self,
        *,
        control,
        initial_state: Tensor,
        output_times: Tensor,
    ) -> Tensor:
        solver_kwargs = {}

        if self.solver == "rk4":
            grid_points = (
                control.grid_points
            )

            if grid_points.numel() < 2:
                raise RuntimeError(
                    "RK4 requires at least two "
                    "control grid points"
                )

            intervals = (
                grid_points[1:]
                - grid_points[:-1]
            )

            minimum_interval = (
                intervals.min()
            )

            step_size = float(
                (
                    minimum_interval
                    / self.fixed_steps_per_interval
                )
                .detach()
                .cpu()
            )

            solver_kwargs[
                "options"
            ] = {
                "step_size": step_size,
            }

        trajectory = torchcde.cdeint(
            X=control,
            func=self.vector_field,
            z0=initial_state,
            t=output_times,
            adjoint=self.use_adjoint,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            **solver_kwargs,
        )

        if not torch.isfinite(
            trajectory
        ).all():
            raise RuntimeError(
                "Neural CDE produced non-finite values"
            )

        return trajectory

    def forward(
        self,
        path: Tensor,
        timestamps: Tensor | None = None,
    ) -> Tensor:
        """Return H(t) at every observed historical time."""

        (
            control,
            times,
            flat_path,
        ) = self.build_control(
            path,
            timestamps,
        )

        initial_state = (
            self._initial_state(
                flat_path
            )
        )

        trajectory = self._integrate(
            control=control,
            initial_state=initial_state,
            output_times=times,
        )

        expected_shape = (
            flat_path.shape[0],
            flat_path.shape[1],
            self.hidden_dim,
        )

        if trajectory.shape != expected_shape:
            raise RuntimeError(
                "Unexpected Neural CDE output shape: "
                f"{tuple(trajectory.shape)}; "
                f"expected {expected_shape}"
            )

        return trajectory

    def encode_final(
        self,
        path: Tensor,
        timestamps: Tensor | None = None,
    ) -> Tensor:
        """Return only the final historical hidden state H(T)."""

        (
            control,
            _,
            flat_path,
        ) = self.build_control(
            path,
            timestamps,
        )

        initial_state = (
            self._initial_state(
                flat_path
            )
        )

        endpoints = self._integrate(
            control=control,
            initial_state=initial_state,
            output_times=control.interval,
        )

        final_state = endpoints[
            :,
            -1,
            :,
        ]

        if final_state.shape != (
            flat_path.shape[0],
            self.hidden_dim,
        ):
            raise RuntimeError(
                "Unexpected final CDE state shape"
            )

        return final_state