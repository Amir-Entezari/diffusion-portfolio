"""
Neural Controlled Differential Equation (Neural CDE) Core.

Reference: Proposal §5.2.1 — Dynamic Evidential Router via Neural CDE.

Implements the continuous-time hidden state evolution:

    dH(t) = f_θ(H(t)) dt + g_θ(H(t)) dX(t)                    (Eq. §5.2.1)

where:
    - H(t) ∈ ℝ^h is the latent hidden state at time t
    - f_θ: ℝ^h → ℝ^h is the autonomous drift network (3-layer MLP, Tanh)
    - g_θ: ℝ^h → ℝ^{h × (N·F)} is the state-dependent sensitivity matrix
    - dX(t) is the natural cubic spline derivative of the input path

The CDE is solved via the "controlled ODE trick" from Kidger et al. (2020):
    dH(t) = [f_θ(H(t)) + g_θ(H(t)) · X'(t)] dt

This reformulation converts the CDE into a standard ODE, enabling the use
of torchdiffeq's adaptive solvers (Dormand-Prince 5(4)) with the adjoint
sensitivity method for O(1) memory backpropagation (§8.2.1).

Memory Advantage (§8.2.1):
    Standard BPTT: O(L · N) memory (stores all intermediate states)
    Adjoint method: O(1) memory (recomputes forward pass during backward)
    For L=1000 steps and N=500 assets, this is the difference between
    OOM crash and feasible training.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Conditional import: torchdiffeq provides ODE solvers
try:
    import torchdiffeq
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    logger.warning(
        "torchdiffeq not installed. Neural CDE will use manual Euler stepping. "
        "Install via: pip install torchdiffeq"
    )


class DriftNetwork(nn.Module):
    r"""Autonomous drift function f_θ(H(t)).

    A multi-layer MLP with Tanh activation, mapping:
        f_θ: ℝ^h → ℝ^h

    Tanh is chosen (not ReLU) because it is:
        1. Bounded: ensures the ODE drift is Lipschitz-continuous,
           guaranteeing existence and uniqueness of the CDE solution.
        2. Smooth: provides well-behaved gradients for the adjoint method.

    Args:
        hidden_dim: h — dimension of the CDE hidden state.
        n_layers: Number of hidden layers (default: 3 per §5.2.1).
        intermediate_dim: Width of hidden layers.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 3,
        intermediate_dim: int = 512,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = hidden_dim
        for i in range(n_layers):
            out_dim = intermediate_dim if i < n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        # Initialise with small weights for stability
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialisation for bounded activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)

    def forward(self, h: Tensor) -> Tensor:
        """Compute the autonomous drift.

        Args:
            h: Hidden state, shape [..., h].

        Returns:
            Drift vector f_θ(H), shape [..., h].
        """
        return self.net(h)


class SensitivityNetwork(nn.Module):
    r"""State-dependent sensitivity matrix g_θ(H(t)).

    Maps the hidden state to a matrix that modulates how the input
    path derivative dX(t) influences the hidden state evolution:

        g_θ: ℝ^h → ℝ^{h × d_input}

    where d_input = N × F (flattened asset × feature dimension).

    The output matrix g_θ(H(t)) ∈ ℝ^{h × d_input} is contracted with
    the path derivative X'(t) ∈ ℝ^{d_input} to produce the controlled
    component g_θ(H(t)) · X'(t) ∈ ℝ^h.

    Args:
        hidden_dim: h — CDE hidden state dimension.
        input_dim: d_input = N·F — flattened input dimension.
        n_layers: Number of hidden layers.
        intermediate_dim: Width of hidden layers.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        input_dim: int = 36,
        n_layers: int = 2,
        intermediate_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        layers = []
        in_dim = hidden_dim
        for i in range(n_layers):
            out_dim = intermediate_dim if i < n_layers - 1 else hidden_dim * input_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Small initialisation for the sensitivity matrix
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, h: Tensor) -> Tensor:
        """Compute the sensitivity matrix.

        Args:
            h: Hidden state, shape [B, h] or [..., h].

        Returns:
            Sensitivity matrix g_θ(H), shape [..., h, d_input].
        """
        batch_shape = h.shape[:-1]
        flat = self.net(h)  # [..., h * d_input]
        return flat.reshape(*batch_shape, self.hidden_dim, self.input_dim)


class CDEFunc(nn.Module):
    """ODE function for the controlled ODE trick.

    Wraps drift f_θ and sensitivity g_θ into a single callable for
    torchdiffeq.  At each time t, the ODE right-hand side is:

        dH/dt = f_θ(H(t)) + g_θ(H(t)) · X'(t)

    where X'(t) = dX/dt is the path derivative evaluated at time t
    via spline interpolation.

    The path derivative function is set externally before integration
    via ``set_path_derivative_fn``.
    """

    def __init__(
        self,
        drift: DriftNetwork,
        sensitivity: SensitivityNetwork,
    ) -> None:
        super().__init__()
        self.drift = drift
        self.sensitivity = sensitivity
        self._path_deriv_fn: Optional[Callable[[Tensor], Tensor]] = None

    def set_path_derivative_fn(
        self,
        fn: Callable[[Tensor], Tensor],
    ) -> None:
        """Set the function that computes dX(t)/dt.

        Args:
            fn: Callable mapping time scalar t → path derivative X'(t).
                Returns shape [B, d_input].
        """
        self._path_deriv_fn = fn

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        """ODE right-hand side: f_θ(H) + g_θ(H) · X'(t).

        Args:
            t: Current time (scalar).
            h: Current hidden state, shape [B, h].

        Returns:
            dH/dt, shape [B, h].
        """
        # Autonomous drift: f_θ(H)
        drift_term = self.drift(h)  # [B, h]

        # Controlled term: g_θ(H) · X'(t)
        if self._path_deriv_fn is not None:
            dX_dt = self._path_deriv_fn(t)  # [B, d_input]
            G = self.sensitivity(h)  # [B, h, d_input]

            # Matrix-vector product: g_θ(H) @ X'(t)
            # [B, h, d_input] @ [B, d_input, 1] → [B, h, 1] → [B, h]
            controlled_term = torch.bmm(G, dX_dt.unsqueeze(-1)).squeeze(-1)
        else:
            controlled_term = torch.zeros_like(drift_term)

        return drift_term + controlled_term


class NeuralCDE(nn.Module):
    r"""Neural Controlled Differential Equation (§5.2.1).

    Implements the complete Neural CDE:

    .. math::
        dH(t) = f_\theta(H(t)) \, dt + g_\theta(H(t)) \, dX(t)

    via the controlled ODE trick, solved with torchdiffeq's adaptive
    Dormand-Prince 5(4) solver using the adjoint method for O(1) memory.

    Input Pipeline:
        1. Raw features X(t) are spline-interpolated externally.
        2. The spline derivative function dX/dt is passed to this module.
        3. The CDE is integrated over [t_0, t_1, ..., t_T].
        4. Output: hidden states H(t) at all evaluation times.

    Args:
        input_dim: N·F — flattened asset × feature dimension.
        hidden_dim: h — CDE latent state dimension.
        drift_n_layers: Number of layers in the drift network f_θ.
        drift_hidden_dim: Width of drift layers.
        sensitivity_n_layers: Number of layers in g_θ.
        sensitivity_hidden_dim: Width of sensitivity layers.
        solver: ODE solver name (default: 'dopri5').
        rtol: Relative tolerance for adaptive stepping.
        atol: Absolute tolerance for adaptive stepping.
        use_adjoint: If True, use O(1) memory adjoint method.

    Example:
        >>> cde = NeuralCDE(input_dim=36, hidden_dim=256)
        >>> # Assume path_deriv_fn maps time → [B, 36]
        >>> h_trajectory = cde(h0, eval_times, path_deriv_fn)
        >>> h_trajectory.shape  # [T, B, 256]
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 256,
        drift_n_layers: int = 3,
        drift_hidden_dim: int = 512,
        sensitivity_n_layers: int = 2,
        sensitivity_hidden_dim: int = 256,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-5,
        use_adjoint: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint and HAS_TORCHDIFFEQ

        # Learnable initial hidden state
        self.h0_net = nn.Linear(input_dim, hidden_dim)

        # CDE function (drift + sensitivity)
        self.drift = DriftNetwork(
            hidden_dim=hidden_dim,
            n_layers=drift_n_layers,
            intermediate_dim=drift_hidden_dim,
        )
        self.sensitivity = SensitivityNetwork(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            n_layers=sensitivity_n_layers,
            intermediate_dim=sensitivity_hidden_dim,
        )
        self.cde_func = CDEFunc(self.drift, self.sensitivity)

        logger.info(
            f"NeuralCDE: input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"solver={solver}, adjoint={self.use_adjoint}"
        )

    def _compute_initial_state(self, x0: Tensor) -> Tensor:
        """Compute the initial hidden state from the first observation.

        Args:
            x0: First observation X(t₀), shape [B, N, F] or [B, input_dim].

        Returns:
            Initial hidden state H(t₀), shape [B, h].
        """
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)  # Flatten to [B, N·F]
        return torch.tanh(self.h0_net(x0))

    def forward(
        self,
        x0: Tensor,
        eval_times: Tensor,
        path_derivative_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Solve the Neural CDE over the given time interval.

        Args:
            x0: Initial observation X(t₀), shape [B, N, F] or [B, input_dim].
            eval_times: Times at which to evaluate H(t), shape [T_eval].
                Must be monotonically increasing, starting from t₀.
            path_derivative_fn: Function mapping time t (scalar) to
                the path derivative dX(t)/dt, shape [B, input_dim].

        Returns:
            Hidden state trajectory H(t), shape [T_eval, B, h].
        """
        h0 = self._compute_initial_state(x0)  # [B, h]

        # Set the path derivative function on the CDE func
        self.cde_func.set_path_derivative_fn(path_derivative_fn)

        if HAS_TORCHDIFFEQ:
            # Solve with adaptive ODE solver
            solve_fn = odeint_adjoint if self.use_adjoint else odeint
            h_trajectory = solve_fn(
                self.cde_func,
                h0,
                eval_times,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
            )  # [T_eval, B, h]
        else:
            # Fallback: manual Euler stepping
            h_trajectory = self._euler_solve(h0, eval_times)

        return h_trajectory

    def _euler_solve(
        self,
        h0: Tensor,
        eval_times: Tensor,
    ) -> Tensor:
        """Fallback Euler solver when torchdiffeq is unavailable.

        Args:
            h0: Initial state, shape [B, h].
            eval_times: Evaluation times, shape [T].

        Returns:
            Trajectory, shape [T, B, h].
        """
        T = len(eval_times)
        trajectory = [h0]
        h = h0

        for i in range(1, T):
            dt = eval_times[i] - eval_times[i - 1]
            dh_dt = self.cde_func(eval_times[i - 1], h)
            h = h + dt * dh_dt
            trajectory.append(h)

        return torch.stack(trajectory, dim=0)

    def forward_with_discrete_path(
        self,
        features: Tensor,
        timestamps: Tensor,
    ) -> Tensor:
        """Convenience method: integrate CDE from discrete feature tensor.

        Internally constructs a linear interpolation of the path derivative
        (suitable when spline coefficients are not precomputed).

        Args:
            features: Feature tensor, shape [B, T, N, F] or [B, T, input_dim].
            timestamps: Time points, shape [B, T] or [T].

        Returns:
            Hidden trajectory, shape [T, B, h].
        """
        B = features.shape[0]
        T = features.shape[1]

        # Flatten features
        if features.dim() == 4:
            flat_features = features.reshape(B, T, -1)  # [B, T, N·F]
        else:
            flat_features = features

        # Ensure timestamps is 1D for eval_times
        if timestamps.dim() == 2:
            eval_times = timestamps[0]  # Assume same across batch
        else:
            eval_times = timestamps

        # Compute path derivative via finite differences
        # dX/dt ≈ (X(t+dt) - X(t)) / dt
        path_derivs = torch.zeros_like(flat_features)  # [B, T, d]
        for t_idx in range(1, T):
            dt = eval_times[t_idx] - eval_times[t_idx - 1]
            if dt > 0:
                path_derivs[:, t_idx] = (
                    flat_features[:, t_idx] - flat_features[:, t_idx - 1]
                ) / dt
            else:
                path_derivs[:, t_idx] = path_derivs[:, t_idx - 1]

        # Create path derivative function (piecewise-constant interpolation)
        def path_deriv_fn(t: Tensor) -> Tensor:
            # Find nearest time index
            t_val = t.item() if t.dim() == 0 else t
            idx = torch.searchsorted(eval_times, t_val, right=True) - 1
            idx = max(0, min(idx, T - 1))
            return path_derivs[:, idx]  # [B, d]

        # Solve CDE
        x0 = flat_features[:, 0]  # [B, d]
        return self.forward(x0, eval_times, path_deriv_fn)
