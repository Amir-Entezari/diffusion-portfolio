"""
Phase-Transition Gate for Dynamic Resource Allocation.

Reference: Proposal §5.2.3 — Differentiable Phase-Transition Gate.

Implements the sigmoid convex-combination gate:

    G_k(t) = [1 − σ(γ(u(t) − τ))] · p_k(t) + σ(γ(u(t) − τ)) · 1/K

where:
    - u(t) ∈ (0, 1]: epistemic uncertainty from DirichletHead
    - τ ∈ (0, 1): critical uncertainty threshold
    - γ ∈ ℝ⁺: sigmoid temperature (steepness)
    - p_k(t): regime probabilities from DirichletHead
    - K: number of regimes

Limit Behaviour (§5.2.3):
    u(t) << τ  →  σ ≈ 0  →  G_k ≈ p_k      (trust the model's prediction)
    u(t) >> τ  →  σ ≈ 1  →  G_k ≈ 1/K       (uniform consensus: "I don't know")

This gate dynamically allocates computational resources:
    - During calm markets (low u): Phase 1 TDA and Phase 2 Quantum
      operate with full discriminative power.
    - During Black Swan events (high u): the system collapses to uniform
      consensus, preventing catastrophic hallucination from overconfident
      but wrong regime assignments.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class PhaseTransitionGate(nn.Module):
    r"""Differentiable phase-transition gate for uncertainty-aware routing.

    .. math::
        G_k(t) = \left[1 - \sigma\bigl(\gamma(u(t) - \tau)\bigr)\right] \cdot p_k(t)
               + \sigma\bigl(\gamma(u(t) - \tau)\bigr) \cdot \frac{1}{K}

    The gate provides a smooth, differentiable interpolation between
    the model's confident regime prediction and a safe uniform fallback.

    Args:
        n_regimes: K — number of market regime classes.
        uncertainty_threshold: τ — critical uncertainty threshold.
        gate_temperature: γ — sigmoid steepness parameter.

    Example:
        >>> gate = PhaseTransitionGate(n_regimes=4, uncertainty_threshold=0.4)
        >>> # Low uncertainty → trust predictions
        >>> G_calm = gate(u=torch.tensor([0.1]), p=torch.tensor([[0.7, 0.2, 0.05, 0.05]]))
        >>> # G_calm ≈ [0.7, 0.2, 0.05, 0.05]  (close to p)
        >>> # High uncertainty → uniform consensus
        >>> G_crash = gate(u=torch.tensor([0.9]), p=torch.tensor([[0.7, 0.2, 0.05, 0.05]]))
        >>> # G_crash ≈ [0.25, 0.25, 0.25, 0.25]  (close to 1/K)
    """

    def __init__(
        self,
        n_regimes: int = 4,
        uncertainty_threshold: float = 0.4,
        gate_temperature: float = 10.0,
    ) -> None:
        super().__init__()

        self.n_regimes = n_regimes
        self.uncertainty_threshold = uncertainty_threshold
        self.gate_temperature = gate_temperature

        # Store as buffers (not parameters — these are hyperparameters)
        self.register_buffer(
            "tau", torch.tensor(uncertainty_threshold, dtype=torch.float32)
        )
        self.register_buffer(
            "gamma", torch.tensor(gate_temperature, dtype=torch.float32)
        )
        self.register_buffer(
            "uniform", torch.ones(n_regimes, dtype=torch.float32) / n_regimes
        )

        logger.info(
            f"PhaseTransitionGate: K={n_regimes}, τ={uncertainty_threshold}, "
            f"γ={gate_temperature}"
        )

    def forward(
        self,
        uncertainty: Tensor,
        probabilities: Tensor,
    ) -> Tensor:
        r"""Compute the gated regime allocation.

        Args:
            uncertainty: u(t), shape [...].
                Epistemic uncertainty from DirichletHead.
            probabilities: p(t), shape [..., K].
                Regime probabilities from DirichletHead.

        Returns:
            G(t): Gated regime allocation, shape [..., K].
                Always sums to 1 along the last dimension.
                Smoothly interpolates between p(t) and 1/K.
        """
        # σ(γ(u − τ)): collapse factor
        # High u → σ → 1 → collapse to uniform
        # Low u  → σ → 0 → trust predictions
        collapse_factor = torch.sigmoid(
            self.gamma * (uncertainty.unsqueeze(-1) - self.tau)
        )  # [..., 1]

        # Expand uniform to match batch shape
        uniform = self.uniform.expand_as(probabilities)

        # Convex combination
        G = (1.0 - collapse_factor) * probabilities + collapse_factor * uniform

        return G

    def get_regime_confidence(self, gate_output: Tensor) -> Dict[str, Tensor]:
        """Extract interpretable metrics from the gate output.

        Args:
            gate_output: G(t), shape [..., K].

        Returns:
            Dictionary with:
                - ``dominant_regime``: Index of the most confident regime.
                - ``regime_entropy``: Shannon entropy of G(t).
                    Low entropy → concentrated on one regime.
                    High entropy → uniform (uncertain).
                - ``max_gate_value``: Maximum G_k value.
        """
        dominant_regime = gate_output.argmax(dim=-1)

        # Shannon entropy: H = -Σ G_k log G_k
        eps = 1e-10
        entropy = -(gate_output * torch.log(gate_output + eps)).sum(dim=-1)

        max_value = gate_output.max(dim=-1).values

        return {
            "dominant_regime": dominant_regime,
            "regime_entropy": entropy,
            "max_gate_value": max_value,
        }


class EvidentialRouter(nn.Module):
    """Complete Phase 0 Router: Neural CDE → Dirichlet → Gate.

    Combines all Phase 0 components into a single module that takes
    raw features and produces uncertainty-aware regime gate values.

    This is the top-level Phase 0 module used by the rest of the pipeline.

    Args:
        input_dim: N·F — flattened input dimension.
        hidden_dim: h — CDE hidden state dimension.
        n_regimes: K — number of market regimes.
        uncertainty_threshold: τ — gate threshold.
        gate_temperature: γ — gate steepness.
        solver: ODE solver for the Neural CDE.
        use_adjoint: Use O(1) memory adjoint method.
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 256,
        n_regimes: int = 4,
        uncertainty_threshold: float = 0.4,
        gate_temperature: float = 10.0,
        solver: str = "dopri5",
        use_adjoint: bool = True,
        drift_n_layers: int = 3,
        drift_hidden_dim: int = 512,
        sensitivity_n_layers: int = 2,
        sensitivity_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        from models.phase0_router.neural_cde import NeuralCDE
        from models.phase0_router.dirichlet_head import DirichletHead

        self.cde = NeuralCDE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            drift_n_layers=drift_n_layers,
            drift_hidden_dim=drift_hidden_dim,
            sensitivity_n_layers=sensitivity_n_layers,
            sensitivity_hidden_dim=sensitivity_hidden_dim,
            solver=solver,
            use_adjoint=use_adjoint,
        )
        self.dirichlet = DirichletHead(
            hidden_dim=hidden_dim,
            n_regimes=n_regimes,
        )
        self.gate = PhaseTransitionGate(
            n_regimes=n_regimes,
            uncertainty_threshold=uncertainty_threshold,
            gate_temperature=gate_temperature,
        )

    def forward(
        self,
        features: Tensor,
        timestamps: Tensor,
    ) -> Dict[str, Tensor]:
        """Full Phase 0 forward pass.

        Args:
            features: Input features, shape [B, T, N, F] or [B, T, input_dim].
            timestamps: Time points, shape [T] or [B, T].

        Returns:
            Dictionary with:
                - ``hidden_states``: H(t), shape [T, B, h].
                - ``evidence``: e(t), shape [T, B, K].
                - ``alpha``: α(t), shape [T, B, K].
                - ``probabilities``: p(t), shape [T, B, K].
                - ``uncertainty``: u(t), shape [T, B].
                - ``gate_values``: G(t), shape [T, B, K].
        """
        # 1. Solve Neural CDE
        h_trajectory = self.cde.forward_with_discrete_path(
            features, timestamps
        )  # [T, B, h]

        T, B, h = h_trajectory.shape

        # 2. Apply Dirichlet head at each time step
        h_flat = h_trajectory.reshape(T * B, h)
        dirichlet_out = self.dirichlet(h_flat)

        K = self.gate.n_regimes

        # Reshape back to [T, B, ...]
        evidence = dirichlet_out["evidence"].reshape(T, B, K)
        alpha = dirichlet_out["alpha"].reshape(T, B, K)
        probabilities = dirichlet_out["probabilities"].reshape(T, B, K)
        uncertainty = dirichlet_out["uncertainty"].reshape(T, B)

        # 3. Apply phase-transition gate
        gate_values = self.gate(uncertainty, probabilities)  # [T, B, K]

        return {
            "hidden_states": h_trajectory,
            "evidence": evidence,
            "alpha": alpha,
            "probabilities": probabilities,
            "uncertainty": uncertainty,
            "gate_values": gate_values,
        }
