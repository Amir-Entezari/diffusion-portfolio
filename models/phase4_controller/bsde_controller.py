"""
Deep BSDE Portfolio Controller with Huber Friction.

Reference: Proposal §5.7 — Deep BSDE Portfolio Allocation.

Implements the Backward SDE controller:

    dV(t) = −f(t, Y_t, V_t, Z_t, w_t) dt + Z_t^T dW_t

where:
    - V(t): portfolio value process
    - Z_t: hedging strategy (BSDE control)
    - w_t: portfolio weights on the simplex Δ^{N-1}
    - f(·): driver function incorporating transaction costs

The portfolio weights w_t are produced by a policy network:
    w_t = Softmax(g_φ(Y_t, V_t, Z_t, c_unified))

Transaction Cost Model (§5.7.2):
    TC(Δw) = Σ_i c_i · H_δ(Δw_i)

    where H_δ is the Huber function (smooth L1):
        H_δ(x) = x²/(2δ)  if |x| ≤ δ
                  |x| - δ/2  otherwise

    This provides:
        - Differentiability at Δw=0 (unlike absolute value)
        - Linear cost for large trades (realistic market impact)
        - Quadratic cost for small trades (smooth optimization)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class HuberTransactionCost(nn.Module):
    r"""Huber friction model for transaction costs (§5.7.2).

    .. math::
        TC(\Delta w) = \sum_{i=1}^{N} c_i \cdot H_\delta(\Delta w_i)

    Args:
        n_assets: N — number of assets.
        huber_delta: δ — transition point between quadratic and linear.
        cost_per_asset: Per-asset cost coefficients c_i.
            If None, uses uniform cost = 0.001 (10 bps).
    """

    def __init__(
        self,
        n_assets: int = 12,
        huber_delta: float = 0.01,
        cost_per_asset: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta

        if cost_per_asset is not None:
            self.register_buffer("costs", cost_per_asset)
        else:
            self.register_buffer(
                "costs", torch.full((n_assets,), 0.001)
            )

    def forward(self, delta_w: Tensor) -> Tensor:
        r"""Compute transaction costs.

        Args:
            delta_w: Weight changes Δw = w_t − w_{t−1}, shape [..., N].

        Returns:
            Total cost, shape [...].
        """
        huber = F.huber_loss(
            delta_w,
            torch.zeros_like(delta_w),
            reduction="none",
            delta=self.huber_delta,
        )
        # Weight by per-asset costs
        weighted = huber * self.costs
        return weighted.sum(dim=-1)


class PolicyNetwork(nn.Module):
    """Portfolio weight policy network g_φ.

    Maps (Y_t, V_t, Z_t, c_unified) → w_t ∈ Δ^{N-1}

    Args:
        input_dim: Total input dimension.
        n_assets: N — output dimension (portfolio weights).
        hidden_dims: MLP hidden layer dimensions.
        temperature: Softmax temperature for weight sharpening.
    """

    def __init__(
        self,
        input_dim: int,
        n_assets: int = 12,
        hidden_dims: list[int] | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        self.temperature = temperature
        self.n_assets = n_assets

        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.SiLU(),
                nn.Dropout(0.1),
            ])
            in_d = h_d

        layers.append(nn.Linear(in_d, n_assets))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize final layer to near-uniform allocation."""
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        scenario: Tensor,
        value: Tensor,
        z: Tensor,
        c_unified: Tensor,
    ) -> Tensor:
        """Compute portfolio weights.

        Args:
            scenario: Generated scenario Y_t, shape [B, N].
            value: Portfolio value V_t, shape [B, 1].
            z: BSDE control Z_t, shape [B, N].
            c_unified: Conditioning, shape [B, C].

        Returns:
            Portfolio weights w_t, shape [B, N]. On the simplex.
        """
        x = torch.cat([scenario, value, z, c_unified], dim=-1)
        logits = self.net(x) / self.temperature
        return F.softmax(logits, dim=-1)


class ZNetwork(nn.Module):
    """BSDE control Z_t approximation network.

    Approximates the martingale integrand Z_t of the BSDE.

    Args:
        input_dim: Scenario + condition dimension.
        n_assets: Output dimension.
        hidden_dims: MLP hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        n_assets: int = 12,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.SiLU()])
            in_d = h_d
        layers.append(nn.Linear(in_d, n_assets))
        self.net = nn.Sequential(*layers)

    def forward(self, scenario: Tensor, c_unified: Tensor) -> Tensor:
        """Compute Z_t.

        Args:
            scenario: Y_t, shape [B, N].
            c_unified: Conditioning, shape [B, C].

        Returns:
            Z_t, shape [B, N].
        """
        x = torch.cat([scenario, c_unified], dim=-1)
        return self.net(x)


class DeepBSDEController(nn.Module):
    r"""Complete Phase 4: Deep BSDE Portfolio Controller.

    Implements the backward SDE:
        dV(t) = −f(t, Y_t, V_t, Z_t, w_t)dt + Z_t^T dW_t

    with terminal condition:
        V(T) = Φ(Y_T, w_T)  (portfolio terminal value)

    The objective:
        max_φ E[U(V(T))] − λ·TC − γ·Risk

    Args:
        n_assets: N — portfolio dimension.
        condition_dim: c_unified dimension.
        huber_delta: δ for transaction cost Huber function.
        risk_aversion: γ — CVaR risk penalty coefficient.
        cvar_alpha: α — CVaR confidence level.
    """

    def __init__(
        self,
        n_assets: int = 12,
        condition_dim: int = 256,
        huber_delta: float = 0.01,
        risk_aversion: float = 0.5,
        cvar_alpha: float = 0.05,
        policy_hidden: list[int] | None = None,
        z_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.cvar_alpha = cvar_alpha

        # Z network input: scenario (N) + condition (C)
        z_input_dim = n_assets + condition_dim
        self.z_net = ZNetwork(z_input_dim, n_assets, z_hidden)

        # Policy network input: scenario (N) + value (1) + z (N) + condition (C)
        policy_input_dim = n_assets + 1 + n_assets + condition_dim
        self.policy = PolicyNetwork(
            policy_input_dim, n_assets, policy_hidden,
        )

        # Transaction cost
        self.tc_model = HuberTransactionCost(n_assets, huber_delta)

        # Learnable initial portfolio value
        self.V0 = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        scenarios: Tensor,
        c_unified: Tensor,
        dt: float = 1.0 / 252.0,
    ) -> dict[str, Tensor]:
        """Forward pass: simulate BSDE along generated scenarios.

        Args:
            scenarios: Generated paths Y_t, shape [B, T, N].
            c_unified: Conditioning, shape [B, C].
            dt: Time step.

        Returns:
            Dict with weights, values, costs, terminal_value.
        """
        B, T, N = scenarios.shape
        device = scenarios.device

        V = self.V0.expand(B)  # [B]
        prev_w = torch.ones(B, N, device=device) / N  # Equal weight init

        all_weights = []
        all_values = [V]
        all_costs = []

        for t in range(T):
            Y_t = scenarios[:, t, :]  # [B, N]

            # Compute Z_t
            Z_t = self.z_net(Y_t, c_unified)  # [B, N]

            # Compute portfolio weights
            w_t = self.policy(
                Y_t, V.unsqueeze(-1), Z_t, c_unified
            )  # [B, N]

            # Transaction costs
            delta_w = w_t - prev_w
            tc = self.tc_model(delta_w)  # [B]

            # Portfolio return: w_t^T · r_t
            if t < T - 1:
                returns_t = scenarios[:, t + 1, :] - Y_t  # Simple returns
                port_return = (w_t * returns_t).sum(dim=-1)  # [B]
            else:
                port_return = torch.zeros(B, device=device)

            # BSDE update: V_{t+1} = V_t + port_return - tc
            V = V + port_return - tc

            all_weights.append(w_t)
            all_values.append(V)
            all_costs.append(tc)
            prev_w = w_t.detach()  # Detach for next step's TC

        return {
            "weights": torch.stack(all_weights, dim=1),     # [B, T, N]
            "values": torch.stack(all_values, dim=1),        # [B, T+1]
            "costs": torch.stack(all_costs, dim=1),          # [B, T]
            "terminal_value": V,                              # [B]
        }

    def compute_objective(
        self,
        terminal_value: Tensor,
        total_costs: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the BSDE controller objective.

        max E[V(T)] − γ·CVaR_α(−V(T)) − λ·TC

        Args:
            terminal_value: V(T), shape [B].
            total_costs: Sum of all TC, shape [B].

        Returns:
            Dict with loss (to minimize), expected_return, cvar, tc.
        """
        # Expected return (negative for minimization)
        expected_return = terminal_value.mean()

        # CVaR: expected shortfall beyond α-quantile
        sorted_vals, _ = terminal_value.sort()
        n_tail = max(1, int(self.cvar_alpha * len(sorted_vals)))
        cvar = -sorted_vals[:n_tail].mean()  # Negative returns = losses

        # Total cost
        tc_mean = total_costs.mean()

        # Objective: minimize negative return + risk + costs
        loss = -expected_return + self.risk_aversion * cvar + tc_mean

        return {
            "loss": loss,
            "expected_return": expected_return,
            "cvar": cvar,
            "transaction_cost": tc_mean,
        }
