"""
Sprint 6 Test Suite: Phase 4 — Deep BSDE Controller.

Run: python -m pytest tests/test_phase4.py -v
"""

import sys, os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase4_controller.bsde_controller import (
    DeepBSDEController, PolicyNetwork, ZNetwork, HuberTransactionCost,
)


class TestHuberTransactionCost:

    def test_zero_change_zero_cost(self):
        tc = HuberTransactionCost(n_assets=5, huber_delta=0.01)
        delta_w = torch.zeros(4, 5)
        cost = tc(delta_w)
        assert torch.allclose(cost, torch.zeros(4), atol=1e-7)

    def test_positive_cost(self):
        tc = HuberTransactionCost(n_assets=5, huber_delta=0.01)
        delta_w = torch.randn(4, 5) * 0.1
        cost = tc(delta_w)
        assert (cost >= 0).all()

    def test_symmetry(self):
        """Cost should be the same for buy and sell."""
        tc = HuberTransactionCost(n_assets=3, huber_delta=0.01)
        delta_w = torch.tensor([[0.05, -0.03, 0.01]])
        cost_pos = tc(delta_w)
        cost_neg = tc(-delta_w)
        assert torch.allclose(cost_pos, cost_neg, atol=1e-6)

    def test_huber_quadratic_regime(self):
        """For |Δw| < δ, cost should be quadratic (proportional to Δw²)."""
        tc = HuberTransactionCost(n_assets=1, huber_delta=0.1)
        dw1 = torch.tensor([[0.01]])
        dw2 = torch.tensor([[0.02]])
        c1 = tc(dw1)
        c2 = tc(dw2)
        # In quadratic regime: cost ∝ Δw², so c2/c1 ≈ 4
        ratio = c2 / c1
        assert abs(ratio.item() - 4.0) < 0.5

    def test_gradient_flow(self):
        tc = HuberTransactionCost(n_assets=5)
        delta_w = torch.randn(2, 5, requires_grad=True)
        cost = tc(delta_w).sum()
        cost.backward()
        assert delta_w.grad is not None


class TestPolicyNetwork:

    def test_output_on_simplex(self):
        """Weights must sum to 1 (softmax output)."""
        policy = PolicyNetwork(input_dim=50, n_assets=5)
        x = torch.randn(4, 12)  # scenario
        v = torch.randn(4, 1)   # value
        z = torch.randn(4, 12)  # z
        c = torch.randn(4, 25)  # condition
        # Adjust input_dim: 12 + 1 + 12 + 25 = 50
        w = policy(x, v, z, c)
        assert w.shape == (4, 5)
        sums = w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_weights_non_negative(self):
        """Softmax ensures w_i ≥ 0."""
        policy = PolicyNetwork(input_dim=50, n_assets=5)
        w = policy(torch.randn(4, 12), torch.randn(4, 1),
                    torch.randn(4, 12), torch.randn(4, 25))
        assert (w >= 0).all()

    def test_initial_near_uniform(self):
        """With zero-init final layer, initial weights should be near 1/N."""
        policy = PolicyNetwork(input_dim=20, n_assets=4)
        w = policy(torch.zeros(1, 5), torch.zeros(1, 1),
                    torch.zeros(1, 5), torch.zeros(1, 9))
        expected = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        assert torch.allclose(w, expected, atol=0.01)


class TestZNetwork:

    def test_output_shape(self):
        z_net = ZNetwork(input_dim=20, n_assets=5)
        y = torch.randn(4, 12)
        c = torch.randn(4, 8)
        z = z_net(y, c)
        assert z.shape == (4, 5)


class TestDeepBSDEController:

    @pytest.fixture
    def controller(self):
        return DeepBSDEController(
            n_assets=5, condition_dim=16,
            huber_delta=0.01, risk_aversion=0.5, cvar_alpha=0.05,
            policy_hidden=[32, 32], z_hidden=[32],
        )

    def test_forward_shapes(self, controller):
        scenarios = torch.randn(4, 10, 5)  # [B, T, N]
        c = torch.randn(4, 16)
        out = controller(scenarios, c)
        assert out["weights"].shape == (4, 10, 5)
        assert out["values"].shape == (4, 11)   # T+1 values
        assert out["costs"].shape == (4, 10)
        assert out["terminal_value"].shape == (4,)

    def test_weights_on_simplex(self, controller):
        scenarios = torch.randn(4, 5, 5)
        c = torch.randn(4, 16)
        out = controller(scenarios, c)
        sums = out["weights"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_objective_gradient(self, controller):
        scenarios = torch.randn(4, 5, 5)
        c = torch.randn(4, 16)
        out = controller(scenarios, c)
        obj = controller.compute_objective(
            out["terminal_value"], out["costs"].sum(dim=-1)
        )
        obj["loss"].backward()
        grad_count = sum(1 for p in controller.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_costs_non_negative(self, controller):
        scenarios = torch.randn(3, 8, 5)
        c = torch.randn(3, 16)
        out = controller(scenarios, c)
        assert (out["costs"] >= -1e-6).all()

    def test_cvar_computation(self, controller):
        """CVaR should be computable without error."""
        terminal = torch.randn(20)
        costs = torch.rand(20) * 0.01
        obj = controller.compute_objective(terminal, costs)
        assert not torch.isnan(obj["cvar"])
        assert not torch.isnan(obj["loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
