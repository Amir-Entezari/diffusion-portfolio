"""
Sprint 1 Test Suite: Phase 0 — Dynamic Evidential Router.

Tests for Neural CDE, Dirichlet Head, Phase-Transition Gate,
and the combined EvidentialRouter.

Run: python -m pytest tests/test_phase0.py -v
"""

import sys
import os
import math

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase0_router.neural_cde import (
    NeuralCDE, DriftNetwork, SensitivityNetwork, CDEFunc,
)
from models.phase0_router.dirichlet_head import DirichletHead
from models.phase0_router.phase_transition_gate import (
    PhaseTransitionGate, EvidentialRouter,
)


# ═══════════════════════════════════════════════════════════════════
#  Drift Network Tests
# ═══════════════════════════════════════════════════════════════════

class TestDriftNetwork:

    def test_output_shape(self):
        drift = DriftNetwork(hidden_dim=32, n_layers=3, intermediate_dim=64)
        h = torch.randn(8, 32)
        out = drift(h)
        assert out.shape == (8, 32)

    def test_bounded_output(self):
        """Tanh ensures outputs are in [-1, 1] per element."""
        drift = DriftNetwork(hidden_dim=16, n_layers=2, intermediate_dim=32)
        h = torch.randn(4, 16) * 10  # Large input
        out = drift(h)
        assert (out.abs() <= 1.0 + 1e-6).all()


# ═══════════════════════════════════════════════════════════════════
#  Sensitivity Network Tests
# ═══════════════════════════════════════════════════════════════════

class TestSensitivityNetwork:

    def test_output_shape(self):
        sens = SensitivityNetwork(hidden_dim=32, input_dim=12, n_layers=2,
                                  intermediate_dim=32)
        h = torch.randn(8, 32)
        G = sens(h)
        assert G.shape == (8, 32, 12)


# ═══════════════════════════════════════════════════════════════════
#  Neural CDE Tests
# ═══════════════════════════════════════════════════════════════════

class TestNeuralCDE:

    def test_euler_forward(self):
        """CDE should produce valid hidden trajectory with Euler solver."""
        cde = NeuralCDE(
            input_dim=12, hidden_dim=32,
            drift_n_layers=2, drift_hidden_dim=64,
            sensitivity_n_layers=1, sensitivity_hidden_dim=32,
            use_adjoint=False,
        )
        B, T, D = 4, 10, 12
        features = torch.randn(B, T, D)
        timestamps = torch.linspace(0, 1, T)

        h_traj = cde.forward_with_discrete_path(features, timestamps)
        assert h_traj.shape == (T, B, 32)
        assert not torch.isnan(h_traj).any()

    def test_hidden_state_varies_over_time(self):
        """Hidden states should NOT be constant across time."""
        cde = NeuralCDE(input_dim=6, hidden_dim=16, use_adjoint=False,
                        drift_n_layers=2, drift_hidden_dim=32,
                        sensitivity_n_layers=1, sensitivity_hidden_dim=16)
        features = torch.randn(2, 8, 6)
        timestamps = torch.linspace(0, 1, 8)
        h_traj = cde.forward_with_discrete_path(features, timestamps)
        # First and last hidden state should differ
        diff = (h_traj[0] - h_traj[-1]).abs().mean()
        assert diff > 1e-6

    def test_gradient_flow(self):
        """Gradients should flow through the CDE."""
        cde = NeuralCDE(input_dim=6, hidden_dim=16, use_adjoint=False,
                        drift_n_layers=2, drift_hidden_dim=32,
                        sensitivity_n_layers=1, sensitivity_hidden_dim=16)
        features = torch.randn(2, 5, 6, requires_grad=True)
        timestamps = torch.linspace(0, 1, 5)
        h_traj = cde.forward_with_discrete_path(features, timestamps)
        loss = h_traj.sum()
        loss.backward()
        # Check that CDE parameters have gradients
        for name, param in cde.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ═══════════════════════════════════════════════════════════════════
#  Dirichlet Head Tests
# ═══════════════════════════════════════════════════════════════════

class TestDirichletHead:

    def test_output_shapes(self):
        head = DirichletHead(hidden_dim=32, n_regimes=4)
        h = torch.randn(8, 32)
        out = head(h)
        assert out["evidence"].shape == (8, 4)
        assert out["alpha"].shape == (8, 4)
        assert out["strength"].shape == (8,)
        assert out["probabilities"].shape == (8, 4)
        assert out["uncertainty"].shape == (8,)

    def test_evidence_non_negative(self):
        """Softplus ensures evidence ≥ 0."""
        head = DirichletHead(hidden_dim=32, n_regimes=4)
        h = torch.randn(16, 32) * 5
        out = head(h)
        assert (out["evidence"] >= 0).all()

    def test_alpha_geq_one(self):
        """α = e + 1, so α ≥ 1 always."""
        head = DirichletHead(hidden_dim=32, n_regimes=4)
        h = torch.randn(16, 32) * 5
        out = head(h)
        assert (out["alpha"] >= 1.0 - 1e-6).all()

    def test_probabilities_on_simplex(self):
        """Probabilities must sum to 1."""
        head = DirichletHead(hidden_dim=32, n_regimes=4)
        h = torch.randn(8, 32)
        out = head(h)
        sums = out["probabilities"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_uncertainty_range(self):
        """u(t) = K/S, with S ≥ K, so u ∈ (0, 1]."""
        head = DirichletHead(hidden_dim=32, n_regimes=4)
        h = torch.randn(100, 32)
        out = head(h)
        assert (out["uncertainty"] > 0).all()
        assert (out["uncertainty"] <= 1.0 + 1e-6).all()

    def test_zero_evidence_gives_max_uncertainty(self):
        """When evidence ≈ 0, α ≈ 1, S ≈ K, u ≈ 1."""
        head = DirichletHead(hidden_dim=16, n_regimes=4)
        # Override weights to produce near-zero output
        with torch.no_grad():
            head.evidence_proj.weight.zero_()
            head.evidence_proj.bias.fill_(-10.0)  # Softplus(-10) ≈ 0
        h = torch.zeros(1, 16)
        out = head(h)
        assert out["uncertainty"].item() > 0.99

    def test_evidential_loss_runs(self):
        """Evidential loss should compute without error."""
        head = DirichletHead(hidden_dim=32, n_regimes=4)
        h = torch.randn(8, 32)
        out = head(h)
        target = torch.zeros(8, 4)
        target[:, 0] = 1.0  # All samples in regime 0
        loss = head.compute_evidential_loss(
            out["alpha"], target, lambda_kl=0.01, epoch=10, n_epochs=200
        )
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss.item() >= 0


# ═══════════════════════════════════════════════════════════════════
#  Phase-Transition Gate Tests
# ═══════════════════════════════════════════════════════════════════

class TestPhaseTransitionGate:

    def test_output_shape(self):
        gate = PhaseTransitionGate(n_regimes=4)
        u = torch.tensor([0.2, 0.8])
        p = torch.randn(2, 4).softmax(dim=-1)
        G = gate(u, p)
        assert G.shape == (2, 4)

    def test_gate_sums_to_one(self):
        """G_k must always sum to 1 (convex combination of two simplices)."""
        gate = PhaseTransitionGate(n_regimes=4)
        u = torch.rand(10)
        p = torch.randn(10, 4).softmax(dim=-1)
        G = gate(u, p)
        sums = G.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_low_uncertainty_trusts_prediction(self):
        """When u << τ, G ≈ p (trust the model)."""
        gate = PhaseTransitionGate(n_regimes=4, uncertainty_threshold=0.4,
                                   gate_temperature=100.0)
        u = torch.tensor([0.01])  # Very low uncertainty
        p = torch.tensor([[0.7, 0.2, 0.05, 0.05]])
        G = gate(u, p)
        assert torch.allclose(G, p, atol=0.01)

    def test_high_uncertainty_collapses_to_uniform(self):
        """When u >> τ, G ≈ 1/K (uniform consensus)."""
        gate = PhaseTransitionGate(n_regimes=4, uncertainty_threshold=0.4,
                                   gate_temperature=100.0)
        u = torch.tensor([0.99])  # Very high uncertainty
        p = torch.tensor([[0.7, 0.2, 0.05, 0.05]])
        G = gate(u, p)
        uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        assert torch.allclose(G, uniform, atol=0.01)

    def test_gate_is_differentiable(self):
        """Gate must allow gradient flow for end-to-end training."""
        gate = PhaseTransitionGate(n_regimes=4)
        u = torch.tensor([0.5], requires_grad=True)
        p = torch.tensor([[0.4, 0.3, 0.2, 0.1]], requires_grad=True)
        G = gate(u, p)
        G.sum().backward()
        assert u.grad is not None
        assert p.grad is not None

    def test_regime_confidence_metrics(self):
        gate = PhaseTransitionGate(n_regimes=4)
        G = torch.tensor([[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]])
        metrics = gate.get_regime_confidence(G)
        assert metrics["dominant_regime"][0] == 0
        # Uniform should have higher entropy
        assert metrics["regime_entropy"][1] > metrics["regime_entropy"][0]


# ═══════════════════════════════════════════════════════════════════
#  EvidentialRouter Integration Test
# ═══════════════════════════════════════════════════════════════════

class TestEvidentialRouter:

    def test_full_forward_pass(self):
        """Complete Phase 0 pipeline: features → gate values."""
        router = EvidentialRouter(
            input_dim=6, hidden_dim=16, n_regimes=4,
            uncertainty_threshold=0.4, gate_temperature=10.0,
            use_adjoint=False,
            drift_n_layers=2, drift_hidden_dim=32,
            sensitivity_n_layers=1, sensitivity_hidden_dim=16,
        )
        B, T, D = 4, 8, 6
        features = torch.randn(B, T, D)
        timestamps = torch.linspace(0, 1, T)

        out = router(features, timestamps)

        assert out["hidden_states"].shape == (T, B, 16)
        assert out["gate_values"].shape == (T, B, 4)
        assert out["uncertainty"].shape == (T, B)

        # Gate values must sum to 1
        gate_sums = out["gate_values"].sum(dim=-1)
        assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)

        # No NaNs
        for key, val in out.items():
            assert not torch.isnan(val).any(), f"NaN in {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
