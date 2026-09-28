"""
Sprint 3 Test Suite: Phase 2 — Quantum Information Geometry.

Run: python -m pytest tests/test_phase2.py -v
"""

import sys, os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase2_quantum.density_matrix import DensityMatrixBuilder
from models.phase2_quantum.bures_metric import (
    bures_distance, bures_distance_squared, quantum_fidelity,
)
from models.phase2_quantum.log_map import RiemannianLogMap
from utils.matrix_ops import batch_trace, symmetrise


def _make_density(N, batch=None):
    """Helper: create valid density matrices."""
    if batch:
        A = torch.randn(batch, N, N)
    else:
        A = torch.randn(N, N)
    cov = A @ A.transpose(-2, -1) + 0.1 * torch.eye(N)
    tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
    return cov / tr.unsqueeze(-1).unsqueeze(-1)


class TestDensityMatrix:

    def test_unit_trace(self):
        builder = DensityMatrixBuilder(n_assets=5, window_size=10)
        returns = torch.randn(2, 30, 5) * 0.01
        rhos = builder(returns)
        traces = batch_trace(rhos.reshape(-1, 5, 5))
        assert torch.allclose(traces, torch.ones_like(traces), atol=1e-5)

    def test_positive_semidefinite(self):
        builder = DensityMatrixBuilder(n_assets=4, window_size=10)
        returns = torch.randn(2, 20, 4) * 0.01
        rhos = builder(returns)
        for rho in rhos.reshape(-1, 4, 4):
            eigvals = torch.linalg.eigvalsh(rho)
            assert (eigvals >= -1e-6).all()

    def test_output_shape(self):
        builder = DensityMatrixBuilder(n_assets=5, window_size=10)
        returns = torch.randn(3, 25, 5)
        rhos = builder(returns)
        assert rhos.shape == (3, 16, 5, 5)  # T_valid = 25-10+1 = 16


class TestBuresMetric:

    def test_zero_distance_same_matrix(self):
        rho = _make_density(4)
        d = bures_distance(rho, rho)
        assert d.item() < 1e-3  # Small numerical residual from nested sqrt

    def test_fidelity_one_same_matrix(self):
        rho = _make_density(4)
        f = quantum_fidelity(rho, rho)
        assert abs(f.item() - 1.0) < 1e-3

    def test_distance_bounded(self):
        """D_B ∈ [0, √2]."""
        rho1 = _make_density(5)
        rho2 = _make_density(5)
        d = bures_distance(rho1, rho2)
        assert d.item() >= -1e-6
        assert d.item() <= np.sqrt(2) + 0.1

    def test_symmetry(self):
        rho1 = _make_density(4)
        rho2 = _make_density(4)
        d12 = bures_distance(rho1, rho2)
        d21 = bures_distance(rho2, rho1)
        assert torch.allclose(d12, d21, atol=1e-4)

    def test_batch_distance(self):
        rho1 = _make_density(4, batch=3)
        rho2 = _make_density(4, batch=3)
        d = bures_distance(rho1, rho2)
        assert d.shape == (3,)

    def test_gradient_flow(self):
        rho1 = _make_density(4).requires_grad_(True)
        rho2 = _make_density(4)
        d = bures_distance_squared(rho1, rho2)
        d.backward()
        assert rho1.grad is not None
        assert not torch.isnan(rho1.grad).any()


class TestLogMap:

    def test_identity_reference_output_shape(self):
        log_map = RiemannianLogMap(n_assets=5, reference_point="identity")
        rho = _make_density(5, batch=3)
        v = log_map(rho)
        assert v.shape == (3, 15)  # N(N+1)/2 = 15

    def test_unflatten_output(self):
        log_map = RiemannianLogMap(n_assets=4, flatten_output=False)
        rho = _make_density(4)
        v = log_map(rho)
        assert v.shape == (4, 4)
        # Tangent vector should be symmetric
        assert torch.allclose(v, v.T, atol=1e-5)

    def test_reference_maps_to_zero(self):
        """Log map of the reference point itself should be zero."""
        N = 4
        log_map = RiemannianLogMap(n_assets=N, flatten_output=False)
        ref = log_map.reference.unsqueeze(0)  # [1, N, N]
        v = log_map(ref)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-4)

    def test_gradient_flow(self):
        log_map = RiemannianLogMap(n_assets=4)
        rho = _make_density(4).unsqueeze(0).requires_grad_(True)
        v = log_map(rho)
        v.sum().backward()
        assert rho.grad is not None


class TestPhase2Integration:

    def test_returns_to_tangent_vector(self):
        """Full: returns → covariance → density → log map → vector."""
        N = 5
        builder = DensityMatrixBuilder(n_assets=N, window_size=15)
        log_map = RiemannianLogMap(n_assets=N)

        returns = torch.randn(2, 40, N) * 0.01
        rhos = builder(returns)  # [2, 26, 5, 5]
        B, T_v = rhos.shape[:2]

        rhos_flat = rhos.reshape(B * T_v, N, N)
        v = log_map(rhos_flat)  # [B*T_v, 15]
        v = v.reshape(B, T_v, -1)
        assert v.shape == (2, 26, 15)
        assert not torch.isnan(v).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
