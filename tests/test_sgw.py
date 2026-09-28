"""
Sprint 4 Test Suite: SGW Interface.

Run: python -m pytest tests/test_sgw.py -v
"""

import sys, os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sgw_interface.sliced_gw import SlicedGromovWasserstein, _gw_1d
from models.sgw_interface.sgw_cross_attention import (
    SGWCrossAttention, SGWFusionModule,
)


class TestGW1D:

    def test_zero_distance_identical(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        d = _gw_1d(x, x)
        assert d.item() < 1e-6

    def test_non_negative(self):
        x = torch.randn(10)
        y = torch.randn(10)
        d = _gw_1d(x, y)
        assert d.item() >= -1e-6


class TestSlicedGW:

    def test_output_shapes(self):
        sgw = SlicedGromovWasserstein(
            n_projections=10, projection_dim=16,
            topo_input_dim=32, quantum_input_dim=20,
        )
        T = torch.randn(4, 32)
        v = torch.randn(4, 20)
        dist, transport = sgw(T, v)
        assert dist.shape == (4,)
        assert transport.shape == (4, 16, 16)

    def test_distance_non_negative(self):
        sgw = SlicedGromovWasserstein(
            n_projections=5, projection_dim=8,
            topo_input_dim=16, quantum_input_dim=10,
        )
        T = torch.randn(2, 16)
        v = torch.randn(2, 10)
        dist, _ = sgw(T, v)
        assert (dist >= -1e-6).all()

    def test_gradient_flow(self):
        sgw = SlicedGromovWasserstein(
            n_projections=5, projection_dim=8,
            topo_input_dim=16, quantum_input_dim=10,
        )
        T = torch.randn(2, 16, requires_grad=True)
        v = torch.randn(2, 10, requires_grad=True)
        dist, _ = sgw(T, v)
        dist.sum().backward()
        assert T.grad is not None


class TestSGWCrossAttention:

    def test_output_shape(self):
        attn = SGWCrossAttention(
            topo_dim=32, quantum_dim=20,
            n_heads=4, head_dim=16, output_dim=64,
        )
        T = torch.randn(4, 32)
        v = torch.randn(4, 20)
        c = attn(T, v)
        assert c.shape == (4, 64)

    def test_gradient_flow(self):
        attn = SGWCrossAttention(
            topo_dim=16, quantum_dim=10,
            n_heads=2, head_dim=8, output_dim=32,
        )
        T = torch.randn(2, 16, requires_grad=True)
        v = torch.randn(2, 10, requires_grad=True)
        c = attn(T, v)
        c.sum().backward()
        assert T.grad is not None
        assert v.grad is not None


class TestSGWFusion:

    def test_full_fusion(self):
        fusion = SGWFusionModule(
            topo_dim=32, quantum_dim=20,
            n_projections=5, projection_dim=8,
            n_heads=2, head_dim=8, output_dim=64,
        )
        T = torch.randn(3, 32)
        v = torch.randn(3, 20)
        out = fusion(T, v)
        assert out["c_unified"].shape == (3, 64)
        assert out["sgw_distance"].shape == (3,)
        assert not torch.isnan(out["c_unified"]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
