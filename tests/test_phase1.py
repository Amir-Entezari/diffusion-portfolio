"""
Sprint 2 Test Suite: Phase 1 — TDA Temporal Topology Extraction.

Run: python -m pytest tests/test_phase1.py -v
"""

import sys, os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase1_topology.takens_embedding import TakensEmbedding
from models.phase1_topology.vietoris_rips import VietorisRipsFiltration
from models.phase1_topology.persistence_landscape import (
    PersistenceLandscape, PersistenceLandscapeLayer,
)


class TestTakensEmbedding:

    def test_output_shape(self):
        emb = TakensEmbedding(embedding_dim=5, time_delay=2)
        H = torch.randn(4, 30, 16)  # [B, T, h]
        Y = emb(H)
        # T_valid = 30 - (5-1)*2 = 22
        assert Y.shape == (4, 22, 5, 16)

    def test_delay_copies_correct(self):
        """Each delay copy should be shifted by τ."""
        emb = TakensEmbedding(embedding_dim=3, time_delay=2)
        H = torch.arange(20).float().unsqueeze(0).unsqueeze(-1)  # [1,20,1]
        Y = emb(H)
        # Y[0, t, 0] should be H[0, t+4]  (offset = (m-1)*τ = 4)
        # Y[0, t, 1] should be H[0, t+2]
        # Y[0, t, 2] should be H[0, t+0]
        assert Y[0, 0, 0, 0].item() == 4.0   # H[t=4]
        assert Y[0, 0, 1, 0].item() == 2.0   # H[t=2]
        assert Y[0, 0, 2, 0].item() == 0.0   # H[t=0]

    def test_insufficient_timesteps_raises(self):
        emb = TakensEmbedding(embedding_dim=10, time_delay=5)
        H = torch.randn(1, 10, 8)  # Need > (10-1)*5 = 45
        with pytest.raises(ValueError):
            emb(H)

    def test_auto_delay_estimation(self):
        """Auto delay should return a positive integer."""
        emb = TakensEmbedding(embedding_dim=5, time_delay_max_lag=20)
        # Sine wave has clear periodicity
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        H = torch.tensor(signal).float().unsqueeze(0).unsqueeze(-1)
        tau = emb.estimate_delay(H)
        assert isinstance(tau, int)
        assert tau >= 1


class TestVietorisRips:

    def test_persistence_diagrams_structure(self):
        vr = VietorisRipsFiltration(max_edge_length=3.0, max_homology_dim=1)
        # Circle-like point cloud (should have β₁ = 1)
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        points = np.column_stack([np.cos(theta), np.sin(theta)])
        diagrams = vr.compute_persistence(points)
        assert 0 in diagrams
        assert 1 in diagrams
        assert diagrams[0].shape[1] == 2  # (birth, death) pairs

    def test_betti_numbers_nonneg(self):
        vr = VietorisRipsFiltration(max_edge_length=2.0, max_homology_dim=1)
        points = np.random.randn(15, 3)
        diagrams = vr.compute_persistence(points)
        betti = vr.extract_betti_numbers(diagrams, epsilon=1.0)
        for k, val in betti.items():
            assert val >= 0

    def test_batch_persistence(self):
        vr = VietorisRipsFiltration(max_edge_length=2.0, max_homology_dim=1)
        clouds = np.random.randn(3, 10, 4)
        results = vr.compute_persistence_batch(clouds)
        assert len(results) == 3


class TestPersistenceLandscape:

    def test_empty_diagram(self):
        pl = PersistenceLandscape(n_layers=3, n_points=50)
        diagram = np.zeros((0, 2), dtype=np.float32)
        landscape = pl.compute(diagram)
        assert landscape.shape == (3, 50)
        assert (landscape == 0).all()

    def test_single_pair(self):
        pl = PersistenceLandscape(n_layers=3, n_points=100)
        diagram = np.array([[0.0, 1.0]], dtype=np.float32)
        landscape = pl.compute(diagram, epsilon_range=(0, 1))
        # Layer 0 should have a tent peaking at ε=0.5, height=0.5
        assert landscape[0].max() > 0
        # Layers 1+ should be zero (only 1 pair)
        assert (landscape[1] == 0).all()

    def test_tent_peak_location(self):
        pl = PersistenceLandscape(n_layers=1, n_points=1001)
        diagram = np.array([[0.2, 0.8]], dtype=np.float32)
        landscape = pl.compute(diagram, epsilon_range=(0, 1))
        # Peak should be at ε = (0.2+0.8)/2 = 0.5, height = (0.8-0.2)/2 = 0.3
        peak_idx = np.argmax(landscape[0])
        peak_eps = peak_idx / 1000.0
        assert abs(peak_eps - 0.5) < 0.01
        assert abs(landscape[0].max() - 0.3) < 0.01

    def test_landscape_is_piecewise_linear(self):
        """Tent functions are piecewise linear — second differences should be ~0."""
        pl = PersistenceLandscape(n_layers=1, n_points=200)
        diagram = np.array([[0.1, 0.9]], dtype=np.float32)
        landscape = pl.compute(diagram, epsilon_range=(0, 1))
        second_diff = np.diff(landscape[0], n=2)
        # Most second differences should be near zero (linear segments)
        n_nonzero = np.sum(np.abs(second_diff) > 0.01)
        assert n_nonzero <= 5  # Only at the peak and endpoints

    def test_multi_dim_concatenation(self):
        pl = PersistenceLandscape(n_layers=3, n_points=50)
        diagrams = {
            0: np.array([[0.0, 0.5], [0.1, 0.3]], dtype=np.float32),
            1: np.array([[0.2, 0.8]], dtype=np.float32),
        }
        result = pl.compute_multi_dim(diagrams)
        assert result.shape == (6, 50)  # 2 dims × 3 layers


class TestPersistenceLandscapeLayer:

    def test_forward_shape(self):
        layer = PersistenceLandscapeLayer(
            n_homology_dims=2, n_layers=3, n_points=50, output_dim=64,
        )
        diagrams = [
            {0: np.array([[0.0, 0.5]], dtype=np.float32),
             1: np.array([[0.2, 0.8]], dtype=np.float32)},
            {0: np.array([[0.1, 0.6]], dtype=np.float32),
             1: np.zeros((0, 2), dtype=np.float32)},
        ]
        out = layer(diagrams)
        assert out.shape == (2, 64)

    def test_gradient_flow(self):
        """Gradients should flow through the projection layer."""
        layer = PersistenceLandscapeLayer(
            n_homology_dims=1, n_layers=2, n_points=20, output_dim=32,
        )
        diagrams = [
            {0: np.array([[0.0, 1.0]], dtype=np.float32)},
        ]
        out = layer(diagrams)
        loss = out.sum()
        loss.backward()
        for p in layer.projection.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestPhase1Integration:

    def test_takens_to_landscape(self):
        """Full pipeline: H(t) → Takens → VR → Landscape."""
        # 1. Simulate hidden states
        H = torch.randn(2, 50, 8)  # [B, T, h]

        # 2. Takens embedding
        emb = TakensEmbedding(embedding_dim=5, time_delay=2)
        Y = emb(H)  # [2, T_valid, 5, 8]

        # 3. VR filtration on first sample, first time step
        vr = VietorisRipsFiltration(max_edge_length=3.0, max_homology_dim=1)
        point_cloud = Y[0, 0].detach().numpy()  # [5, 8]
        diagrams = vr.compute_persistence(point_cloud)

        # 4. Persistence landscape
        pl = PersistenceLandscape(n_layers=3, n_points=50)
        landscape = pl.compute_multi_dim(diagrams)
        assert landscape.shape[0] == 6  # 2 dims × 3 layers
        assert landscape.shape[1] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
