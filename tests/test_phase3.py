"""
Sprint 5 Test Suite: Phase 3 — Jump-Diffusion Generative Engine.

Run: python -m pytest tests/test_phase3.py -v
"""

import sys, os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase3_diffusion.noise_schedule import NoiseSchedule
from models.phase3_diffusion.jump_process import JumpProcess
from models.phase3_diffusion.score_network import ScoreNetwork, SinusoidalTimeEmbedding
from models.phase3_diffusion.diffusion_engine import JumpDiffusionEngine


class TestNoiseSchedule:

    def test_linear_schedule_shape(self):
        ns = NoiseSchedule(n_steps=100, schedule_type="linear")
        assert ns.betas.shape == (100,)
        assert ns.alphas_cumprod.shape == (100,)

    def test_cosine_schedule_monotonic(self):
        """ᾱ_t must be monotonically decreasing."""
        ns = NoiseSchedule(n_steps=200, schedule_type="cosine")
        diffs = ns.alphas_cumprod[1:] - ns.alphas_cumprod[:-1]
        assert (diffs <= 1e-6).all()

    def test_alphas_cumprod_range(self):
        """ᾱ_t ∈ (0, 1) for all t."""
        ns = NoiseSchedule(n_steps=100, schedule_type="cosine")
        assert (ns.alphas_cumprod > 0).all()
        assert (ns.alphas_cumprod <= 1.0).all()

    def test_q_sample_shape(self):
        ns = NoiseSchedule(n_steps=100, schedule_type="linear")
        x_0 = torch.randn(4, 12)
        t = torch.randint(0, 100, (4,))
        noise = torch.randn_like(x_0)
        x_t = ns.q_sample(x_0, t, noise)
        assert x_t.shape == (4, 12)

    def test_q_sample_t0_close_to_x0(self):
        """At t=0, x_t should be very close to x_0."""
        ns = NoiseSchedule(n_steps=1000, schedule_type="cosine")
        x_0 = torch.randn(4, 12)
        t = torch.zeros(4, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = ns.q_sample(x_0, t, noise)
        assert torch.allclose(x_t, x_0, atol=0.1)

    def test_q_sample_tmax_close_to_noise(self):
        """At t=T-1, x_t should be mostly noise."""
        ns = NoiseSchedule(n_steps=1000, schedule_type="cosine")
        x_0 = torch.ones(100, 12) * 5.0  # Strong signal
        t = torch.full((100,), 999, dtype=torch.long)
        noise = torch.randn(100, 12)
        x_t = ns.q_sample(x_0, t, noise)
        # Signal should be mostly destroyed
        assert x_t.std() > 0.5  # Should have noise variance

    def test_get_coefficients(self):
        ns = NoiseSchedule(n_steps=100, schedule_type="linear")
        t = torch.tensor([0, 50, 99])
        coeff = ns.get_coefficients(t)
        assert coeff["sqrt_alpha_cumprod"].shape == (3,)
        assert coeff["beta"].shape == (3,)


class TestJumpProcess:

    def test_sample_jumps_shape(self):
        jp = JumpProcess(jump_intensity=0.5)
        mask, magnitudes = jp.sample_jumps((4, 12), torch.device("cpu"))
        assert mask.shape == (4, 12)
        assert magnitudes.shape == (4, 12)

    def test_jump_mask_binary(self):
        jp = JumpProcess(jump_intensity=0.5)
        mask, _ = jp.sample_jumps((100, 12), torch.device("cpu"))
        # Mask should be 0 or 1
        assert ((mask == 0.0) | (mask == 1.0)).all()

    def test_jumps_only_where_masked(self):
        """Magnitudes should be zero where mask is zero."""
        jp = JumpProcess(jump_intensity=0.3)
        mask, mag = jp.sample_jumps((100, 12), torch.device("cpu"))
        zero_mask = mask == 0.0
        assert (mag[zero_mask] == 0.0).all()

    def test_forward_shape(self):
        jp = JumpProcess(jump_intensity=0.1)
        x = torch.randn(4, 12)
        out = jp(x, 0)
        assert out.shape == (4, 12)


class TestScoreNetwork:

    def test_output_shape(self):
        net = ScoreNetwork(
            data_dim=12, channels=[32, 64],
            time_embed_dim=32, condition_dim=64,
        )
        x = torch.randn(4, 12)
        t = torch.randint(0, 100, (4,))
        c = torch.randn(4, 64)
        out = net(x, t, c)
        assert out.shape == (4, 12)

    def test_gradient_flow(self):
        net = ScoreNetwork(
            data_dim=8, channels=[16, 32],
            time_embed_dim=16, condition_dim=32,
        )
        x = torch.randn(2, 8, requires_grad=True)
        t = torch.randint(0, 50, (2,))
        c = torch.randn(2, 32)
        out = net(x, t, c)
        out.sum().backward()
        assert x.grad is not None
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_time_embedding(self):
        emb = SinusoidalTimeEmbedding(64)
        t = torch.tensor([0, 50, 999])
        out = emb(t)
        assert out.shape == (3, 64)
        # Different times → different embeddings
        assert not torch.allclose(out[0], out[1])


class TestDiffusionEngine:

    @pytest.fixture
    def engine(self):
        return JumpDiffusionEngine(
            data_dim=8, condition_dim=32, n_steps=50,
            schedule_type="linear", jump_intensity=0.1,
            channels=[16, 32],
        )

    def test_training_loss(self, engine):
        x_0 = torch.randn(4, 8)
        c = torch.randn(4, 32)
        result = engine.training_loss(x_0, c)
        assert result["loss"].dim() == 0
        assert not torch.isnan(result["loss"])
        assert result["loss"].item() >= 0
        assert result["predicted_noise"].shape == (4, 8)

    def test_training_loss_backward(self, engine):
        x_0 = torch.randn(4, 8)
        c = torch.randn(4, 32)
        result = engine.training_loss(x_0, c)
        result["loss"].backward()
        grad_count = sum(1 for p in engine.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_sample_shape(self, engine):
        c = torch.randn(3, 32)
        samples = engine.sample(c)
        assert samples.shape == (3, 8)
        assert not torch.isnan(samples).any()

    def test_sample_finite(self, engine):
        """Samples should be finite (no explosion)."""
        c = torch.randn(2, 32)
        samples = engine.sample(c)
        assert torch.isfinite(samples).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
