"""
Sprint 7 Test Suite: TTSA Training Loop.

Run: python -m pytest tests/test_ttsa.py -v
"""

import sys, os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ttsa_trainer import TTSATrainer, TTSAScheduler


# --- Minimal mock models for testing the training loop ---

class MockGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 8)

    def forward(self, x):
        return self.linear(x)


class MockController(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 5)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


class TestTTSAScheduler:

    def test_decay_rates(self):
        sched = TTSAScheduler(
            inner_lr_initial=1e-3, outer_lr_initial=5e-4,
            inner_decay_power=0.6, outer_decay_power=0.8,
        )
        # At step 0
        assert sched.get_inner_lr(0) == 1e-3
        assert sched.get_outer_lr(0) == 5e-4

        # After many steps, outer should decay faster
        ratio_early = sched.get_outer_lr(10) / sched.get_inner_lr(10)
        ratio_late = sched.get_outer_lr(1000) / sched.get_inner_lr(1000)
        assert ratio_late < ratio_early  # β/α → 0

    def test_invalid_powers_rejected(self):
        with pytest.raises(AssertionError):
            TTSAScheduler(inner_decay_power=0.8, outer_decay_power=0.6)

    def test_sum_diverges(self):
        """Σ α_k = ∞ (Robbins-Monro condition). p < 1 ⟹ harmonic series diverges."""
        sched = TTSAScheduler(
            inner_lr_initial=1.0, outer_lr_initial=0.5,
            inner_decay_power=0.6, outer_decay_power=0.8,
        )
        # Sum of 1/(1+k)^0.6 diverges (p < 1)
        total = sum(sched.get_inner_lr(k) for k in range(10000))
        assert total > 50.0  # Should be large for unit LR (actual ≈ 97.6)


class TestTTSATrainer:

    @pytest.fixture
    def trainer(self):
        gen = MockGenerator()
        ctrl = MockController()
        return TTSATrainer(
            generator=gen, controller=ctrl,
            inner_lr=1e-3, outer_lr=5e-4,
            inner_decay_power=0.6, outer_decay_power=0.8,
            n_inner_steps_per_outer=3,
        )

    def test_inner_step(self, trainer):
        batch = {"x": torch.randn(4, 10)}

        def inner_loss_fn(batch):
            out = trainer.generator(batch["x"])
            return {"loss": out.mean()}

        metrics = trainer.inner_step(batch, inner_loss_fn)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_outer_step(self, trainer):
        batch = {"x": torch.randn(4, 10)}

        def outer_loss_fn(batch):
            # Stackelberg wall: detach generator output
            with torch.no_grad():
                gen_out = trainer.generator(batch["x"])
            out = trainer.controller(gen_out)
            return {"loss": out.mean()}

        metrics = trainer.outer_step(batch, outer_loss_fn)
        assert "loss" in metrics

    def test_full_train_step(self, trainer):
        batch = {"x": torch.randn(4, 10)}

        def inner_loss_fn(batch):
            out = trainer.generator(batch["x"])
            return {"loss": out.mean()}

        def outer_loss_fn(batch):
            with torch.no_grad():
                gen_out = trainer.generator(batch["x"])
            out = trainer.controller(gen_out)
            return {"loss": out.mean()}

        metrics = trainer.train_step(batch, inner_loss_fn, outer_loss_fn)
        assert "inner/loss" in metrics
        assert "outer/loss" in metrics
        assert "inner_lr" in metrics
        assert "outer_lr" in metrics
        assert metrics["step"] == 0

    def test_lr_decay_across_steps(self, trainer):
        batch = {"x": torch.randn(4, 10)}

        def loss_fn(batch):
            return {"loss": trainer.generator(batch["x"]).mean()}

        def outer_fn(batch):
            with torch.no_grad():
                g = trainer.generator(batch["x"])
            return {"loss": trainer.controller(g).mean()}

        # Run several steps
        lrs = []
        for _ in range(5):
            m = trainer.train_step(batch, loss_fn, outer_fn)
            lrs.append((m["inner_lr"], m["outer_lr"]))

        # LRs should decrease
        for i in range(1, len(lrs)):
            assert lrs[i][0] <= lrs[i-1][0] + 1e-8
            assert lrs[i][1] <= lrs[i-1][1] + 1e-8

    def test_ttsa_conditions_verified(self, trainer):
        batch = {"x": torch.randn(4, 10)}

        def loss_fn(batch):
            return {"loss": trainer.generator(batch["x"]).mean()}

        def outer_fn(batch):
            with torch.no_grad():
                g = trainer.generator(batch["x"])
            return {"loss": trainer.controller(g).mean()}

        trainer.train_step(batch, loss_fn, outer_fn)
        trainer.train_step(batch, loss_fn, outer_fn)

        conditions = trainer.verify_ttsa_conditions()
        assert conditions["inner_lr_positive"]
        assert conditions["outer_lr_positive"]

    def test_stackelberg_wall(self, trainer):
        """Generator gradients must NOT be affected by outer loss."""
        batch = {"x": torch.randn(4, 10)}

        # Save generator parameters before outer step
        gen_params_before = {
            n: p.clone() for n, p in trainer.generator.named_parameters()
        }

        def outer_loss_fn(batch):
            with torch.no_grad():
                gen_out = trainer.generator(batch["x"])
            # gen_out is detached — no grad to generator
            out = trainer.controller(gen_out)
            return {"loss": out.mean()}

        trainer.outer_step(batch, outer_loss_fn)

        # Generator params should be UNCHANGED
        for n, p in trainer.generator.named_parameters():
            assert torch.equal(p, gen_params_before[n]), (
                f"Generator param {n} changed during outer step!"
            )

    def test_checkpoint_roundtrip(self, trainer, tmp_path):
        batch = {"x": torch.randn(4, 10)}

        def loss_fn(batch):
            return {"loss": trainer.generator(batch["x"]).mean()}

        def outer_fn(batch):
            with torch.no_grad():
                g = trainer.generator(batch["x"])
            return {"loss": trainer.controller(g).mean()}

        trainer.train_step(batch, loss_fn, outer_fn)
        trainer.train_step(batch, loss_fn, outer_fn)

        path = str(tmp_path / "ckpt.pt")
        trainer.save_checkpoint(path)

        # Create fresh trainer and load
        gen2 = MockGenerator()
        ctrl2 = MockController()
        trainer2 = TTSATrainer(
            generator=gen2, controller=ctrl2,
            inner_lr=1e-3, outer_lr=5e-4,
            inner_decay_power=0.6, outer_decay_power=0.8,
        )
        trainer2.load_checkpoint(path)

        assert trainer2.global_step == 2
        assert len(trainer2.metrics_history) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
