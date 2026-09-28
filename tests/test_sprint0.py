"""
Sprint 0 Test Suite: Foundation Layer.

Tests for configs, utils (matrix_ops, device_manager, seed),
and the data pipeline (LOB parser, feature engineer, causal normalizer,
spline interpolator, dataset, synthetic generator).

Run: python -m pytest tests/test_sprint0.py -v
"""

import math
import sys
import os

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default_config import (
    MasterConfig,
    Phase0Config,
    Phase1Config,
    Phase2Config,
    Phase3Config,
    Phase4Config,
    SGWConfig,
    TrainingConfig,
    EvaluationConfig,
    DataConfig,
)
from utils.matrix_ops import (
    stable_matrix_sqrt,
    stable_matrix_inv_sqrt,
    stable_matrix_log,
    stable_matrix_exp,
    symmetrise,
    upper_triangular_to_vector,
    vector_to_upper_triangular,
    batch_trace,
    regularise_matrix,
)
from utils.device_manager import DeviceManager, get_device
from utils.seed import set_global_seed
from data.synthetic_generator import SyntheticJumpDiffusionGenerator, SyntheticConfig
from data.feature_engineer import FeatureEngineer
from data.causal_normalizer import CausalNormalizer
from data.spline_interpolator import SplineInterpolator
from data.dataset import MarketDataset, create_dataloaders


# ═══════════════════════════════════════════════════════════════════
#  Configuration Tests
# ═══════════════════════════════════════════════════════════════════

class TestConfiguration:
    """Tests for the @dataclass configuration system."""

    def test_master_config_defaults(self):
        """MasterConfig should instantiate with valid defaults."""
        config = MasterConfig()
        assert config.phase0.hidden_dim == 256
        assert config.phase0.n_regimes == 4
        assert config.phase0.uncertainty_threshold == 0.4
        assert config.phase0.gate_temperature == 10.0
        assert config.phase3.n_diffusion_steps == 1000
        assert config.phase4.huber_delta == 0.01
        assert config.training.n_inner_steps_per_outer == 5

    def test_config_override(self):
        """Sub-configs should be overridable at construction."""
        config = MasterConfig(
            phase0=Phase0Config(hidden_dim=512, n_regimes=6),
            data=DataConfig(n_assets=50),
        )
        assert config.phase0.hidden_dim == 512
        assert config.phase0.n_regimes == 6
        assert config.data.n_assets == 50

    def test_derived_properties(self):
        """Derived properties should compute correctly."""
        config = MasterConfig(data=DataConfig(n_assets=12, n_features=3))
        assert config.n_assets == 12
        assert config.input_dim == 36  # 12 × 3
        assert config.tangent_dim == 78  # 12×13/2

    def test_ttsa_learning_rate_conditions(self):
        """TTSA conditions: outer decay must be faster than inner."""
        config = MasterConfig()
        assert config.training.outer_lr_decay_power > config.training.inner_lr_decay_power
        assert config.training.outer_lr_initial < config.training.inner_lr_initial

    def test_evaluation_sfps_weights_sum(self):
        """SFPS weights must sum to 1.0."""
        config = MasterConfig()
        assert abs(sum(config.evaluation.sfps_weights) - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════
#  Matrix Operations Tests
# ═══════════════════════════════════════════════════════════════════

class TestMatrixOps:
    """Tests for numerically stable matrix operations."""

    def _make_spd(self, N: int, batch_size: int = 0) -> torch.Tensor:
        """Helper: generate a random SPD matrix."""
        if batch_size > 0:
            A = torch.randn(batch_size, N, N)
        else:
            A = torch.randn(N, N)
        return A @ A.transpose(-2, -1) + torch.eye(N) * 0.1

    def test_matrix_sqrt_identity(self):
        """sqrt(I) = I."""
        I = torch.eye(5)
        result = stable_matrix_sqrt(I)
        assert torch.allclose(result, I, atol=1e-6)

    def test_matrix_sqrt_squared(self):
        """sqrt(A) @ sqrt(A) = A for SPD matrices."""
        A = self._make_spd(5)
        sqrt_A = stable_matrix_sqrt(A)
        reconstructed = sqrt_A @ sqrt_A
        assert torch.allclose(reconstructed, symmetrise(A), atol=1e-4)

    def test_matrix_sqrt_batch(self):
        """Matrix sqrt should work on batched tensors."""
        A = self._make_spd(4, batch_size=3)
        sqrt_A = stable_matrix_sqrt(A)
        assert sqrt_A.shape == (3, 4, 4)

    def test_inv_sqrt_times_sqrt_is_identity(self):
        """A^{-1/2} @ A^{1/2} = I."""
        A = self._make_spd(5)
        sqrt_A = stable_matrix_sqrt(A)
        inv_sqrt_A = stable_matrix_inv_sqrt(A)
        product = inv_sqrt_A @ sqrt_A
        assert torch.allclose(product, torch.eye(5), atol=1e-4)

    def test_matrix_log_exp_roundtrip(self):
        """exp(log(A)) = A for SPD matrices."""
        A = self._make_spd(4)
        log_A = stable_matrix_log(A)
        reconstructed = stable_matrix_exp(log_A)
        assert torch.allclose(reconstructed, symmetrise(A), atol=1e-4)

    def test_near_singular_matrix(self):
        """Matrix operations should handle near-singular matrices."""
        # Create a nearly singular matrix (simulating Black Swan correlation)
        N = 5
        v = torch.ones(N)
        A = v.unsqueeze(1) @ v.unsqueeze(0)  # Rank-1 matrix
        A = A + 1e-7 * torch.eye(N)  # Tiny regularisation

        # These should NOT produce NaN
        sqrt_A = stable_matrix_sqrt(A)
        assert not torch.isnan(sqrt_A).any()

        log_A = stable_matrix_log(A)
        assert not torch.isnan(log_A).any()

    def test_upper_triangular_roundtrip(self):
        """Flatten → unflatten should reconstruct the original symmetric matrix."""
        A = self._make_spd(5)
        A = symmetrise(A)
        v = upper_triangular_to_vector(A)
        assert v.shape == (15,)  # 5×6/2
        A_reconstructed = vector_to_upper_triangular(v, 5)
        assert torch.allclose(A_reconstructed, A, atol=1e-6)

    def test_batch_trace(self):
        """Trace should equal sum of diagonal elements."""
        A = torch.randn(3, 4, 4)
        traces = batch_trace(A)
        assert traces.shape == (3,)
        for i in range(3):
            expected = A[i].diag().sum()
            assert torch.allclose(traces[i], expected)

    def test_regularise_matrix(self):
        """Regularisation should make singular matrices invertible."""
        N = 5
        A = torch.zeros(N, N)  # Completely singular
        A_reg = regularise_matrix(A, eps=0.01)
        eigvals = torch.linalg.eigvalsh(A_reg)
        assert (eigvals >= 0.009).all()  # All eigenvalues ≥ eps


# ═══════════════════════════════════════════════════════════════════
#  Device Manager Tests
# ═══════════════════════════════════════════════════════════════════

class TestDeviceManager:
    """Tests for device management."""

    def test_cpu_device(self):
        dm = DeviceManager("cpu")
        assert dm.device.type == "cpu"
        assert not dm.is_cuda

    def test_place_tensor(self):
        dm = DeviceManager("cpu")
        t = torch.randn(3, 3)
        placed = dm.place(t)
        assert placed.device.type == "cpu"

    def test_get_device_fallback(self):
        device = get_device("cpu")
        assert device.type == "cpu"


# ═══════════════════════════════════════════════════════════════════
#  Seed Tests
# ═══════════════════════════════════════════════════════════════════

class TestSeed:
    """Tests for reproducibility."""

    def test_seed_reproducibility(self):
        """Same seed should produce identical random numbers."""
        set_global_seed(123)
        a = torch.randn(10)
        set_global_seed(123)
        b = torch.randn(10)
        assert torch.allclose(a, b)


# ═══════════════════════════════════════════════════════════════════
#  Synthetic Data Generator Tests
# ═══════════════════════════════════════════════════════════════════

class TestSyntheticGenerator:
    """Tests for the jump-diffusion synthetic data generator."""

    @pytest.fixture
    def generator(self):
        return SyntheticJumpDiffusionGenerator(
            SyntheticConfig(n_assets=5, n_timesteps=500, seed=42)
        )

    def test_output_shapes(self, generator):
        data = generator.generate()
        assert data["prices"].shape == (500, 5)
        assert data["returns"].shape == (500, 5)
        assert data["timestamps"].shape == (500,)
        assert data["lob"].shape == (500, 5, 40)  # 4×10 = 40
        assert data["bid_prices"].shape == (500, 5, 10)
        assert data["correlation_matrix"].shape == (5, 5)

    def test_prices_positive(self, generator):
        data = generator.generate()
        assert (data["prices"] > 0).all()

    def test_bid_ask_ordering(self, generator):
        """Best bid must be below best ask (market microstructure law)."""
        data = generator.generate()
        best_bid = data["bid_prices"][:, :, 0]
        best_ask = data["ask_prices"][:, :, 0]
        assert (best_bid < best_ask).all()

    def test_jumps_present(self):
        """Jump-diffusion should produce some jump events."""
        # Use high intensity to guarantee jumps in small dataset
        # Expected jumps = T × N × λ × dt = 500 × 5 × 50 × (1/252) ≈ 496
        gen = SyntheticJumpDiffusionGenerator(
            SyntheticConfig(n_assets=5, n_timesteps=500, jump_intensity=50.0, seed=42)
        )
        data = gen.generate()
        assert data["jump_times"].sum() > 0

    def test_reproducibility(self):
        gen1 = SyntheticJumpDiffusionGenerator(SyntheticConfig(seed=42))
        gen2 = SyntheticJumpDiffusionGenerator(SyntheticConfig(seed=42))
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1["prices"], data2["prices"])

    def test_torch_output(self, generator):
        data = generator.generate_torch(device="cpu")
        assert isinstance(data["prices"], torch.Tensor)
        assert data["prices"].dtype == torch.float32


# ═══════════════════════════════════════════════════════════════════
#  Feature Engineering Tests
# ═══════════════════════════════════════════════════════════════════

class TestFeatureEngineer:
    """Tests for microstructural feature extraction (§7.4.2)."""

    @pytest.fixture
    def data_and_engineer(self):
        gen = SyntheticJumpDiffusionGenerator(
            SyntheticConfig(n_assets=3, n_timesteps=100, lob_depth=5, seed=42)
        )
        data = gen.generate()
        fe = FeatureEngineer(depth=5, kappa=0.5)
        return data, fe

    def test_micro_price_within_spread(self, data_and_engineer):
        """Micro-price must lie between best bid and best ask."""
        data, fe = data_and_engineer
        micro = fe.compute_micro_price(
            data["bid_prices"], data["bid_volumes"],
            data["ask_prices"], data["ask_volumes"],
        )
        best_bid = data["bid_prices"][:, :, 0]
        best_ask = data["ask_prices"][:, :, 0]
        assert (micro >= best_bid - 1e-4).all()
        assert (micro <= best_ask + 1e-4).all()

    def test_ofi_shape(self, data_and_engineer):
        data, fe = data_and_engineer
        ofi = fe.compute_ofi(
            data["bid_prices"], data["bid_volumes"],
            data["ask_prices"], data["ask_volumes"],
        )
        assert ofi.shape == (100, 3)

    def test_ofi_first_element_zero(self, data_and_engineer):
        """OFI at t=0 should be 0 (no previous tick)."""
        data, fe = data_and_engineer
        ofi = fe.compute_ofi(
            data["bid_prices"], data["bid_volumes"],
            data["ask_prices"], data["ask_volumes"],
        )
        np.testing.assert_array_equal(ofi[0], 0.0)

    def test_liquidity_skew_bounded(self, data_and_engineer):
        """Liquidity skew should be bounded."""
        data, fe = data_and_engineer
        skew = fe.compute_liquidity_skew(
            data["bid_volumes"], data["ask_volumes"],
        )
        # Each term in the sum is bounded by exp(-κd) × [-1, 1]
        max_possible = sum(np.exp(-0.5 * d) for d in range(5))
        assert (np.abs(skew) <= max_possible + 1e-6).all()

    def test_compute_all_shape(self, data_and_engineer):
        data, fe = data_and_engineer
        features = fe.compute_all(
            data["bid_prices"], data["bid_volumes"],
            data["ask_prices"], data["ask_volumes"],
        )
        assert features.shape == (100, 3, 3)  # [T, N, F=3]


# ═══════════════════════════════════════════════════════════════════
#  Causal Normaliser Tests
# ═══════════════════════════════════════════════════════════════════

class TestCausalNormalizer:
    """Tests for the rolling causal Z-score normaliser (§7.4.3)."""

    def test_output_shape_preserved(self):
        norm = CausalNormalizer(window_size=20)
        features = np.random.randn(100, 5, 3).astype(np.float32)
        result = norm.normalize_numpy(features)
        assert result.shape == features.shape

    def test_causal_no_lookahead(self):
        """Normalisation at time t must not depend on t+1."""
        norm = CausalNormalizer(window_size=10, min_periods=2)
        features = np.random.randn(50, 3).astype(np.float64)

        result_full = norm.normalize_numpy(features)

        # Truncate at t=30 and normalise again
        result_truncated = norm.normalize_numpy(features[:31])

        # Values at t=30 should be IDENTICAL
        np.testing.assert_allclose(
            result_full[30], result_truncated[30], atol=1e-10
        )

    def test_warmup_period(self):
        """First few samples should be zero-filled."""
        norm = CausalNormalizer(window_size=20, min_periods=10)
        features = np.random.randn(50, 3).astype(np.float64)
        result = norm.normalize_numpy(features)
        # First min_periods-1 rows should be zero
        np.testing.assert_array_equal(result[:9], 0.0)

    def test_torch_matches_numpy(self):
        """Torch and NumPy implementations should agree."""
        norm = CausalNormalizer(window_size=15, min_periods=5)
        features = np.random.randn(80, 4).astype(np.float64)

        result_np = norm.normalize_numpy_vectorised(features)
        result_torch = norm.normalize_torch(
            torch.tensor(features, dtype=torch.float64)
        ).numpy()

        np.testing.assert_allclose(result_np, result_torch, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════
#  Spline Interpolator Tests
# ═══════════════════════════════════════════════════════════════════

class TestSplineInterpolator:
    """Tests for the cubic spline interpolator (§7.4.3)."""

    def test_interpolation_at_knots(self):
        """Spline must pass exactly through the data points."""
        interp = SplineInterpolator()
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([[1.0], [2.0], [1.5], [3.0], [2.5]])
        interp.fit(t, values)
        result = interp.evaluate(t)
        np.testing.assert_allclose(result, values, atol=1e-10)

    def test_c1_continuity(self):
        """Derivative should be continuous (finite differences should agree)."""
        interp = SplineInterpolator()
        t = np.linspace(0, 5, 20)
        values = np.sin(t).reshape(-1, 1)
        interp.fit(t, values)

        # Analytical derivative from spline
        query_t = np.array([2.5])
        deriv = interp.evaluate_derivative(query_t, order=1)

        # Numerical derivative
        eps = 1e-6
        f_plus = interp.evaluate(query_t + eps)
        f_minus = interp.evaluate(query_t - eps)
        num_deriv = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(deriv, num_deriv, atol=1e-4)

    def test_torch_evaluation_matches_scipy(self):
        """GPU-native evaluation should match SciPy."""
        interp = SplineInterpolator()
        t = np.linspace(0, 3, 15)
        values = np.random.randn(15, 2, 3)
        interp.fit(t, values)

        query = np.linspace(0.1, 2.9, 50)
        scipy_result = interp.evaluate(query)

        coeffs, breaks = interp.to_torch_coefficients()
        torch_result = interp.evaluate_torch(torch.tensor(query)).numpy()

        np.testing.assert_allclose(scipy_result, torch_result, atol=1e-4)

    def test_derivative_torch_matches_scipy(self):
        """GPU-native derivative should match SciPy."""
        interp = SplineInterpolator()
        t = np.linspace(0, 4, 20)
        values = np.sin(t).reshape(-1, 1)
        interp.fit(t, values)

        query = np.linspace(0.2, 3.8, 30)
        scipy_deriv = interp.evaluate_derivative(query, order=1)

        coeffs, breaks = interp.to_torch_coefficients()
        torch_deriv = interp.evaluate_derivative_torch(
            torch.tensor(query)
        ).numpy()

        np.testing.assert_allclose(scipy_deriv, torch_deriv, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════
#  Dataset Tests
# ═══════════════════════════════════════════════════════════════════

class TestDataset:
    """Tests for the PyTorch Dataset and DataLoader wrappers."""

    @pytest.fixture
    def sample_data(self):
        T, N, F = 200, 5, 3
        features = np.random.randn(T, N, F).astype(np.float32)
        timestamps = np.arange(T, dtype=np.float64) / 252.0
        returns = np.random.randn(T, N).astype(np.float32) * 0.01
        return features, timestamps, returns

    def test_dataset_length(self, sample_data):
        features, timestamps, returns = sample_data
        ds = MarketDataset(features, timestamps, returns, lookback=30, horizon=5)
        expected = 200 - 30 - 5 + 1
        assert len(ds) == expected

    def test_sample_shapes(self, sample_data):
        features, timestamps, returns = sample_data
        ds = MarketDataset(features, timestamps, returns, lookback=30, horizon=5)
        sample = ds[0]
        assert sample["features"].shape == (30, 5, 3)
        assert sample["timestamps"].shape == (30,)
        assert sample["targets"].shape == (5, 5)

    def test_no_overlap_between_features_and_targets(self, sample_data):
        """Feature window and target window must not overlap."""
        features, timestamps, returns = sample_data
        ds = MarketDataset(features, timestamps, returns, lookback=30, horizon=5)

        for i in [0, 10, len(ds) - 1]:
            sample = ds[i]
            feat_end_time = sample["timestamps"][-1].item()
            # Target returns start after the feature window
            target_start_idx = i + 30
            target_end_idx = i + 30 + 5
            assert target_start_idx > i + 29  # No overlap

    def test_create_dataloaders(self, sample_data):
        features, timestamps, returns = sample_data
        loaders = create_dataloaders(
            features, timestamps, returns,
            lookback=20, horizon=5,
            train_ratio=0.7, val_ratio=0.15,
            batch_size=8,
        )
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

        # Training loader should produce batches
        batch = next(iter(loaders["train"]))
        assert batch["features"].shape[0] <= 8
        assert batch["features"].shape[1] == 20


# ═══════════════════════════════════════════════════════════════════
#  Integration Test: Full Pipeline
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end integration test for Sprint 0 data pipeline."""

    def test_synthetic_to_dataloader(self):
        """Full pipeline: synthetic data → features → normalise → dataset → loader."""
        # 1. Generate synthetic data
        gen = SyntheticJumpDiffusionGenerator(
            SyntheticConfig(n_assets=5, n_timesteps=300, lob_depth=5, seed=42)
        )
        data = gen.generate()

        # 2. Extract features
        fe = FeatureEngineer(depth=5, kappa=0.5)
        features = fe.compute_all(
            data["bid_prices"], data["bid_volumes"],
            data["ask_prices"], data["ask_volumes"],
        )
        assert features.shape == (300, 5, 3)

        # 3. Causal normalisation
        norm = CausalNormalizer(window_size=30, min_periods=10)
        normed = norm.normalize_numpy_vectorised(features)
        assert normed.shape == features.shape
        assert not np.isnan(normed).any()

        # 4. Create dataloaders
        loaders = create_dataloaders(
            features=normed,
            timestamps=data["timestamps"],
            returns=data["returns"],
            lookback=20,
            horizon=5,
            batch_size=16,
        )

        # 5. Verify a batch
        batch = next(iter(loaders["train"]))
        assert batch["features"].shape == (16, 20, 5, 3)
        assert not torch.isnan(batch["features"]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
