"""
Synthetic Jump-Diffusion Data Generator.

Generates multivariate asset paths using a jump-diffusion SDE for
development and unit testing without requiring real LOB data.

The existing synthetic generator follows the SDE:

    dY_t = μ(Y_t, t) dt + σ(t) dW_t + J(Y_{t⁻}) dN_t

where:
    - dW_t: Standard multivariate Wiener process (Brownian motion)
    - dN_t: Poisson counting process with intensity λ
    - J(·): Jump magnitudes from a heavy-tailed distribution

The generator also produces synthetic LOB-like data (bid/ask prices
and volumes) for testing the feature engineering pipeline.

This supports existing development tests without data acquisition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    n_assets: int = 12
    """N: Number of assets."""

    n_timesteps: int = 5000
    """T: Number of discrete time steps."""

    dt: float = 1.0 / 252.0
    """Δt: Time step size (1/252 = daily)."""

    # --- Drift ---
    mu_range: Tuple[float, float] = (-0.05, 0.15)
    """Range of annualised drift per asset."""

    # --- Diffusion ---
    sigma_range: Tuple[float, float] = (0.1, 0.4)
    """Range of annualised volatility per asset."""

    correlation_strength: float = 0.3
    """Off-diagonal correlation strength (generates correlated returns)."""

    # --- Jump Process ---
    jump_intensity: float = 0.02
    """λ: Poisson jump intensity (probability of jump per step)."""

    jump_mean: float = -0.03
    """Mean of the jump magnitude distribution (negative = crashes)."""

    jump_std: float = 0.05
    """Standard deviation of jump magnitudes."""

    # --- LOB Simulation ---
    lob_depth: int = 10
    """D: Number of LOB levels per side."""

    spread_bps: float = 5.0
    """Average bid-ask spread in basis points."""

    volume_mean: float = 1000.0
    """Average volume per LOB level."""

    volume_std: float = 500.0
    """Volume standard deviation."""

    # --- Seed ---
    seed: Optional[int] = 42


class SyntheticJumpDiffusionGenerator:
    r"""Generator for synthetic multivariate jump-diffusion paths.

    Simulates asset prices via Euler-Maruyama discretisation of:

    .. math::
        dY_t = \mu \cdot Y_t \, dt + \sigma \cdot Y_t \, dW_t + J \cdot Y_{t^-} \, dN_t

    where Y_t is the price vector, dW_t is correlated Brownian motion
    (via Cholesky decomposition of the correlation matrix), and dN_t
    is a compound Poisson process.

    The output includes:
        1. Price paths [T, N]
        2. Log-returns [T, N]
        3. Synthetic LOB snapshots [T, N, 4D] for the feature pipeline
        4. Timestamps [T]

    Args:
        config: Synthetic data configuration.

    Example:
        >>> gen = SyntheticJumpDiffusionGenerator()
        >>> data = gen.generate()
        >>> data["prices"].shape   # [5000, 12]
        >>> data["returns"].shape  # [5000, 12]
        >>> data["lob"].shape      # [5000, 12, 40]  (depth=10, 4D)
    """

    def __init__(self, config: Optional[SyntheticConfig] = None) -> None:
        self.config = config or SyntheticConfig()

    def generate(self) -> Dict[str, np.ndarray]:
        """Generate a complete synthetic dataset.

        Returns:
            Dictionary with keys:
                - ``prices``: Asset price paths, shape [T, N].
                - ``log_prices``: Log-price paths, shape [T, N].
                - ``returns``: Log-returns, shape [T, N].
                - ``timestamps``: Time points, shape [T].
                - ``lob``: Synthetic LOB state, shape [T, N, 4D].
                - ``bid_prices``: shape [T, N, D].
                - ``bid_volumes``: shape [T, N, D].
                - ``ask_prices``: shape [T, N, D].
                - ``ask_volumes``: shape [T, N, D].
                - ``correlation_matrix``: True Σ, shape [N, N].
                - ``jump_times``: Boolean mask of jump events, shape [T, N].
        """
        cfg = self.config

        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        N = cfg.n_assets
        T = cfg.n_timesteps
        dt = cfg.dt

        logger.info(
            f"Generating synthetic data: N={N}, T={T}, dt={dt:.6f}, "
            f"λ_jump={cfg.jump_intensity}"
        )

        # --- Generate Correlated Brownian Motion ---
        # Construct correlation matrix with off-diagonal strength
        corr_matrix = np.eye(N) * (1 - cfg.correlation_strength)
        corr_matrix += cfg.correlation_strength
        # Ensure positive-definiteness via eigenvalue correction
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Cholesky decomposition for correlated normals
        L = np.linalg.cholesky(corr_matrix)

        # Per-asset drift and volatility
        mu = np.random.uniform(cfg.mu_range[0], cfg.mu_range[1], size=N)
        sigma = np.random.uniform(cfg.sigma_range[0], cfg.sigma_range[1], size=N)

        # --- Euler-Maruyama Simulation ---
        prices = np.zeros((T, N))
        prices[0] = np.random.uniform(50, 200, size=N)  # Initial prices
        log_prices = np.log(prices[0])

        log_price_paths = np.zeros((T, N))
        log_price_paths[0] = log_prices

        jump_mask = np.zeros((T, N), dtype=bool)

        for t in range(1, T):
            # Continuous component: geometric Brownian motion
            # dlog(Y) = (μ − σ²/2)dt + σ dW
            z = np.random.randn(N)
            correlated_z = L @ z  # Apply correlation

            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * correlated_z

            # Jump component: compound Poisson
            jump_events = np.random.poisson(cfg.jump_intensity * dt, size=N)
            jump_magnitudes = np.where(
                jump_events > 0,
                np.random.normal(cfg.jump_mean, cfg.jump_std, size=N) * jump_events,
                0.0,
            )
            jump_mask[t] = jump_events > 0

            # Update log-prices
            log_price_paths[t] = log_price_paths[t - 1] + drift + diffusion + jump_magnitudes
            prices[t] = np.exp(log_price_paths[t])

        # Compute returns
        returns = np.diff(log_price_paths, axis=0)
        returns = np.vstack([np.zeros((1, N)), returns])  # Pad first row

        # --- Generate Synthetic LOB ---
        lob_data = self._generate_lob(prices, cfg)

        # --- Timestamps ---
        timestamps = np.arange(T, dtype=np.float64) * dt

        result = {
            "prices": prices.astype(np.float32),
            "log_prices": log_price_paths.astype(np.float32),
            "returns": returns.astype(np.float32),
            "timestamps": timestamps,
            "correlation_matrix": corr_matrix.astype(np.float32),
            "jump_times": jump_mask,
            **lob_data,
        }

        logger.info(
            f"Synthetic data generated: "
            f"price_range=[{prices.min():.2f}, {prices.max():.2f}], "
            f"n_jumps={jump_mask.sum()}, "
            f"return_std={returns.std():.4f}"
        )

        return result

    def _generate_lob(
        self,
        prices: np.ndarray,
        cfg: SyntheticConfig,
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic LOB snapshots around the price paths.

        Creates D levels of bid/ask prices and volumes, centered around
        the simulated mid-prices.

        Args:
            prices: Mid-price paths, shape [T, N].
            cfg: Configuration.

        Returns:
            Dict with bid_prices, bid_volumes, ask_prices, ask_volumes,
            and full lob state tensor.
        """
        T, N = prices.shape
        D = cfg.lob_depth
        spread = cfg.spread_bps * 1e-4  # Convert bps to decimal

        bid_prices = np.zeros((T, N, D), dtype=np.float32)
        ask_prices = np.zeros((T, N, D), dtype=np.float32)
        bid_volumes = np.zeros((T, N, D), dtype=np.float32)
        ask_volumes = np.zeros((T, N, D), dtype=np.float32)

        for d in range(D):
            # Spread widens at deeper levels
            level_spread = spread * (1 + 0.5 * d)
            tick_size = prices * level_spread / D

            bid_prices[:, :, d] = prices - (d + 0.5) * tick_size
            ask_prices[:, :, d] = prices + (d + 0.5) * tick_size

            # Volumes: exponentially decaying with depth + noise
            decay = np.exp(-0.3 * d)
            bid_volumes[:, :, d] = np.maximum(
                np.random.normal(
                    cfg.volume_mean * decay,
                    cfg.volume_std * decay,
                    size=(T, N),
                ),
                1.0,
            )
            ask_volumes[:, :, d] = np.maximum(
                np.random.normal(
                    cfg.volume_mean * decay,
                    cfg.volume_std * decay,
                    size=(T, N),
                ),
                1.0,
            )

        # Construct full LOB state tensor: [T, N, 4D]
        lob_state = np.zeros((T, N, 4 * D), dtype=np.float32)
        for d in range(D):
            lob_state[:, :, 2 * d] = bid_prices[:, :, d]
            lob_state[:, :, 2 * d + 1] = bid_volumes[:, :, d]
        for d in range(D):
            lob_state[:, :, 2 * D + 2 * d] = ask_prices[:, :, d]
            lob_state[:, :, 2 * D + 2 * d + 1] = ask_volumes[:, :, d]

        return {
            "bid_prices": bid_prices,
            "bid_volumes": bid_volumes,
            "ask_prices": ask_prices,
            "ask_volumes": ask_volumes,
            "lob": lob_state,
        }

    def generate_torch(
        self,
        device: str = "cpu",
    ) -> Dict[str, Tensor]:
        """Generate synthetic data directly as PyTorch tensors.

        Args:
            device: Target device string.

        Returns:
            Same structure as ``generate()`` but with ``torch.Tensor`` values.
        """
        np_data = self.generate()
        torch_data = {}

        for key, val in np_data.items():
            if isinstance(val, np.ndarray):
                if val.dtype == bool:
                    torch_data[key] = torch.tensor(val, device=device)
                else:
                    torch_data[key] = torch.tensor(
                        val, dtype=torch.float32, device=device
                    )
            else:
                torch_data[key] = val

        return torch_data
