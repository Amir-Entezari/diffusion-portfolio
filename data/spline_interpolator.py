"""
Natural Cubic Spline Interpolation for Continuous-Time Mapping.

Reference: Proposal §7.4.3 — Continuous-Time Spline Interpolation.

Bridges the gap between discrete, asynchronous LOB ticks and the
continuous-time mathematics of Phase 0 (Neural CDE).

The normalised sequence of features F̃(t₀), F̃(t₁), ... is elevated
into a continuous, piecewise-differentiable trajectory:

    X(t) ∈ C¹([0, T], ℝ^F)

possessing continuous first and second derivatives.

This smooth, infinite-resolution trajectory is the exact mathematical
object fed into the Neural CDE integral:
    ∫ g_θ(H(s)) dX(s)

The spline coefficients are computed once and stored, allowing the CDE
solver to query the path derivative dX(t)/dt at any arbitrary time t.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import Tensor

logger = logging.getLogger(__name__)


class SplineInterpolator:
    """Natural Cubic Spline interpolator for continuous-time path construction.

    Converts discrete, irregularly-sampled time series into continuous C¹
    trajectories suitable for Neural CDE integration.

    The key insight from the proposal (§7.4.3):
        "This generates a continuous function X(t) ∈ C¹([0,T], ℝ^F),
         possessing continuous first and second derivatives."

    For the Neural CDE, we need both X(t) and dX(t)/dt at arbitrary
    query times.  The cubic spline provides both analytically.

    Mathematical Properties of Natural Cubic Splines:
        - C² continuity (continuous second derivative at knot points)
        - Minimal curvature (minimises ∫ |X''(t)|² dt)
        - Natural boundary conditions: X''(t₀) = X''(t_N) = 0

    Attributes:
        splines: Fitted SciPy CubicSpline objects (one per feature dimension).
        timestamps: Original discrete timestamps.
        coefficients: Torch tensor of spline coefficients for GPU evaluation.

    Example:
        >>> interp = SplineInterpolator()
        >>> timestamps = np.array([0.0, 0.1, 0.3, 0.5, 1.0])
        >>> values = np.random.randn(5, 12, 3)  # [T, N, F]
        >>> interp.fit(timestamps, values)
        >>> query_t = np.linspace(0, 1, 100)
        >>> continuous_path = interp.evaluate(query_t)
        >>> continuous_path.shape  # [100, 12, 3]
    """

    def __init__(self) -> None:
        self._spline: Optional[CubicSpline] = None
        self._timestamps: Optional[np.ndarray] = None
        self._t_min: float = 0.0
        self._t_max: float = 1.0
        self._data_shape: Optional[Tuple[int, ...]] = None

        # Cached torch coefficients for GPU evaluation
        self._torch_coeffs: Optional[Tensor] = None
        self._torch_breaks: Optional[Tensor] = None

    def fit(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        bc_type: str = "natural",
    ) -> "SplineInterpolator":
        """Fit natural cubic splines to discrete time-series data.

        Args:
            timestamps: Discrete time points, shape [T].
                Must be monotonically increasing.
            values: Feature values at each timestamp, shape [T, ...].
                Typically [T, N, F] for multi-asset multi-feature data.
            bc_type: Boundary condition type.
                ``"natural"``: X''(t₀) = X''(t_N) = 0 (zero curvature).
                ``"clamped"``: X'(t₀) = X'(t_N) = 0 (zero slope at ends).
                ``"not-a-knot"``: Continuous third derivative at second/
                    penultimate knots.

        Returns:
            self (for method chaining).
        """
        T = timestamps.shape[0]
        self._data_shape = values.shape[1:]

        # Flatten trailing dimensions for scipy
        flat_values = values.reshape(T, -1)  # [T, D]

        self._timestamps = timestamps.copy()
        self._t_min = timestamps[0]
        self._t_max = timestamps[-1]

        # Fit cubic spline: scipy handles multi-dimensional y natively
        self._spline = CubicSpline(
            timestamps,
            flat_values,
            bc_type=bc_type,
            extrapolate=False,
        )

        logger.info(
            f"Spline fitted: T={T} knots, "
            f"t ∈ [{self._t_min:.4f}, {self._t_max:.4f}], "
            f"data_shape={self._data_shape}"
        )

        return self

    def evaluate(
        self,
        query_times: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the spline at arbitrary continuous query times.

        Args:
            query_times: Times at which to evaluate, shape [T_query].
                Must be within [t_min, t_max].

        Returns:
            Interpolated values, shape [T_query, *data_shape].
        """
        if self._spline is None:
            raise RuntimeError("Must call fit() before evaluate().")

        flat_result = self._spline(query_times)  # [T_query, D]
        return flat_result.reshape(len(query_times), *self._data_shape)

    def evaluate_derivative(
        self,
        query_times: np.ndarray,
        order: int = 1,
    ) -> np.ndarray:
        """Evaluate the spline derivative at arbitrary times.

        For the Neural CDE, we need dX(t)/dt to compute the controlled
        integral: ∫ g_θ(H(s)) dX(s) = ∫ g_θ(H(s)) · X'(s) ds.

        Args:
            query_times: Times, shape [T_query].
            order: Derivative order (1 = velocity, 2 = acceleration).

        Returns:
            Derivative values, shape [T_query, *data_shape].
        """
        if self._spline is None:
            raise RuntimeError("Must call fit() before evaluate_derivative().")

        flat_deriv = self._spline(query_times, nu=order)  # [T_query, D]
        return flat_deriv.reshape(len(query_times), *self._data_shape)

    def to_torch_coefficients(
        self,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        """Extract spline coefficients as PyTorch tensors for GPU evaluation.

        The cubic spline between knots t_k and t_{k+1} is defined as:
            X(t) = c₃·(t−t_k)³ + c₂·(t−t_k)² + c₁·(t−t_k) + c₀

        This method extracts [c₃, c₂, c₁, c₀] per segment per dimension,
        allowing fully GPU-native evaluation without scipy.

        Args:
            device: Target device.

        Returns:
            Tuple of:
                - coefficients: shape [n_segments, 4, D] — spline coefficients.
                - breakpoints: shape [n_knots] — knot positions.
        """
        if self._spline is None:
            raise RuntimeError("Must call fit() before to_torch_coefficients().")

        # scipy CubicSpline stores coefficients as [order, n_segments, D]
        # order: c[0]=cubic, c[1]=quadratic, c[2]=linear, c[3]=constant
        c = self._spline.c  # [4, n_segments, D]
        breakpoints = self._spline.x  # [n_knots]

        self._torch_coeffs = torch.tensor(
            c.transpose(1, 0, 2),  # [n_segments, 4, D]
            dtype=torch.float32,
            device=device,
        )
        self._torch_breaks = torch.tensor(
            breakpoints,
            dtype=torch.float32,
            device=device,
        )

        return self._torch_coeffs, self._torch_breaks

    def evaluate_torch(
        self,
        query_times: Tensor,
        coefficients: Optional[Tensor] = None,
        breakpoints: Optional[Tensor] = None,
    ) -> Tensor:
        """GPU-native spline evaluation using precomputed coefficients.

        Evaluates the piecewise cubic polynomial:
            X(t) = c₃·Δt³ + c₂·Δt² + c₁·Δt + c₀
        where Δt = t − t_k and t_k is the left breakpoint of the segment.

        This is fully differentiable and runs entirely on GPU, enabling
        the Neural CDE solver to query the path without CPU roundtrips.

        Args:
            query_times: Query times, shape [T_query].
            coefficients: Spline coefficients, shape [n_segments, 4, D].
                If None, uses cached coefficients from to_torch_coefficients().
            breakpoints: Knot positions, shape [n_knots].

        Returns:
            Interpolated values, shape [T_query, *data_shape].
        """
        if coefficients is None:
            coefficients = self._torch_coeffs
        if breakpoints is None:
            breakpoints = self._torch_breaks

        if coefficients is None or breakpoints is None:
            raise RuntimeError(
                "Call to_torch_coefficients() first, or provide them explicitly."
            )

        # Find the segment index for each query time
        # searchsorted returns the index where query_times would be inserted
        segment_idx = torch.searchsorted(breakpoints, query_times, right=True) - 1
        segment_idx = segment_idx.clamp(0, coefficients.shape[0] - 1)

        # Compute Δt = t − t_k
        t_left = breakpoints[segment_idx]  # [T_query]
        dt = (query_times - t_left).unsqueeze(-1)  # [T_query, 1]

        # Gather coefficients for each segment
        c = coefficients[segment_idx]  # [T_query, 4, D]
        c3, c2, c1, c0 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]

        # Horner's method: c3·dt³ + c2·dt² + c1·dt + c0
        result = ((c3 * dt + c2) * dt + c1) * dt + c0  # [T_query, D]

        if self._data_shape is not None:
            result = result.reshape(len(query_times), *self._data_shape)

        return result

    def evaluate_derivative_torch(
        self,
        query_times: Tensor,
        coefficients: Optional[Tensor] = None,
        breakpoints: Optional[Tensor] = None,
    ) -> Tensor:
        """GPU-native evaluation of dX(t)/dt.

        Derivative of the cubic polynomial:
            dX/dt = 3·c₃·Δt² + 2·c₂·Δt + c₁

        Essential for the Neural CDE computation:
            dH(t) = f_θ(H(t))dt + g_θ(H(t))·(dX/dt)·dt

        Args:
            query_times: shape [T_query].
            coefficients: shape [n_segments, 4, D].
            breakpoints: shape [n_knots].

        Returns:
            Path derivatives, shape [T_query, *data_shape].
        """
        if coefficients is None:
            coefficients = self._torch_coeffs
        if breakpoints is None:
            breakpoints = self._torch_breaks

        if coefficients is None or breakpoints is None:
            raise RuntimeError("Call to_torch_coefficients() first.")

        segment_idx = torch.searchsorted(breakpoints, query_times, right=True) - 1
        segment_idx = segment_idx.clamp(0, coefficients.shape[0] - 1)

        t_left = breakpoints[segment_idx]
        dt = (query_times - t_left).unsqueeze(-1)

        c = coefficients[segment_idx]
        c3, c2, c1 = c[:, 0], c[:, 1], c[:, 2]

        # d/dt [c3·dt³ + c2·dt² + c1·dt + c0] = 3c3·dt² + 2c2·dt + c1
        result = (3.0 * c3 * dt + 2.0 * c2) * dt + c1

        if self._data_shape is not None:
            result = result.reshape(len(query_times), *self._data_shape)

        return result
