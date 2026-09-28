"""
Persistence Landscape Vectorisation.

Reference: Proposal §5.3.4 — Persistence Landscape Vectorisation.

Converts persistence diagrams into fixed-dimensional vectors in a
Hilbert space via the persistence landscape transform:

    λ_k(t, ε) = k-max_i max(0, min(ε − b_i, d_i − ε))

where:
    - k: landscape layer (k-th largest tent function envelope)
    - (b_i, d_i): birth-death pair from the persistence diagram
    - ε: filtration parameter (x-axis of the landscape)

Key Property:
    Persistence landscapes live in L^p(ℝ), a separable Banach space,
    making them compatible with standard statistical operations
    (mean, variance, inner products) that persistence diagrams lack.

The vectorised landscapes T_t ∈ ℝ^{K × n_points} are the final output
of Phase 1, ready for SGW cross-attention fusion with Phase 2.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class PersistenceLandscape:
    r"""Persistence landscape vectorisation.

    .. math::
        \lambda_k(\epsilon) = \text{k-max}_i \; \max\!\left(0,\;
        \min(\epsilon - b_i,\; d_i - \epsilon)\right)

    Each persistence pair (b_i, d_i) generates a tent function:
        - Rises linearly from 0 at ε = b_i to peak at ε = (b_i+d_i)/2
        - Falls linearly back to 0 at ε = d_i
        - Height = (d_i − b_i) / 2 (half the persistence)

    The k-th landscape layer is the k-th largest value among all
    tent functions at each ε.

    Args:
        n_layers: Number of landscape layers to compute.
        n_points: Number of discretisation points for ε.
        epsilon_range: (ε_min, ε_max) for the discretisation grid.

    Example:
        >>> pl = PersistenceLandscape(n_layers=5, n_points=100)
        >>> diagram = np.array([[0.1, 0.5], [0.2, 0.8], [0.3, 0.4]])
        >>> landscape = pl.compute(diagram)
        >>> landscape.shape  # [5, 100]
    """

    def __init__(
        self,
        n_layers: int = 5,
        n_points: int = 100,
        epsilon_range: Optional[tuple] = None,
    ) -> None:
        self.n_layers = n_layers
        self.n_points = n_points
        self.epsilon_range = epsilon_range

    def _tent_function(
        self,
        epsilon: np.ndarray,
        birth: float,
        death: float,
    ) -> np.ndarray:
        """Compute the tent function for a single persistence pair.

        Args:
            epsilon: Grid points, shape [n_points].
            birth: Birth time b_i.
            death: Death time d_i.

        Returns:
            Tent function values, shape [n_points].
        """
        return np.maximum(0.0, np.minimum(epsilon - birth, death - epsilon))

    def compute(
        self,
        diagram: np.ndarray,
        epsilon_range: Optional[tuple] = None,
    ) -> np.ndarray:
        """Compute persistence landscape from a persistence diagram.

        Args:
            diagram: Persistence pairs, shape [n_pairs, 2].
                Each row is (birth, death) with birth < death.
            epsilon_range: Override (ε_min, ε_max).

        Returns:
            Landscape array, shape [n_layers, n_points].
        """
        if len(diagram) == 0:
            return np.zeros((self.n_layers, self.n_points), dtype=np.float32)

        # Determine ε range
        if epsilon_range is not None:
            eps_min, eps_max = epsilon_range
        elif self.epsilon_range is not None:
            eps_min, eps_max = self.epsilon_range
        else:
            eps_min = diagram[:, 0].min()
            eps_max = diagram[:, 1].max()

        epsilon = np.linspace(eps_min, eps_max, self.n_points)

        # Compute all tent functions
        n_pairs = len(diagram)
        tents = np.zeros((n_pairs, self.n_points), dtype=np.float64)
        for i in range(n_pairs):
            tents[i] = self._tent_function(epsilon, diagram[i, 0], diagram[i, 1])

        # k-max: sort tent values at each ε, take k-th largest
        sorted_tents = np.sort(tents, axis=0)[::-1]  # Descending

        landscape = np.zeros((self.n_layers, self.n_points), dtype=np.float32)
        for k in range(min(self.n_layers, n_pairs)):
            landscape[k] = sorted_tents[k]

        return landscape

    def compute_multi_dim(
        self,
        diagrams: Dict[int, np.ndarray],
        epsilon_range: Optional[tuple] = None,
    ) -> np.ndarray:
        """Compute landscapes for all homological dimensions and concatenate.

        Args:
            diagrams: {dim: persistence_pairs} from VR filtration.
            epsilon_range: Override ε range.

        Returns:
            Concatenated landscapes, shape [n_dims × n_layers, n_points].
        """
        all_landscapes = []
        for dim in sorted(diagrams.keys()):
            landscape = self.compute(diagrams[dim], epsilon_range)
            all_landscapes.append(landscape)

        return np.concatenate(all_landscapes, axis=0)


class PersistenceLandscapeLayer(nn.Module):
    """Neural network wrapper for persistence landscape computation.

    Wraps the landscape computation in an nn.Module for integration
    with the PyTorch computation graph. Since TDA operations are not
    natively differentiable, this module uses a learned linear projection
    to embed the landscape vectors into a neural-friendly representation.

    Args:
        n_homology_dims: Number of homological dimensions (e.g., 3 for β₀,β₁,β₂).
        n_layers: Landscape layers per dimension.
        n_points: Discretisation points.
        output_dim: Dimension of the output embedding.
        max_edge_length: ε_max for filtration range.
    """

    def __init__(
        self,
        n_homology_dims: int = 3,
        n_layers: int = 5,
        n_points: int = 100,
        output_dim: int = 128,
        max_edge_length: float = 2.0,
    ) -> None:
        super().__init__()

        self.landscape_computer = PersistenceLandscape(
            n_layers=n_layers,
            n_points=n_points,
            epsilon_range=(0.0, max_edge_length),
        )

        input_dim = n_homology_dims * n_layers * n_points
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.n_homology_dims = n_homology_dims
        self.n_layers = n_layers
        self.n_points = n_points

    def forward(
        self,
        persistence_diagrams: List[Dict[int, np.ndarray]],
    ) -> Tensor:
        """Compute landscape embeddings for a batch of diagrams.

        Args:
            persistence_diagrams: List of B diagram dicts.

        Returns:
            Landscape embeddings T_t, shape [B, output_dim].
        """
        batch_landscapes = []

        for diagrams in persistence_diagrams:
            landscape = self.landscape_computer.compute_multi_dim(diagrams)
            batch_landscapes.append(landscape.flatten())

        # Stack and convert to tensor
        landscape_tensor = torch.tensor(
            np.stack(batch_landscapes),
            dtype=torch.float32,
        )

        # Move to same device as projection weights
        device = next(self.projection.parameters()).device
        landscape_tensor = landscape_tensor.to(device)

        # Project to output dimension
        return self.projection(landscape_tensor)
