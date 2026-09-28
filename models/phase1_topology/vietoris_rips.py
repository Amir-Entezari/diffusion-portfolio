"""
Vietoris-Rips Filtration Construction.

Reference: Proposal §5.3.2 — Vietoris-Rips Complex.

Constructs the filtered simplicial complex:

    VR(ε) = {σ ⊆ Y(t) | d(u,v) ≤ ε, ∀ u,v ∈ σ}

A simplex σ is included in VR(ε) if and only if every pair of its
vertices has pairwise distance at most ε.  As ε increases from 0 to ∞,
we get an increasing sequence of complexes (the filtration):

    VR(ε₀) ⊆ VR(ε₁) ⊆ ... ⊆ VR(ε_max)

This filtration is the input to persistent homology computation.

Backend: GUDHI or giotto-tda (configurable).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Conditional imports for TDA backends
try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

try:
    from gtda.homology import VietorisRipsPersistence
    HAS_GIOTTO = True
except ImportError:
    HAS_GIOTTO = False


class VietorisRipsFiltration:
    """Vietoris-Rips filtration construction from point clouds.

    Given a point cloud Y(t) ∈ ℝ^{m × h} (from Takens embedding),
    constructs the VR filtration and extracts persistence diagrams.

    Args:
        max_edge_length: ε_max — maximum filtration parameter.
        max_homology_dim: Maximum homological dimension (0, 1, or 2).
        backend: 'gudhi' or 'giotto-tda'.

    Example:
        >>> vr = VietorisRipsFiltration(max_edge_length=2.0, max_homology_dim=2)
        >>> point_cloud = np.random.randn(20, 5)  # 20 points in ℝ^5
        >>> diagrams = vr.compute_persistence(point_cloud)
    """

    def __init__(
        self,
        max_edge_length: float = 2.0,
        max_homology_dim: int = 2,
        backend: str = "gudhi",
    ) -> None:
        self.max_edge_length = max_edge_length
        self.max_homology_dim = max_homology_dim
        self.backend = backend

        if backend == "gudhi" and not HAS_GUDHI:
            if HAS_GIOTTO:
                logger.warning("GUDHI not available, falling back to giotto-tda.")
                self.backend = "giotto-tda"
            else:
                logger.warning(
                    "Neither GUDHI nor giotto-tda available. "
                    "Using distance-matrix fallback (limited functionality)."
                )
                self.backend = "fallback"
        elif backend == "giotto-tda" and not HAS_GIOTTO:
            if HAS_GUDHI:
                logger.warning("giotto-tda not available, falling back to GUDHI.")
                self.backend = "gudhi"
            else:
                self.backend = "fallback"

    def compute_persistence_gudhi(
        self,
        point_cloud: np.ndarray,
    ) -> List[Tuple[int, Tuple[float, float]]]:
        """Compute persistence diagrams using GUDHI.

        Args:
            point_cloud: Points, shape [n_points, dim].

        Returns:
            List of (dimension, (birth, death)) tuples.
        """
        rips = gudhi.RipsComplex(
            points=point_cloud.tolist(),
            max_edge_length=self.max_edge_length,
        )
        simplex_tree = rips.create_simplex_tree(
            max_dimension=self.max_homology_dim + 1
        )
        simplex_tree.compute_persistence()
        return simplex_tree.persistence()

    def compute_persistence_fallback(
        self,
        point_cloud: np.ndarray,
    ) -> List[Tuple[int, Tuple[float, float]]]:
        """Simplified persistence using distance matrix thresholding.

        This fallback does NOT compute true persistent homology — it only
        provides approximate β₀ (connected components) via single-linkage
        clustering. Used when no TDA library is installed.

        Args:
            point_cloud: Points, shape [n_points, dim].

        Returns:
            Approximate persistence pairs for dimension 0 only.
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import single, fcluster

        dists = pdist(point_cloud)
        Z = single(dists)

        pairs = []
        n = len(point_cloud)
        # Each merge in single-linkage creates a persistence pair
        for i in range(len(Z)):
            birth = 0.0
            death = min(Z[i, 2], self.max_edge_length)
            if death > birth:
                pairs.append((0, (birth, death)))

        # Add a single infinite bar for β₀
        pairs.append((0, (0.0, float('inf'))))

        return pairs

    def compute_persistence(
        self,
        point_cloud: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Compute persistence diagrams for all dimensions.

        Args:
            point_cloud: Points, shape [n_points, dim].

        Returns:
            Dictionary mapping dimension k → array of (birth, death) pairs.
            Shape: {k: [n_pairs_k, 2]}.
        """
        if self.backend == "gudhi":
            raw = self.compute_persistence_gudhi(point_cloud)
        else:
            raw = self.compute_persistence_fallback(point_cloud)

        # Organise by dimension
        diagrams: Dict[int, list] = {k: [] for k in range(self.max_homology_dim + 1)}
        for dim, (birth, death) in raw:
            if dim <= self.max_homology_dim:
                if death == float('inf'):
                    death = self.max_edge_length  # Cap infinite bars
                diagrams[dim].append([birth, death])

        # Convert to numpy arrays
        result = {}
        for k in range(self.max_homology_dim + 1):
            if diagrams[k]:
                result[k] = np.array(diagrams[k], dtype=np.float32)
            else:
                result[k] = np.zeros((0, 2), dtype=np.float32)

        return result

    def compute_persistence_batch(
        self,
        point_clouds: np.ndarray,
    ) -> List[Dict[int, np.ndarray]]:
        """Compute persistence for a batch of point clouds.

        Args:
            point_clouds: Shape [B, n_points, dim].

        Returns:
            List of B persistence diagram dictionaries.
        """
        batch_diagrams = []
        for i in range(point_clouds.shape[0]):
            diag = self.compute_persistence(point_clouds[i])
            batch_diagrams.append(diag)
        return batch_diagrams

    def extract_betti_numbers(
        self,
        diagrams: Dict[int, np.ndarray],
        epsilon: float,
    ) -> Dict[int, int]:
        r"""Extract Betti numbers at a specific filtration scale.

        .. math::
            \beta_k(\epsilon) = \#\{(b_i, d_i) \in \mathcal{D}_k \mid b_i \leq \epsilon < d_i\}

        Args:
            diagrams: Persistence diagrams from compute_persistence.
            epsilon: Filtration scale at which to count.

        Returns:
            {k: β_k} — Betti numbers per dimension.
        """
        betti = {}
        for k, pairs in diagrams.items():
            if len(pairs) == 0:
                betti[k] = 0
            else:
                alive = np.sum((pairs[:, 0] <= epsilon) & (pairs[:, 1] > epsilon))
                betti[k] = int(alive)
        return betti
