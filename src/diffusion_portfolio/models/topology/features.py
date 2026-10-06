"""Topological features from Neural CDE latent trajectories."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _persistence_statistics(
    intervals: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    """Convert persistence intervals to a fixed-size feature vector."""

    if intervals.size == 0:
        lifetimes = np.empty(
            0,
            dtype=np.float64,
        )

    else:
        intervals = np.asarray(
            intervals,
            dtype=np.float64,
        )

        finite = np.isfinite(
            intervals[:, 1]
        )

        intervals = intervals[
            finite
        ]

        lifetimes = (
            intervals[:, 1]
            - intervals[:, 0]
        )

        lifetimes = lifetimes[
            lifetimes > 0
        ]

    if len(lifetimes) == 0:
        summary = np.zeros(
            5,
            dtype=np.float64,
        )

        top = np.zeros(
            top_k,
            dtype=np.float64,
        )

        return np.concatenate(
            [
                summary,
                top,
            ]
        )

    lifetimes = np.sort(
        lifetimes
    )[::-1]

    total = float(
        lifetimes.sum()
    )

    probabilities = (
        lifetimes
        / total
    )

    entropy = float(
        -np.sum(
            probabilities
            * np.log(
                probabilities
                + 1e-12
            )
        )
    )

    summary = np.asarray(
        [
            float(len(lifetimes)),
            float(lifetimes.mean()),
            float(lifetimes.std()),
            float(lifetimes.max()),
            entropy,
        ],
        dtype=np.float64,
    )

    top = np.zeros(
        top_k,
        dtype=np.float64,
    )

    count = min(
        top_k,
        len(lifetimes),
    )

    top[:count] = lifetimes[
        :count
    ]

    return np.concatenate(
        [
            summary,
            top,
        ]
    )


def trajectory_persistence_features(
    trajectory: Tensor,
    *,
    max_homology_dim: int = 1,
    top_k: int = 3,
) -> Tensor:
    """Extract fixed-dimensional persistence features.

    Parameters
    ----------
    trajectory:
        CDE latent trajectory with shape
        [batch, time, hidden_dim].

    max_homology_dim:
        Highest homology dimension summarized.

    top_k:
        Number of largest persistence lifetimes retained
        for each homology dimension.

    Returns
    -------
    Tensor
        Shape:
        [batch, (max_homology_dim + 1) * (5 + top_k)]
    """

    if trajectory.ndim != 3:
        raise ValueError(
            "trajectory must have shape "
            "[batch, time, hidden_dim]"
        )

    if trajectory.shape[1] < 2:
        raise ValueError(
            "trajectory must contain at least "
            "two time points"
        )

    if max_homology_dim < 0:
        raise ValueError(
            "max_homology_dim cannot be negative"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be positive"
        )

    try:
        import gudhi
    except ImportError as error:
        raise ImportError(
            "TDA features require the topology "
            "optional dependency. Install with "
            "`pip install -e '.[topology]'`."
        ) from error

    values = (
        trajectory
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    all_features = []

    for sample in values:
        complex_ = gudhi.RipsComplex(
            points=sample,
        )

        simplex_tree = (
            complex_
            .create_simplex_tree(
                max_dimension=(
                    max_homology_dim
                    + 1
                )
            )
        )

        simplex_tree.persistence()

        sample_features = []

        for dimension in range(
            max_homology_dim + 1
        ):
            intervals = (
                simplex_tree
                .persistence_intervals_in_dimension(
                    dimension
                )
            )

            sample_features.append(
                _persistence_statistics(
                    intervals,
                    top_k=top_k,
                )
            )

        all_features.append(
            np.concatenate(
                sample_features
            )
        )

    features = np.stack(
        all_features,
        axis=0,
    )

    return torch.as_tensor(
        features,
        dtype=trajectory.dtype,
        device=trajectory.device,
    )