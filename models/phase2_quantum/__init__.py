"""Phase 2: Quantum Information Geometry (§5.4)."""

from models.phase2_quantum.density_matrix import DensityMatrixBuilder
from models.phase2_quantum.bures_metric import (
    bures_distance, bures_distance_squared, quantum_fidelity,
)
from models.phase2_quantum.log_map import RiemannianLogMap

__all__ = [
    "DensityMatrixBuilder",
    "bures_distance",
    "bures_distance_squared",
    "quantum_fidelity",
    "RiemannianLogMap",
]
