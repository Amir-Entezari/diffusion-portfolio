"""SGW Interface: Sliced Gromov-Wasserstein Manifold Alignment (§5.5)."""

from models.sgw_interface.sliced_gw import SlicedGromovWasserstein
from models.sgw_interface.sgw_cross_attention import (
    SGWCrossAttention, SGWFusionModule,
)

__all__ = [
    "SlicedGromovWasserstein",
    "SGWCrossAttention",
    "SGWFusionModule",
]
