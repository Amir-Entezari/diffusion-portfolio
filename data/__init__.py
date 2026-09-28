"""Data ingestion and preprocessing sub-package.

Implements the complete data pipeline from §7.4:
    1. LOB parsing → raw tensor X_LOB(t)
    2. Feature engineering → [P_micro, OFI, S_skew]
    3. Causal normalisation → rolling Z-score
    4. Spline interpolation → continuous C¹ path
    5. PyTorch Dataset / DataLoader wrappers
    6. Synthetic data generator for development
"""

from data.lob_parser import LOBParser
from data.feature_engineer import FeatureEngineer
from data.causal_normalizer import CausalNormalizer
from data.spline_interpolator import SplineInterpolator
from data.dataset import MarketDataset, create_dataloaders
from data.synthetic_generator import SyntheticJumpDiffusionGenerator

__all__ = [
    "LOBParser",
    "FeatureEngineer",
    "CausalNormalizer",
    "SplineInterpolator",
    "MarketDataset",
    "create_dataloaders",
    "SyntheticJumpDiffusionGenerator",
]
