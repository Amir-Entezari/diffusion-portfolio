"""
Global Seed Management for Reproducibility.

Sets seeds for all sources of randomness used in the framework:
    - Python's ``random`` module
    - NumPy's random number generator
    - PyTorch CPU and CUDA generators
    - CUDA deterministic algorithms (optional)

Reproducibility is critical for:
    1. Ablation studies comparing phases in isolation.
    2. Debugging gradient flow issues in the bi-level TTSA loop.
    3. Generating identical synthetic scenarios across runs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: The global seed value.
        deterministic: If True, enforce deterministic CUDA operations.
            This may significantly degrade performance due to:
                - Disabling CUDA convolution autotuning.
                - Forcing deterministic scatter/gather operations.

    Note:
        Full determinism on CUDA is not always possible.  Some operations
        (e.g., atomicAdd in scatter) are inherently non-deterministic.
        Set ``deterministic=True`` only for debugging, not production.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+ deterministic flag
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Some operations don't have deterministic implementations
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.use_deterministic_algorithms(True)
        logger.warning(
            "Deterministic mode enabled. Performance may be degraded. "
            "Only use for debugging / reproducibility verification."
        )
    else:
        torch.backends.cudnn.benchmark = True  # Enable autotuning

    logger.info(f"Global seed set to {seed} (deterministic={deterministic})")
