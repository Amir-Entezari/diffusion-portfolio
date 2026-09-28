"""
Device Management for GPU/CPU Placement.

Provides a centralised device manager for consistent tensor placement
across all phases of the architecture.  Handles:
    - Automatic CUDA detection and selection.
    - Explicit device placement with ``.to(device)`` helpers.
    - VRAM monitoring for adaptive batch sizing.
    - Multi-GPU support via device indices.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class DeviceManager:
    """Centralised device manager for the framework.

    Ensures all tensors and modules are placed on the correct device
    throughout the pipeline.

    Attributes:
        device: The ``torch.device`` object used for all computations.
        is_cuda: Whether CUDA is available and being used.

    Example:
        >>> dm = DeviceManager("cuda")
        >>> tensor = dm.place(torch.randn(3, 3))
        >>> tensor.device
        device(type='cuda', index=0)
    """

    def __init__(self, device_str: str = "cuda") -> None:
        """Initialise the device manager.

        Args:
            device_str: Device specification.  Supported values:
                - ``"cuda"``: Auto-select first available CUDA device.
                - ``"cuda:0"``, ``"cuda:1"``, ...: Specific GPU index.
                - ``"cpu"``: CPU-only execution.
        """
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available. Falling back to CPU. "
                "This will significantly impact performance."
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)

        self.is_cuda = self.device.type == "cuda"

        if self.is_cuda:
            gpu_name = torch.cuda.get_device_name(self.device)
            vram_total = torch.cuda.get_device_properties(self.device).total_mem
            vram_gb = vram_total / (1024 ** 3)
            logger.info(
                f"Device: {gpu_name} | VRAM: {vram_gb:.1f} GB | "
                f"CUDA Version: {torch.version.cuda}"
            )
        else:
            logger.info("Device: CPU")

    def place(self, tensor_or_module: Tensor | torch.nn.Module) -> Tensor | torch.nn.Module:
        """Move a tensor or module to the managed device.

        Args:
            tensor_or_module: PyTorch tensor or ``nn.Module`` to place.

        Returns:
            The input moved to ``self.device``.
        """
        return tensor_or_module.to(self.device)

    def get_vram_usage(self) -> dict[str, float]:
        """Get current VRAM usage statistics (CUDA only).

        Returns:
            Dictionary with keys:
                - ``allocated_gb``: Currently allocated VRAM.
                - ``reserved_gb``: Currently reserved (cached) VRAM.
                - ``max_allocated_gb``: Peak allocated VRAM.
                - ``total_gb``: Total VRAM on the device.

        Raises:
            RuntimeError: If not running on CUDA.
        """
        if not self.is_cuda:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0,
                    "max_allocated_gb": 0.0, "total_gb": 0.0}

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(self.device).total_mem / (1024 ** 3)

        return {
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "max_allocated_gb": round(max_allocated, 3),
            "total_gb": round(total, 3),
        }

    def empty_cache(self) -> None:
        """Explicitly free unused cached VRAM."""
        if self.is_cuda:
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"DeviceManager(device={self.device})"


def get_device(device_str: str = "cuda") -> torch.device:
    """Simple helper to resolve a device string to a ``torch.device``.

    Args:
        device_str: ``"cuda"``, ``"cpu"``, or ``"cuda:N"``.

    Returns:
        Resolved ``torch.device``.
    """
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)
