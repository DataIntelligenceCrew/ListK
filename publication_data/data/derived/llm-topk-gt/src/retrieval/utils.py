"""Shared utilities for retrieval module.

This module provides common functions used across retrievers.
"""

import logging
from typing import Optional

logger: logging.Logger = logging.getLogger(__name__)


def get_device(preferred: Optional[str] = None) -> str:
    """Get the compute device to use.

    Parameters
    ----------
    preferred : str, optional
        Preferred device ('cuda', 'cpu', or 'mps'). Auto-detect if None.

    Returns
    -------
    str
        Device string for PyTorch.
    """
    # Import torch lazily to avoid import errors if not installed
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, using CPU")
        return "cpu"

    if preferred is not None:
        return preferred

    if torch.cuda.is_available():
        device: str = "cuda"
        device_name: str = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS device")
    else:
        device = "cpu"
        logger.info("Using CPU device")

    return device


def get_gpu_memory_gb() -> float:
    """Get total GPU memory in GB.

    Returns
    -------
    float
        GPU memory in GB, or 0 if no GPU available.
    """
    try:
        import torch
    except ImportError:
        return 0.0

    if torch.cuda.is_available():
        total_bytes: int = torch.cuda.get_device_properties(0).total_memory
        return total_bytes / (1024**3)
    return 0.0


def estimate_batch_size(
    model_memory_gb: float,
    available_memory_gb: Optional[float] = None,
    safety_factor: float = 0.7,
) -> int:
    """Estimate appropriate batch size based on available memory.

    Parameters
    ----------
    model_memory_gb : float
        Estimated model memory usage in GB.
    available_memory_gb : float, optional
        Available GPU memory in GB. Auto-detected if None.
    safety_factor : float, optional
        Safety factor (0-1) for memory headroom. Default 0.7.

    Returns
    -------
    int
        Recommended batch size.
    """
    if available_memory_gb is None:
        available_memory_gb = get_gpu_memory_gb()

    if available_memory_gb == 0:
        # CPU fallback - use conservative batch size
        return 16

    usable_memory: float = available_memory_gb * safety_factor - model_memory_gb
    # Rough estimate: ~0.1 GB per batch item for dense models
    estimated_batch: int = max(1, int(usable_memory / 0.1))
    return min(estimated_batch, 64)  # Cap at 64
