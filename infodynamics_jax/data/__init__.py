# infodynamics_jax/data/__init__.py
"""
Data preprocessing utilities for image-based GP models.

This module provides:
- patchify: Extract and reassemble image patches
- conditioning: Position and global conditioning channels
"""
from .patchify import (
    PatchifyCFG,
    PatchifyResult,
    patchify,
    reassemble,
    reassemble_batch,
)
from .conditioning import (
    build_position_channels,
    build_global_channels,
    concat_conditioning,
    apply_conditioning,
)
from .representations import build_observation, ObservationCFG

__all__ = [
    # Patchify
    "PatchifyCFG",
    "PatchifyResult",
    "patchify",
    "reassemble",
    "reassemble_batch",
    # Conditioning
    "build_position_channels",
    "build_global_channels",
    "concat_conditioning",
    "apply_conditioning",
    # Representations
    "build_observation",
    "ObservationCFG",
]
