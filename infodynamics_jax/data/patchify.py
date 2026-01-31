# infodynamics_jax/data/patchify.py
"""
Patch extraction and reassembly utilities.

Provides sliding-window patchification with overlap and Hanning-window blending
for seamless reassembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class PatchifyCFG:
    """
    Configuration for image patchification.

    Attributes:
        patch_size: Size of extracted patches (square)
        stride: Stride between patches (default: patch_size // 2 for 50% overlap)
        max_patches: Maximum number of patches to keep (None = keep all)
        shuffle: Whether to shuffle patches
        seed: Random seed for shuffling
    """
    patch_size: int = 8
    stride: Optional[int] = None
    max_patches: Optional[int] = None
    shuffle: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.stride is None:
            self.stride = max(1, self.patch_size // 2)


@dataclass
class PatchifyResult:
    """
    Result of patchification operation.

    Attributes:
        patches: (N_patches, ps, ps, C) array of patches
        positions: (N_patches, 2) array of (y, x) positions
        img_ids: (N_patches,) array of source image indices
        indices: (N_patches,) array of selected indices (for reproducibility)
        cfg: Configuration used
        full_shape: Original image shape (N, H, W, C)
    """
    patches: np.ndarray
    positions: np.ndarray
    img_ids: np.ndarray
    indices: np.ndarray
    cfg: PatchifyCFG
    full_shape: Tuple[int, int, int, int]

    @property
    def n_patches(self) -> int:
        return self.patches.shape[0]

    @property
    def patch_size(self) -> int:
        return self.patches.shape[1]

    @property
    def n_channels(self) -> int:
        return self.patches.shape[3]

    @property
    def patches_per_image(self) -> int:
        """Number of patches per original image."""
        return len(np.unique(self.positions, axis=0))


def patchify(
    images: np.ndarray,
    cfg: PatchifyCFG,
) -> PatchifyResult:
    """
    Extract patches from a batch of images.

    Args:
        images: (N, H, W, C) array of images
        cfg: PatchifyCFG configuration

    Returns:
        PatchifyResult containing patches and metadata
    """
    N, H, W, C = images.shape
    ps = cfg.patch_size
    stride = cfg.stride if cfg.stride is not None else max(1, ps // 2)

    if ps > min(H, W):
        raise ValueError(f"patch_size {ps} > image size ({H}, {W})")

    # Extract all patches
    patches_list = []
    positions_list = []
    for i in range(0, H - ps + 1, stride):
        for j in range(0, W - ps + 1, stride):
            patches_list.append(images[:, i:i+ps, j:j+ps, :])
            positions_list.append((i, j))

    if len(patches_list) == 0:
        raise ValueError("No patches extracted; check patch_size/stride")

    # Stack: (P, N, ps, ps, C) -> reshape to (N*P, ps, ps, C)
    patches_stacked = np.stack(patches_list, axis=1)  # (N, P, ps, ps, C)
    P = patches_stacked.shape[1]
    positions = np.array(positions_list, dtype=np.int32)  # (P, 2)

    # Flatten to (N*P, ps, ps, C)
    patches_flat = patches_stacked.reshape(-1, ps, ps, C)

    # Build img_ids and position arrays for all patches
    img_ids = np.repeat(np.arange(N)[:, None], P, axis=1).reshape(-1)
    positions_flat = np.tile(positions, (N, 1))

    # Shuffle and subsample
    rng = np.random.RandomState(cfg.seed)
    indices = np.arange(patches_flat.shape[0])
    if cfg.shuffle:
        rng.shuffle(indices)

    if cfg.max_patches is not None:
        max_p = min(cfg.max_patches, len(indices))
        indices = indices[:max_p]

    patches_out = patches_flat[indices]
    positions_out = positions_flat[indices]
    img_ids_out = img_ids[indices]

    return PatchifyResult(
        patches=patches_out,
        positions=positions_out,
        img_ids=img_ids_out,
        indices=indices,
        cfg=cfg,
        full_shape=(N, H, W, C),
    )


def reassemble(
    patches: np.ndarray,
    positions: np.ndarray,
    full_H: int,
    full_W: int,
    patch_size: int,
    use_window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reassemble patches into a full image using blending.

    Args:
        patches: (K, ps, ps, C) array of patches
        positions: (K, 2) array of (y, x) positions
        full_H, full_W: Target image dimensions
        patch_size: Patch size
        use_window: Use Hanning window for blending (recommended)

    Returns:
        Tuple of (image, weights):
        - image: (H, W, C) reassembled image
        - weights: (H, W, 1) weight map showing coverage
    """
    ps = patch_size
    C = patches.shape[-1]
    canvas = np.zeros((full_H, full_W, C), dtype=np.float32)
    weight = np.zeros((full_H, full_W, 1), dtype=np.float32)

    if use_window:
        w1 = np.hanning(ps).astype(np.float32)
        w2 = np.outer(w1, w1)
        w2 = w2 / (np.max(w2) + 1e-8)
        w = w2[..., None]
    else:
        w = np.ones((ps, ps, 1), dtype=np.float32)

    for k in range(patches.shape[0]):
        i, j = positions[k]
        canvas[i:i+ps, j:j+ps, :] += patches[k] * w
        weight[i:i+ps, j:j+ps, :] += w

    # Normalize by weights
    image = canvas / (weight + 1e-8)
    return image, weight


def reassemble_batch(
    result: PatchifyResult,
    patches: np.ndarray,
    img_id: int,
    use_window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reassemble patches for a specific image from a PatchifyResult.

    Args:
        result: PatchifyResult from patchify()
        patches: (N_patches, ps, ps, C) array of patches (may be different from result.patches)
        img_id: Image ID to reassemble
        use_window: Use Hanning window for blending

    Returns:
        Tuple of (image, weights)
    """
    _, H, W, _ = result.full_shape
    ps = result.patch_size

    # Select patches for this image
    mask = result.img_ids == img_id
    if not np.any(mask):
        raise ValueError(f"No patches found for img_id={img_id}")

    selected_patches = patches[mask]
    selected_positions = result.positions[mask]

    return reassemble(selected_patches, selected_positions, H, W, ps, use_window)
