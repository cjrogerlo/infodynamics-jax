# infodynamics_jax/data/conditioning.py
"""
Conditioning channel utilities for patch-based models.

Provides position encoding and global context channels to give patches
awareness of their spatial location and the overall image content.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .patchify import PatchifyResult


def build_position_channels(
    result: PatchifyResult,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Build position encoding channels for patches.

    Creates 2-channel position encoding (y, x) normalized to [-1, 1] based on
    patch center coordinates in the original image.

    Args:
        result: PatchifyResult from patchify()
        scale: Scale factor for position values

    Returns:
        (N_patches, ps, ps, 2) array of position channels
    """
    _, H, W, _ = result.full_shape
    ps = result.patch_size
    positions = result.positions

    # Compute patch centers
    centers = positions.astype(np.float32).copy()
    centers[:, 0] = centers[:, 0] + (ps - 1) * 0.5  # y center
    centers[:, 1] = centers[:, 1] + (ps - 1) * 0.5  # x center

    # Normalize to [-1, 1]
    y_norm = centers[:, 0] / max(1.0, H - 1) * 2.0 - 1.0
    x_norm = centers[:, 1] / max(1.0, W - 1) * 2.0 - 1.0

    # Broadcast to patch shape
    N = result.n_patches
    pos_channels = np.zeros((N, ps, ps, 2), dtype=np.float32)
    pos_channels[..., 0] = y_norm[:, None, None]
    pos_channels[..., 1] = x_norm[:, None, None]

    return pos_channels * scale


def build_global_channels(
    images: np.ndarray,
    result: PatchifyResult,
    mode: Literal["lowpass", "mean"] = "lowpass",
    scale: float = 1.0,
) -> np.ndarray:
    """
    Build global context channels for patches.

    Creates channels that encode global image information, giving each patch
    awareness of the overall image content.

    Args:
        images: (N_images, H, W, C) original images
        result: PatchifyResult from patchify()
        mode: "lowpass" = area-downsampled image; "mean" = per-image mean color
        scale: Scale factor for values

    Returns:
        (N_patches, ps, ps, C) array of global channels
    """
    N_img, H, W, C = images.shape
    ps = result.patch_size

    if mode == "mean":
        # Global mean per image
        global_feat = np.mean(images, axis=(1, 2), keepdims=True)  # (N_img, 1, 1, C)
        global_feat = np.tile(global_feat, (1, ps, ps, 1))  # (N_img, ps, ps, C)
    elif mode == "lowpass":
        # Area downsample to patch size
        if H % ps == 0 and W % ps == 0:
            fy = H // ps
            fx = W // ps
            global_feat = images.reshape(N_img, ps, fy, ps, fx, C).mean(axis=(2, 4))
        else:
            # Fallback: strided sampling
            global_feat = images[:, ::max(1, H // ps), ::max(1, W // ps), :]
            global_feat = global_feat[:, :ps, :ps, :]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Expand to all patches using img_ids
    global_channels = global_feat[result.img_ids]  # (N_patches, ps, ps, C)

    return global_channels.astype(np.float32) * scale


def concat_conditioning(
    patches: np.ndarray,
    position_channels: Optional[np.ndarray] = None,
    global_channels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Concatenate conditioning channels to patches.

    Args:
        patches: (N, ps, ps, C) base patches
        position_channels: (N, ps, ps, 2) position encoding (optional)
        global_channels: (N, ps, ps, C_global) global context (optional)

    Returns:
        (N, ps, ps, C + C_pos + C_global) concatenated array
    """
    arrays = [patches]
    if position_channels is not None:
        arrays.append(position_channels)
    if global_channels is not None:
        arrays.append(global_channels)

    return np.concatenate(arrays, axis=-1)


def apply_conditioning(
    images: np.ndarray,
    result: PatchifyResult,
    add_position: bool = True,
    position_scale: float = 1.0,
    add_global: bool = True,
    global_mode: Literal["lowpass", "mean"] = "lowpass",
    global_scale: float = 1.0,
) -> np.ndarray:
    """
    Apply position and global conditioning to patches.

    Convenience function that combines build_position_channels,
    build_global_channels, and concat_conditioning.

    Args:
        images: (N_images, H, W, C) original images
        result: PatchifyResult from patchify()
        add_position: Add position encoding channels
        position_scale: Scale for position values
        add_global: Add global context channels
        global_mode: "lowpass" or "mean"
        global_scale: Scale for global values

    Returns:
        (N_patches, ps, ps, C') conditioned patches
    """
    patches = result.patches

    pos_ch = None
    if add_position:
        pos_ch = build_position_channels(result, scale=position_scale)

    global_ch = None
    if add_global:
        global_ch = build_global_channels(
            images, result, mode=global_mode, scale=global_scale
        )

    return concat_conditioning(patches, pos_ch, global_ch)
