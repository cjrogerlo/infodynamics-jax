# infodynamics_jax/infodynamics/likelihoods/linear_ops/sobel.py
"""Sobel linear operator."""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .stack import LinearOp


def _sobel_kernels(dtype):
    kx = jnp.array([[-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0]], dtype=dtype)
    ky = jnp.array([[-1.0, -2.0, -1.0],
                    [ 0.0,  0.0,  0.0],
                    [ 1.0,  2.0,  1.0]], dtype=dtype)
    return kx, ky


def _conv2d_same(x, k):
    # x: (N, H, W, C)
    k = jnp.asarray(k, dtype=x.dtype)
    kh, kw = k.shape
    k4 = k.reshape((kh, kw, 1, 1))
    if x.shape[-1] == 1:
        k4c = k4
        fg = 1
    else:
        k4c = jnp.tile(k4, (1, 1, 1, x.shape[-1]))
        fg = x.shape[-1]
    return jax.lax.conv_general_dilated(
        x, k4c, window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=fg,
    )


def sobel_op(H: int, W: int, C: int, use_luma: bool = True) -> LinearOp:
    """Build a Sobel operator mapping y -> [Sx y, Sy y]."""
    out_channels = 2 if use_luma else 2 * C
    out_dim = H * W * out_channels

    def _to_luma(img):
        img = img[..., :3]
        return (0.2989 * img[..., 0]
                + 0.5870 * img[..., 1]
                + 0.1140 * img[..., 2])

    def _apply_flat(y_flat: jnp.ndarray) -> jnp.ndarray:
        N = y_flat.shape[0]
        img = y_flat.reshape((N, H, W, C))
        kx, ky = _sobel_kernels(img.dtype)
        if use_luma:
            lum = _to_luma(img)[..., None]
            gx = _conv2d_same(lum, kx)[..., 0]
            gy = _conv2d_same(lum, ky)[..., 0]
            out = jnp.stack([gx, gy], axis=-1)
        else:
            gx = _conv2d_same(img, kx)
            gy = _conv2d_same(img, ky)
            out = jnp.concatenate([gx, gy], axis=-1)
        return out.reshape((N, -1))

    return LinearOp(name="sobel", in_shape=(H, W, C), out_dim=out_dim, apply_flat_fn=_apply_flat)
