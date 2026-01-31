# infodynamics_jax/infodynamics/likelihoods/linear_ops/laplacian.py
"""Laplacian linear operator."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .stack import LinearOp


def _laplacian_kernel(dtype, eight_connected: bool = False):
    """Return Laplacian kernel.

    Args:
        dtype: Data type for the kernel.
        eight_connected: If True, use 8-connected Laplacian (includes diagonals).
                        If False, use 4-connected Laplacian.
    """
    if eight_connected:
        # 8-connected Laplacian
        k = jnp.array([[1.0,  1.0, 1.0],
                       [1.0, -8.0, 1.0],
                       [1.0,  1.0, 1.0]], dtype=dtype)
    else:
        # 4-connected Laplacian
        k = jnp.array([[0.0,  1.0, 0.0],
                       [1.0, -4.0, 1.0],
                       [0.0,  1.0, 0.0]], dtype=dtype)
    return k


def _conv2d_same(x, k):
    """Apply 2D convolution with 'same' padding."""
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


def laplacian_op(
    H: int,
    W: int,
    C: int,
    use_luma: bool = True,
    eight_connected: bool = False,
) -> LinearOp:
    """Build a Laplacian operator mapping y -> Laplacian(y).

    Args:
        H: Image height.
        W: Image width.
        C: Number of channels.
        use_luma: If True, convert to grayscale before applying Laplacian.
        eight_connected: If True, use 8-connected Laplacian kernel.

    Returns:
        LinearOp that applies the Laplacian operator.
    """
    out_channels = 1 if use_luma else C
    out_dim = H * W * out_channels

    def _to_luma(img):
        img = img[..., :3]
        return (0.2989 * img[..., 0]
                + 0.5870 * img[..., 1]
                + 0.1140 * img[..., 2])

    def _apply_flat(y_flat: jnp.ndarray) -> jnp.ndarray:
        N = y_flat.shape[0]
        img = y_flat.reshape((N, H, W, C))
        k = _laplacian_kernel(img.dtype, eight_connected)
        if use_luma:
            lum = _to_luma(img)[..., None]
            lap = _conv2d_same(lum, k)
        else:
            lap = _conv2d_same(img, k)
        return lap.reshape((N, -1))

    return LinearOp(name="laplacian", in_shape=(H, W, C), out_dim=out_dim, apply_flat_fn=_apply_flat)
