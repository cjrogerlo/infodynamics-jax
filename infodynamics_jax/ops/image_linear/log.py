# infodynamics_jax/infodynamics/likelihoods/linear_ops/log.py
"""LoG (Laplacian of Gaussian) linear operator."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .stack import LinearOp


def _make_log_kernel(size: int, sigma: float, dtype):
    ax = jnp.arange(-(size // 2), size // 2 + 1, dtype=dtype)
    xx, yy = jnp.meshgrid(ax, ax, indexing="ij")
    rr2 = xx * xx + yy * yy
    sigma2 = jnp.array(sigma, dtype=dtype) ** 2
    norm = (rr2 - 2.0 * sigma2) / (sigma2 * sigma2 + 1e-12)
    ker = norm * jnp.exp(-rr2 / (2.0 * sigma2 + 1e-12))
    ker = ker - jnp.mean(ker)
    return ker


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


def log_op(H: int, W: int, C: int, size: int = 5, sigma: float = 1.0, use_luma: bool = True) -> LinearOp:
    """Build a LoG operator mapping y -> LoG(y)."""
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
        k = _make_log_kernel(size, sigma, img.dtype)
        if use_luma:
            lum = _to_luma(img)[..., None]
            out = _conv2d_same(lum, k)[..., 0]
        else:
            out = _conv2d_same(img, k)
        return out.reshape((N, -1))

    return LinearOp(name="log", in_shape=(H, W, C), out_dim=out_dim, apply_flat_fn=_apply_flat)
