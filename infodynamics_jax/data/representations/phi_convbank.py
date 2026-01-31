from __future__ import annotations
from typing import Dict, Tuple

import jax.numpy as jnp

from infodynamics_jax.ops.image_linear import sobel_op, laplacian_op, log_op, stack_ops


def phi_convbank(images: jnp.ndarray, use_luma: bool = True) -> Tuple[jnp.ndarray, Dict]:
    """Simple convolution bank representation (identity + Sobel + Laplacian + LoG)."""
    N, H, W, C = images.shape
    op = stack_ops(
        [
            sobel_op(H, W, C, use_luma=use_luma),
            laplacian_op(H, W, C, use_luma=use_luma),
            log_op(H, W, C, sigma=1.0, use_luma=use_luma),
        ],
        weights=[1.0, 1.0, 1.0],
        name="convbank",
    )
    Y = op.apply_flat(images.reshape(N, -1)).reshape(N, -1)
    return Y, {"kind": "convbank", "op": "sobel+laplacian+log"}
