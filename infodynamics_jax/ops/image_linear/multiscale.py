# infodynamics_jax/infodynamics/likelihoods/linear_ops/multiscale.py
"""Multi-scale operator utilities."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp

from .stack import LinearOp


def downsample_mean(img: jnp.ndarray, s: int) -> jnp.ndarray:
    """Mean-pool downsample by factor s. img: (N, H, W, C)."""
    if s == 1:
        return img
    N, H, W, C = img.shape
    if (H % s) != 0 or (W % s) != 0:
        raise ValueError("H and W must be divisible by scale for mean downsample")
    x = img.reshape(N, H // s, s, W // s, s, C).mean(axis=(2, 4))
    return x


def multiscale_op(
    op_factory: Callable[..., LinearOp],
    H: int,
    W: int,
    C: int,
    scales: Sequence[int],
    weights: Sequence[float],
    weight_map: Optional[jnp.ndarray] = None,
    **op_kwargs,
) -> LinearOp:
    """
    Build a multi-scale operator by applying op_factory at each scale
    and concatenating outputs.
    """
    if len(scales) == 0:
        raise ValueError("scales must be non-empty")
    if len(scales) != len(weights):
        raise ValueError("weights length must match scales length")

    w_map = None
    if weight_map is not None:
        w_map = jnp.asarray(weight_map)
        if w_map.ndim == 2:
            w_map = w_map[..., None]
        if w_map.ndim != 3 or w_map.shape[0] != H or w_map.shape[1] != W or w_map.shape[2] != 1:
            raise ValueError("weight_map must have shape (H, W) or (H, W, 1)")

    ops = []
    for s in scales:
        hs = H // s
        ws = W // s
        if hs <= 0 or ws <= 0:
            raise ValueError("Invalid scale for given H, W")
        ops.append(op_factory(hs, ws, C, **op_kwargs))

    out_dim = int(sum(op.out_dim for op in ops))

    def _apply_flat(y_flat: jnp.ndarray) -> jnp.ndarray:
        N = y_flat.shape[0]
        img = y_flat.reshape((N, H, W, C))
        outs = []
        for op, s, w in zip(ops, scales, weights):
            img_s = downsample_mean(img, int(s))
            y_s = img_s.reshape((N, -1))
            out = op.apply_flat(y_s)
            if w_map is not None:
                hs = H // int(s)
                ws = W // int(s)
                out_ch = op.out_dim // (hs * ws)
                out = out.reshape((N, hs, ws, out_ch))
                w_s = downsample_mean(w_map[None, ...], int(s))[0]
                out = out * w_s
                out = out.reshape((N, -1))
            outs.append(jnp.array(w, dtype=y_flat.dtype) * out)
        return jnp.concatenate(outs, axis=1)

    return LinearOp(name="multiscale", in_shape=(H, W, C), out_dim=out_dim, apply_flat_fn=_apply_flat)
