# infodynamics_jax/infodynamics/likelihoods/linear_ops/stack.py
"""
Linear operator utilities for output-space feature construction.

This module provides:
- LinearOp: an implicit linear operator on flattened image outputs
- identity_op: identity operator
- stack_ops: concatenate multiple operators into one A
- materialize_A / compute_S: explicit A and S = A A^T (for small output dims)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from infodynamics_jax.gp.utils import safe_cholesky


@dataclass
class LinearOp:
    """
    Implicit linear operator A acting on flattened outputs y.

    apply_flat_fn: (N, D) -> (N, M)
    in_shape: (H, W, C)
    out_dim: M
    """
    name: str
    in_shape: Tuple[int, int, int]
    out_dim: int
    apply_flat_fn: Callable[[jnp.ndarray], jnp.ndarray]

    def apply_flat(self, y_flat: jnp.ndarray) -> jnp.ndarray:
        return self.apply_flat_fn(y_flat)


def identity_op(H: int, W: int, C: int) -> LinearOp:
    """Identity operator A = I on flattened outputs."""
    D = H * W * C

    def _apply_flat(y_flat: jnp.ndarray) -> jnp.ndarray:
        return y_flat

    return LinearOp(name="identity", in_shape=(H, W, C), out_dim=D, apply_flat_fn=_apply_flat)


def stack_ops(ops: Sequence[LinearOp], weights: Optional[Sequence[float]] = None, name: str = "stack") -> LinearOp:
    """
    Stack multiple operators vertically: A = [w1 A1; w2 A2; ...].
    Outputs are concatenated along feature dimension.
    """
    if len(ops) == 0:
        raise ValueError("stack_ops requires at least one op")

    H, W, C = ops[0].in_shape
    for op in ops:
        if op.in_shape != (H, W, C):
            raise ValueError("All ops must share the same input shape")

    if weights is None:
        weights = [1.0] * len(ops)
    if len(weights) != len(ops):
        raise ValueError("weights length must match ops length")

    out_dim = int(sum(op.out_dim for op in ops))

    def _apply_flat(y_flat: jnp.ndarray) -> jnp.ndarray:
        outs = []
        for op, w in zip(ops, weights):
            outs.append(jnp.array(w, dtype=y_flat.dtype) * op.apply_flat(y_flat))
        return jnp.concatenate(outs, axis=1)

    return LinearOp(name=name, in_shape=(H, W, C), out_dim=out_dim, apply_flat_fn=_apply_flat)



def weighted_op(op: LinearOp, weight_map: jnp.ndarray, name: str = "weighted") -> LinearOp:
    """Apply a fixed per-pixel weight map to an operator's output."""
    H, W, _ = op.in_shape
    if op.out_dim % (H * W) != 0:
        raise ValueError("weighted_op expects per-pixel outputs")
    out_ch = op.out_dim // (H * W)

    w = jnp.asarray(weight_map)
    if w.ndim == 2:
        w = w[..., None]
    if w.ndim != 3 or w.shape[0] != H or w.shape[1] != W or w.shape[2] != 1:
        raise ValueError("weight_map must have shape (H, W) or (H, W, 1)")

    def _apply_flat(y_flat: jnp.ndarray) -> jnp.ndarray:
        out = op.apply_flat(y_flat)
        out = out.reshape((y_flat.shape[0], H, W, out_ch))
        out = out * w
        return out.reshape((y_flat.shape[0], -1))

    return LinearOp(name=f"{name}:{op.name}", in_shape=op.in_shape, out_dim=op.out_dim, apply_flat_fn=_apply_flat)


def materialize_A(op: LinearOp, dtype=jnp.float32) -> jnp.ndarray:
    """
    Explicitly materialize A by applying op to the canonical basis.
    This is only practical for small output dimensions (e.g., patch sizes).
    Returns A with shape (M, D).
    """
    H, W, C = op.in_shape
    D = H * W * C
    eye = jnp.eye(D, dtype=dtype)
    # Treat each basis vector as one sample (N = D)
    out = op.apply_flat(eye)  # (D, M)
    A = out.T  # (M, D)
    return A


def compute_S(op: LinearOp, dtype=jnp.float32, jitter: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute S = A A^T and its Cholesky factor.
    Returns (S, Ls).
    """
    A = materialize_A(op, dtype=dtype)
    S = A @ A.T
    Ls = safe_cholesky(S + jnp.array(jitter, dtype=dtype) * jnp.eye(S.shape[0], dtype=dtype))
    return S, Ls



def compute_S_for_ops(
    ops: Sequence[LinearOp],
    weights: Optional[Sequence[float]] = None,
    mode: str = "full",
    dtype=jnp.float32,
    jitter: float = 1e-6,
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    Compute S and its Cholesky for a list of ops, with selectable mode.

    mode: "full" | "blockdiag" | "none"
    """
    if mode == "none":
        return None, None
    if mode == "blockdiag":
        return compute_S_blockdiag(ops, weights=weights, dtype=dtype, jitter=jitter)
    op = stack_ops(ops, weights=weights)
    return compute_S(op, dtype=dtype, jitter=jitter)


def compute_S_blockdiag(ops: Sequence[LinearOp], weights: Optional[Sequence[float]] = None,
                        dtype=jnp.float32, jitter: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute S for stacked operators using block-diagonal structure.
    S = diag(w1^2 A1 A1^T, w2^2 A2 A2^T, ...)
    Returns (S, Ls).
    """
    if weights is None:
        weights = [1.0] * len(ops)
    if len(weights) != len(ops):
        raise ValueError("weights length must match ops length")

    blocks = []
    for op, w in zip(ops, weights):
        A = materialize_A(op, dtype=dtype)
        S_i = (jnp.array(w, dtype=dtype) ** 2) * (A @ A.T)
        blocks.append(S_i)

    S = jax.scipy.linalg.block_diag(*blocks) if len(blocks) > 1 else blocks[0]
    Ls = safe_cholesky(S + jnp.array(jitter, dtype=dtype) * jnp.eye(S.shape[0], dtype=dtype))
    return S, Ls
