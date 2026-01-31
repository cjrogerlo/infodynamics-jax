"""
Y-only per-image circulation operators.

This module defines a per-image antisymmetric operator C_lambda(Y) that:
  - depends only on Y-derived stats (no particles / no current X),
  - applies a fixed Q-space plane basis to each image independently.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import jax.numpy as jnp

Array = jnp.ndarray


def canonical_q_pairs(q_dim: int, k: int | None = None) -> Tuple[Tuple[int, int], ...]:
    """Canonical (i, j) plane index pairs in Q-space."""
    pairs = [(i, j) for i in range(int(q_dim)) for j in range(i + 1, int(q_dim))]
    if k is None:
        return tuple(pairs)
    return tuple(pairs[: int(k)])


def apply_Cv_per_image(v: Array, omega: Array, pairs: Sequence[Tuple[int, int]]) -> Array:
    """
    Apply per-image skew operator to v.

    Args:
        v: (N, Q) vector field (e.g., gradient) per image
        omega: (N, K) per-image coefficients
        pairs: list of K (i, j) plane index pairs in Q-space

    Returns:
        cv: (N, Q) array with (Cv)_n = C^{(n)} v_n
    """
    v = jnp.asarray(v)
    omega = jnp.asarray(omega)
    n, q_dim = v.shape
    k = min(len(pairs), omega.shape[1])
    if k == 0:
        return jnp.zeros_like(v)

    cv = jnp.zeros_like(v)
    for idx in range(k):
        i, j = pairs[idx]
        vi = v[:, i]
        vj = v[:, j]
        w = omega[:, idx]
        cv = cv.at[:, i].add(w * vj)
        cv = cv.at[:, j].add(-w * vi)
    return cv


def make_cg_operator_y_only(
    stats_per_image: Array,
    W: Array,
    schedule_fn: Callable,
    pairs: Sequence[Tuple[int, int]],
    gain: float = 1.0,
):
    """
    Build a Y-only C_lambda operator from precomputed stats.

    Args:
        stats_per_image: (N, m) Y-derived stats (fixed once per dataset/batch)
        W: (m, K) gating matrix
        schedule_fn: schedule(lam) -> scalar
        pairs: K plane index pairs in Q-space
        gain: optional scalar multiplier for omega

    Returns:
        cg_op: (_x, v, lam) -> C_lam v, where v is (N, Q)
    """
    stats_per_image = jnp.asarray(stats_per_image)
    W = jnp.asarray(W)
    omega_raw = jnp.tanh(stats_per_image @ W) * jnp.asarray(gain, dtype=W.dtype)

    def cg_op(_x, v, lam=1.0):
        omega = jnp.asarray(schedule_fn(lam)) * omega_raw
        return apply_Cv_per_image(v, omega, pairs)

    return cg_op

