"""
Circulation operators (apply Cv without materialising C).

All functions in this module are purely linear-algebraic:
- no energies
- no gradients (we may act on a vector that happens to be a gradient)
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

Array = jnp.ndarray


@dataclass(frozen=True)
class Planes:
    """
    A stack of K oriented 2D planes in R^d.

    q1, q2 are (d, K) with (approximately) orthonormal columns.
    The induced antisymmetric operator is:
      C = Σ_k ω_k (q1_k q2_k^T - q2_k q1_k^T).
    """

    q1: Array  # (d, K)
    q2: Array  # (d, K)

    @property
    def k(self) -> int:
        return int(self.q1.shape[1])


def apply_Cv(planes: Planes, omega: Array, v: Array) -> Array:
    """
    Compute Cv where C = Σ_k ω_k (q1_k q2_k^T - q2_k q1_k^T).

    Args:
        planes: Planes(q1, q2) with q1/q2 shape (d, K)
        omega: (K,) coefficients
        v: (d,) vector
    """
    omega = jnp.asarray(omega)
    k = min(int(omega.shape[0]), planes.k)
    if k == 0:
        return jnp.zeros_like(v)

    q1 = planes.q1[:, :k]
    q2 = planes.q2[:, :k]
    a = q2.T @ v  # (k,)
    b = q1.T @ v  # (k,)
    wa = omega[:k] * a
    wb = omega[:k] * b
    return q1 @ wa - q2 @ wb

