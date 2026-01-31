"""
Coefficient parameterisations: omega(stats(Y)).

This module does not apply schedules; scheduling is handled at the stage level.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

Array = jnp.ndarray


@dataclass(frozen=True)
class ConstantOmega:
    omega0: Array  # (K,)

    def __call__(self, stats: Array | None = None) -> Array:
        return self.omega0


@dataclass(frozen=True)
class LinearTanhGate:
    """
    omega = gain * tanh(stats @ W)

    stats: (m,) or (B, m)
    W: (m, K)
    """

    W: Array
    gain: float = 1.0

    def __call__(self, stats: Array) -> Array:
        stats = jnp.asarray(stats)
        return jnp.asarray(self.gain, dtype=stats.dtype) * jnp.tanh(stats @ self.W)
