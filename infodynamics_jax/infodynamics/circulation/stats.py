"""
Data statistics for Y-conditioned circulation (images, etc).

Keep this minimal: stats can be computed offline and passed in.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

Array = jnp.ndarray


@dataclass(frozen=True)
class PrecomputedStats:
    """
    stats_per_item: (B, m) or pooled (m,)
    """

    stats_per_item: Array

    def pooled(self, mode: str = "mean") -> Array:
        x = self.stats_per_item
        if x.ndim == 1:
            return x
        if mode == "mean":
            return jnp.mean(x, axis=0)
        if mode == "median":
            return jnp.median(x, axis=0)
        raise ValueError(f"Unknown pooling mode: {mode}")

