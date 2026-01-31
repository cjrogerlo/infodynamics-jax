"""
Stage-frozen circulation cache objects.

These are intended to be built once per annealing stage / mutation block,
then reused inside inner steps (Regime-1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from .operator import Planes, apply_Cv
from .planes import stage_schedule

Array = jnp.ndarray


@dataclass(frozen=True)
class StageFrozenCirculation:
    planes: Planes
    omega: Array  # (K,)
    schedule_fn: Callable = stage_schedule

    def apply(self, v: Array, lam) -> Array:
        """Compute C_beta v in vector space."""
        scale = self.schedule_fn(lam)
        return jnp.asarray(scale) * apply_Cv(self.planes, self.omega, v)
