# infodynamics_jax/kernels/params.py
from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass(frozen=True)
class KernelParams:
    """
    Unified kernel hyperparameters (pytree, JIT-friendly).

    Only a subset is used by each kernel.
    """
    lengthscale: jnp.ndarray        # (Q,) or scalar
    variance: jnp.ndarray           # scalar

    # optional / kernel-specific (keep as arrays for JIT)
    offset: jnp.ndarray = jnp.array(0.0)      # linear/poly
    degree: jnp.ndarray = jnp.array(2.0)      # polynomial
    period: jnp.ndarray = jnp.array(1.0)      # periodic
    alpha: jnp.ndarray = jnp.array(1.0)       # rational quadratic

    def tree_flatten(self):
        children = (
            self.lengthscale, self.variance,
            self.offset, self.degree, self.period, self.alpha,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def ensure_ard(lengthscale, Q: int):
    """Broadcast scalar lengthscale to (Q,) if needed."""
    ls = jnp.asarray(lengthscale)
    if ls.ndim == 0:
        return jnp.ones((Q,), dtype=ls.dtype) * ls
    return ls
