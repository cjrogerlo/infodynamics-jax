from __future__ import annotations
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass(frozen=True)
class KernelParams:
    """
    Unified kernel hyperparameters as a pytree.

    Unused fields are allowed; kernels just ignore what they don't need.
    """

    # Core
    lengthscale: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))
    variance: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))

    # Matern / RQ / Periodic
    nu: jnp.ndarray = field(default_factory=lambda: jnp.array(1.5))
    alpha: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))
    period: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))

    # Linear / Polynomial
    offset: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    degree: jnp.ndarray = field(default_factory=lambda: jnp.array(2.0))

    # ---- pytree protocol ----
    def tree_flatten(self):
        children = (
            self.lengthscale,
            self.variance,
            self.nu,
            self.alpha,
            self.period,
            self.offset,
            self.degree,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            lengthscale=children[0],
            variance=children[1],
            nu=children[2],
            alpha=children[3],
            period=children[4],
            offset=children[5],
            degree=children[6],
        )
