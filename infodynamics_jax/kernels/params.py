from dataclasses import dataclass, field
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass(frozen=True)
class KernelParams:
    lengthscale: jnp.ndarray
    variance: jnp.ndarray
    offset: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    degree: jnp.ndarray = field(default_factory=lambda: jnp.array(2.0))
    period: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))
    alpha: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))

    degree: jnp.ndarray = field(
        default_factory=lambda: jnp.array(2.0)
    )

    period: jnp.ndarray = field(
        default_factory=lambda: jnp.array(1.0)
    )

    alpha: jnp.ndarray = field(
        default_factory=lambda: jnp.array(1.0)
    )

