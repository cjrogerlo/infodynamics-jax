from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass
class Phi:
    """
    Structural (slow) hyperparameters φ.

    kernel_params:
        Hyperparameters of the kernel.
    Z:
        Inducing locations.
    likelihood_params:
        Hyperparameters of the likelihood.
    """
    kernel_params: dict
    Z: jnp.ndarray
    likelihood_params: dict

    def tree_flatten(self):
        children = (
            self.kernel_params,
            self.Z,
            self.likelihood_params,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kernel_params, Z, likelihood_params = children
        return cls(
            kernel_params=kernel_params,
            Z=Z,
            likelihood_params=likelihood_params,
        )
