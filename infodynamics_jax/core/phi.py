# infodynamics_jax/core/phi.py
from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from ..kernels.params import KernelParams

@register_pytree_node_class
@dataclass(frozen=True)
class Phi:
    """
    Structural (slow) hyperparameters φ.

    kernel_params: KernelParams (pytree)
    Z: inducing locations (M,Q)
    likelihood_params: dict pytree (e.g. {"noise_var": ... , ...})
    """
    kernel_params: KernelParams
    Z: jnp.ndarray
    likelihood_params: dict
    jitter: float = 1e-8

    # --- backward-compat aliases (do NOT use in new code) ---
    @property
    def theta(self):
        return self.kernel_params

    def tree_flatten(self):
        children = (self.kernel_params, self.Z, self.likelihood_params, jnp.array(self.jitter))
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        kernel_params, Z, likelihood_params, jitter_arr = children
        return cls(kernel_params=kernel_params, Z=Z, likelihood_params=likelihood_params, jitter=float(jitter_arr))
