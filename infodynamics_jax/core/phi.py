# infodynamics_jax/core/phi.py
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from ..gp.kernels.params import KernelParams

@register_pytree_node_class
@dataclass(frozen=True)
class Phi:
    """
    Structural (slow) hyperparameters φ.

    kernel_params: KernelParams (pytree)
    Z: inducing locations (M,Q)
    likelihood_params: dict pytree (e.g. {"noise_var": ... , ...})

    This is the *only* object mutated by inference algorithms.
    All energies and inference methods must treat Phi as opaque.
    """
    kernel_params: KernelParams
    Z: jnp.ndarray
    likelihood_params: dict
    jitter: float = 1e-6  # Stored as float, but can be JAX array in pytree operations
    # Increased from 1e-8 to 1e-6 for better numerical stability

    # --- backward-compat aliases (do NOT use in new code) ---
    @property
    def theta(self):
        return self.kernel_params

    def tree_flatten(self):
        # Convert jitter to JAX array, keeping its current shape
        # This supports both scalar (shape ()) and batched (shape (n,)) cases
        # Check if jitter is already a JAX array or a traced value
        if hasattr(self.jitter, '__array__') or hasattr(self.jitter, 'shape'):
            # Already a JAX array or traced value, use as-is
            jitter_val = self.jitter
        else:
            # It's a Python float, convert to JAX array
            jitter_val = jnp.array(self.jitter, dtype=jnp.float32)
        children = (self.kernel_params, self.Z, self.likelihood_params, jitter_val)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        kernel_params, Z, likelihood_params, jitter_arr = children
        # Keep jitter as a JAX array to support vmap/jit
        # The dataclass will accept both Python floats and JAX arrays
        # When vmap-ing, jitter_arr will have shape (n_particles,) or ()
        return cls(kernel_params=kernel_params, Z=Z, likelihood_params=likelihood_params, jitter=jitter_arr)
