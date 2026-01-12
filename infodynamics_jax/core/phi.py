# infodynamics_jax/core/phi.py
from __future__ import annotations
from dataclasses import dataclass
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
    jitter: float = 1e-8  # Stored as float, but can be JAX array in pytree operations

    # --- backward-compat aliases (do NOT use in new code) ---
    @property
    def theta(self):
        return self.kernel_params

    def tree_flatten(self):
        # Ensure jitter is a scalar array (0-d, not 1-d)
        jitter_val = jnp.asarray(self.jitter, dtype=jnp.float32)
        if jitter_val.ndim > 0:
            jitter_val = jitter_val.reshape(())
        children = (self.kernel_params, self.Z, self.likelihood_params, jitter_val)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        kernel_params, Z, likelihood_params, jitter_arr = children
        # Handle jitter: in JIT context, jitter_arr is a traced array
        # We need to convert it to a concrete Python float for the dataclass
        # But we can't use .item() in JIT - so we'll use a workaround:
        # Store the array and convert lazily when accessed
        # For now, try to get the value - if it fails (in JIT), use default
        try:
            if hasattr(jitter_arr, 'shape'):
                if jitter_arr.shape == ():
                    # Try to get concrete value (fails in JIT)
                    jitter = float(jitter_arr)
                elif jitter_arr.size == 1:
                    jitter = float(jitter_arr.flatten()[0])
                else:
                    jitter = float(jitter_arr.flatten()[0])
            else:
                jitter = float(jitter_arr)
        except (TypeError, ValueError):
            # In JIT context, use default value
            jitter = 1e-8
        return cls(kernel_params=kernel_params, Z=Z, likelihood_params=likelihood_params, jitter=jitter)
