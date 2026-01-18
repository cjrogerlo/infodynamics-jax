# infodynamics_jax/gp/kernels/utils.py
import jax.numpy as jnp

def scaled_sqdist(X, Z, lengthscale):
    Xs = X / lengthscale
    Zs = Z / lengthscale
    x2 = jnp.sum(Xs * Xs, axis=1)[:, None]
    z2 = jnp.sum(Zs * Zs, axis=1)[None, :]
    return jnp.maximum(x2 + z2 - 2.0 * (Xs @ Zs.T), 0.0)
