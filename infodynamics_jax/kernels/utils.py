# infodynamics_jax/kernels/utils.py
import jax.numpy as jnp

def scaled_sqdist(X: jnp.ndarray, Z: jnp.ndarray, lengthscale: jnp.ndarray) -> jnp.ndarray:
    """
    Compute squared distance ||(X/ell) - (Z/ell)||^2.

    X: (N,Q)
    Z: (M,Q)
    lengthscale: scalar or (Q,)
    returns: (N,M)
    """
    ell = jnp.asarray(lengthscale)
    Xs = X / ell
    Zs = Z / ell
    x2 = jnp.sum(Xs * Xs, axis=1)[:, None]      # (N,1)
    z2 = jnp.sum(Zs * Zs, axis=1)[None, :]      # (1,M)
    sq = x2 + z2 - 2.0 * (Xs @ Zs.T)
    return jnp.maximum(sq, 0.0)
