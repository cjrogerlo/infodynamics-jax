# infodynamics_jax/gp/kernels/utils.py
import jax.numpy as jnp

def scaled_sqdist(X, Z, lengthscale):
    Xs = X / lengthscale
    Zs = Z / lengthscale
    x2 = jnp.sum(Xs * Xs, axis=1)[:, None]
    z2 = jnp.sum(Zs * Zs, axis=1)[None, :]
    return jnp.maximum(x2 + z2 - 2.0 * (Xs @ Zs.T), 0.0)


def scaled_sqdist_ard(X, Z, lengthscale_vec):
    """
    ARD (Automatic Relevance Determination) squared distance.
    
    Each input dimension has its own lengthscale.
    
    Args:
        X: (N, D) input array
        Z: (M, D) inducing/test array
        lengthscale_vec: (D,) per-dimension lengthscales
    
    Returns:
        (N, M) squared distances with ARD scaling
    """
    Xs = X / lengthscale_vec
    Zs = Z / lengthscale_vec
    x2 = jnp.sum(Xs * Xs, axis=1)[:, None]
    z2 = jnp.sum(Zs * Zs, axis=1)[None, :]
    return jnp.maximum(x2 + z2 - 2.0 * (Xs @ Zs.T), 0.0)

