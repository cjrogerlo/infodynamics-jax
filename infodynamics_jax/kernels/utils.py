import jax.numpy as jnp

def _as_ard(x, Q):
    x = jnp.asarray(x)
    return x if x.ndim > 0 else jnp.ones((Q,), dtype=x.dtype) * x

def scaled_sqdist(X, Z, lengthscale):
    """
    X: (N,Q), Z: (M,Q)
    lengthscale: scalar or (Q,)
    return: (N,M)
    """
    Q = X.shape[-1]
    ell = _as_ard(lengthscale, Q)
    Xs = X / ell
    Zs = Z / ell
    x2 = jnp.sum(Xs * Xs, axis=-1, keepdims=True)
    z2 = jnp.sum(Zs * Zs, axis=-1, keepdims=True).T
    return jnp.maximum(x2 + z2 - 2.0 * (Xs @ Zs.T), 0.0))
