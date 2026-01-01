import jax.numpy as jnp

def white(params, X, Z):
    """
    White noise kernel (only non-zero on diagonal when X == Z)
    """
    var = params["variance"]
    eq = jnp.all(X[:, None, :] == Z[None, :, :], axis=-1)
    return var * eq.astype(X.dtype)