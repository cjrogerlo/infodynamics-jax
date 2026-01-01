import jax.numpy as jnp

def linear(params, X, Z):
    """
    params:
      variance: scalar
      offset: scalar or (Q,)
    """
    offset = params.get("offset", 0.0)
    Xc = X - offset
    Zc = Z - offset
    return params["variance"] * (Xc @ Zc.T)