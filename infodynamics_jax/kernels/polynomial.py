import jax.numpy as jnp

def polynomial(params, X, Z):
    """
    params:
      variance
      offset
      degree (int)
    """
    offset = params.get("offset", 0.0)
    degree = params["degree"]
    return params["variance"] * (X @ Z.T + offset) ** degree