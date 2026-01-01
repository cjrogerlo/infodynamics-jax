import jax.numpy as jnp

def periodic(params, X, Z):
    """
    params:
      lengthscale
      variance
      period
    """
    period = params["period"]
    ell = params["lengthscale"]

    X = X / ell
    Z = Z / ell

    diff = X[:, None, :] - Z[None, :, :]
    sin2 = jnp.sum(jnp.sin(jnp.pi * diff / period) ** 2, axis=-1)

    return params["variance"] * jnp.exp(-2.0 * sin2)