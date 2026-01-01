import jax.numpy as jnp
from .utils import scaled_sqdist

def matern12(params, X, Z):
    r = jnp.sqrt(scaled_sqdist(X, Z, params["lengthscale"]) + 1e-30)
    return params["variance"] * jnp.exp(-r)