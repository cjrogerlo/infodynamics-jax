import jax.numpy as jnp
from .utils import scaled_sqdist

def matern52(params, X, Z):
    r = jnp.sqrt(scaled_sqdist(X, Z, params["lengthscale"]) + 1e-30)
    t = jnp.sqrt(5.0) * r
    return params["variance"] * (1.0 + t + t**2 / 3.0) * jnp.exp(-t)