import jax.numpy as jnp
from .utils import scaled_sqdist

def rbf(params, X, Z):
    sq = scaled_sqdist(X, Z, params["lengthscale"])
    return params["variance"] * jnp.exp(-0.5 * sq)
