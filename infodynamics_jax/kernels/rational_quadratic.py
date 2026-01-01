import jax.numpy as jnp
from .utils import scaled_sqdist

def rational_quadratic(params, X, Z):
    """
    params:
      lengthscale
      variance
      alpha
    """
    sq = scaled_sqdist(X, Z, params["lengthscale"])
    alpha = params["alpha"]
    return params["variance"] * (1.0 + sq / (2.0 * alpha)) ** (-alpha)