# infodynamics_jax/kernels/rational_quadratic.py
import jax.numpy as jnp
from .params import KernelParams
from .utils import scaled_sqdist

def rational_quadratic(X, Z, params: KernelParams):
    ell = params.lengthscale
    var = params.variance
    alpha = params.alpha
    r2 = scaled_sqdist(X, Z, ell)
    return var * (1.0 + r2 / (2.0 * alpha)) ** (-alpha)
