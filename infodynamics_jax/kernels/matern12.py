# infodynamics_jax/kernels/matern12.py
import jax.numpy as jnp
from .params import KernelParams
from .utils import scaled_sqdist

def matern12(X, Z, params: KernelParams):
    """
    Matern 1/2 kernel (Exponential):
        k(r) = variance * exp(-r)
    """
    ell = params.lengthscale
    var = params.variance

    # r = ||x - z|| / ell
    r = jnp.sqrt(scaled_sqdist(X, Z, ell) + 1e-12)
    return var * jnp.exp(-r)
