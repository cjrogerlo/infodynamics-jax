# infodynamics_jax/gp/kernels/matern32.py
import jax.numpy as jnp
from .params import KernelParams
from .utils import scaled_sqdist

def matern32(X, Z, params: KernelParams):
    ell = params.lengthscale
    var = params.variance
    r = jnp.sqrt(scaled_sqdist(X, Z, ell) + 1e-12)
    sqrt3 = jnp.sqrt(3.0)
    return var * (1.0 + sqrt3 * r) * jnp.exp(-sqrt3 * r)
