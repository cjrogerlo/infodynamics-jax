# infodynamics_jax/gp/kernels/rbf.py
import jax.numpy as jnp
from .params import KernelParams
from .utils import scaled_sqdist

def rbf(X, Z, params: KernelParams):
    ell = params.lengthscale
    var = params.variance
    r2 = scaled_sqdist(X, Z, ell)
    return var * jnp.exp(-0.5 * r2)
