# infodynamics_jax/kernels/matern52.py
import jax.numpy as jnp
from .params import KernelParams
from .utils import scaled_sqdist

def matern52(X, Z, params: KernelParams):
    """
    Matern 5/2 kernel.
    """
    ell = params.lengthscale
    var = params.variance

    r2 = scaled_sqdist(X, Z, ell)
    r = jnp.sqrt(r2 + 1e-12)

    sqrt5 = jnp.sqrt(5.0)
    return var * (1.0 + sqrt5 * r + (5.0 / 3.0) * r2) * jnp.exp(-sqrt5 * r)
