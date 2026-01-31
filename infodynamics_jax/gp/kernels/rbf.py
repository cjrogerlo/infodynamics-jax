# infodynamics_jax/gp/kernels/rbf.py
import jax.numpy as jnp
from .params import KernelParams
from .utils import scaled_sqdist, scaled_sqdist_ard

def rbf(X, Z, params: KernelParams):
    ell = params.lengthscale
    var = params.variance
    r2 = scaled_sqdist(X, Z, ell)
    return var * jnp.exp(-0.5 * r2)


def rbf_ard(X, Z, lengthscale_vec, variance):
    """
    ARD RBF kernel with per-dimension lengthscales.
    
    K(x, z) = variance * exp(-0.5 * sum_d ((x_d - z_d) / ell_d)^2)
    
    Args:
        X: (N, D) input array
        Z: (M, D) inducing/test array
        lengthscale_vec: (D,) per-dimension lengthscales
        variance: scalar signal variance
    
    Returns:
        (N, M) kernel matrix
    """
    r2 = scaled_sqdist_ard(X, Z, lengthscale_vec)
    return variance * jnp.exp(-0.5 * r2)

