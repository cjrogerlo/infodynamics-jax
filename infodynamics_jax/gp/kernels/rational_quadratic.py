# infodynamics_jax/gp/kernels/rational_quadratic.py
import jax.numpy as jnp
from .params import KernelParams
from .utils import scaled_sqdist, scaled_sqdist_ard

def rational_quadratic(X, Z, params: KernelParams):
    ell = params.lengthscale
    var = params.variance
    alpha = params.alpha
    r2 = scaled_sqdist(X, Z, ell)
    return var * (1.0 + r2 / (2.0 * alpha)) ** (-alpha)


def rational_quadratic_ard(X, Z, lengthscale_vec, alpha, variance):
    """
    ARD Rational Quadratic kernel with per-dimension lengthscales.
    
    RQ is a scale mixture of RBFs, useful for multi-scale features.
    K(x, z) = variance * (1 + r2 / (2 * alpha))^{-alpha}
    
    Args:
        X: (N, D) input array
        Z: (M, D) inducing/test array
        lengthscale_vec: (D,) per-dimension lengthscales
        alpha: RQ shape parameter (smaller = heavier tails)
        variance: scalar signal variance
    
    Returns:
        (N, M) kernel matrix
    """
    r2 = scaled_sqdist_ard(X, Z, lengthscale_vec)
    return variance * (1.0 + r2 / (2.0 * alpha)) ** (-alpha)

