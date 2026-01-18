# infodynamics_jax/gp/kernels/matern52.py
import jax.numpy as jnp
from .utils import scaled_sqdist
from .params import KernelParams

def matern52(X, Z, params: KernelParams):
    """
    Matérn ν=5/2:
        k(r) = σ^2 (1 + √5 r + 5/3 r^2) exp(-√5 r)
    where r = ||(x - z)/ℓ||.
    """
    ell = params.lengthscale
    var = params.variance

    r2 = scaled_sqdist(X, Z, ell)
    r = jnp.sqrt(r2 + 1e-12)
    s5r = jnp.sqrt(5.0) * r
    poly = (1.0 + s5r + (5.0 / 3.0) * (r2))
    return var * poly * jnp.exp(-s5r)

matern52 = matern52
