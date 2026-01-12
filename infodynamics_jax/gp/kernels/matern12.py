import jax.numpy as jnp
from .utils import scaled_sqdist
from .params import KernelParams

def matern12(X, Z, params: KernelParams):
    """
    Matérn ν=1/2:
        k(r) = σ^2 exp(-r)
    where r = ||(x - z)/ℓ||.
    """
    ell = params.lengthscale
    var = params.variance

    r2 = scaled_sqdist(X, Z, ell)  # (N,M)
    r = jnp.sqrt(r2 + 1e-12)
    return var * jnp.exp(-r)

matern12 = matern12
