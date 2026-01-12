# infodynamics_jax/kernels/periodic.py
import jax.numpy as jnp
from .params import KernelParams

def periodic(X, Z, params: KernelParams):
    ell = params.lengthscale
    p = params.period
    diff = jnp.pi * (X[:, None, :] - Z[None, :, :]) / p
    sin2 = jnp.sum(jnp.sin(diff) ** 2, axis=-1)
    return params.variance * jnp.exp(-2.0 * sin2 / (ell ** 2))
