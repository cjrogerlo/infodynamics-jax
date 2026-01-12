# infodynamics_jax/kernels/polynomial.py
import jax.numpy as jnp
from .params import KernelParams

def polynomial(X, Z, params: KernelParams):
    return params.variance * (X @ Z.T + params.offset) ** params.degree