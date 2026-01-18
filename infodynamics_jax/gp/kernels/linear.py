# infodynamics_jax/gp/kernels/linear.py
import jax.numpy as jnp
from .params import KernelParams

def linear(X, Z, params: KernelParams):
    return params.variance * (X - params.offset) @ (Z - params.offset).T