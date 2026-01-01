# infodynamics_jax/kernels/white.py
import jax.numpy as jnp
from .params import KernelParams

def white(X, Z, params: KernelParams):
    if X.shape[0] != Z.shape[0]:
        return jnp.zeros((X.shape[0], Z.shape[0]))
    return params.variance * jnp.eye(X.shape[0])
