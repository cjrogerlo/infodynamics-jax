# infodynamics_jax/gp/kernels/arccosine.py
"""
ArcCosine Kernel (Order 0 and Order 1).

The ArcCosine kernel is derived from the infinite-width limit of a neural
network with ReLU (order 1) or step (order 0) activations.

Order 1 (ReLU):
    k(x, x') = (σ² / π) * ||x|| * ||x'|| * J(θ)
    where J(θ) = sin(θ) + (π - θ) * cos(θ)
    and θ = arccos(x · x' / (||x|| ||x'||))

This kernel is particularly useful for shape-sensitive models as it
encodes angular relationships between inputs.

Reference:
    Cho, Y., & Saul, L. K. (2009). Kernel methods for deep learning.
    In NIPS.
"""
import jax.numpy as jnp
from .params import KernelParams

EPS = 1e-9


def arc_cosine_order0(X, Z, params: KernelParams):
    """
    ArcCosine kernel (order 0, step activation).

    k(x, x') = σ² * (1 - θ/π)
    where θ = arccos(x · x' / (||x|| ||x'||))

    Parameters
    ----------
    X : jnp.ndarray
        Input array of shape (N, D).
    Z : jnp.ndarray
        Input array of shape (M, D).
    params : KernelParams
        Kernel parameters. Uses `variance` (σ²).

    Returns
    -------
    K : jnp.ndarray
        Kernel matrix of shape (N, M).
    """
    var = params.variance

    # Compute norms
    nX = jnp.sqrt(jnp.sum(X**2, axis=-1, keepdims=True) + EPS)  # (N, 1)
    nZ = jnp.sqrt(jnp.sum(Z**2, axis=-1, keepdims=True) + EPS)  # (M, 1)

    # Dot product
    dot = X @ Z.T  # (N, M)

    # Cosine of angle
    cos_theta = jnp.clip(dot / (nX @ nZ.T + EPS), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arccos(cos_theta)

    return var * (1.0 - theta / jnp.pi)


def arc_cosine_order1(X, Z, params: KernelParams):
    """
    ArcCosine kernel (order 1, ReLU activation).

    k(x, x') = (σ² / π) * ||x|| * ||x'|| * J(θ)
    where J(θ) = sin(θ) + (π - θ) * cos(θ)
    and θ = arccos(x · x' / (||x|| ||x'||))

    This kernel captures angular structure in the input space,
    making it useful for shape-sensitive image models.

    Parameters
    ----------
    X : jnp.ndarray
        Input array of shape (N, D).
    Z : jnp.ndarray
        Input array of shape (M, D).
    params : KernelParams
        Kernel parameters. Uses `variance` (σ²).

    Returns
    -------
    K : jnp.ndarray
        Kernel matrix of shape (N, M).
    """
    var = params.variance

    # Compute norms
    nX = jnp.sqrt(jnp.sum(X**2, axis=-1, keepdims=True) + EPS)  # (N, 1)
    nZ = jnp.sqrt(jnp.sum(Z**2, axis=-1, keepdims=True) + EPS)  # (M, 1)

    # Dot product
    dot = X @ Z.T  # (N, M)

    # Cosine of angle (clipped for numerical stability)
    cos_theta = jnp.clip(dot / (nX @ nZ.T + EPS), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arccos(cos_theta)

    # J(θ) = sin(θ) + (π - θ) * cos(θ)
    J_theta = jnp.sin(theta) + (jnp.pi - theta) * cos_theta

    # Kernel: (σ² / π) * ||x|| * ||x'|| * J(θ)
    return var * (1.0 / jnp.pi) * (nX @ nZ.T) * J_theta


# Backward-compatible alias
arccosine = arc_cosine_order1
