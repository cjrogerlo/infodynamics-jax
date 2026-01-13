# infodynamics_jax/models/gp_utils.py
"""
GP utility functions for SMC inference.

Provides functions for FITC prediction, prior sampling, and other
GP-related utilities needed for particle inference.
"""
from __future__ import annotations

from typing import Tuple, Dict, Callable
import jax
import jax.numpy as jnp
from jax import random


def fitc_predict(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    Z: jnp.ndarray,
    phi: jnp.ndarray,
    X_test: jnp.ndarray,
    kernel_fn: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    FITC prediction at test points.
    
    Args:
        X_train: Training inputs [N, D]
        y_train: Training outputs [N,]
        Z: Inducing points [M, D]
        phi: Hyperparameters [log_lengthscale, log_outputscale, log_noise_var]
        X_test: Test inputs [N*, D]
        kernel_fn: Kernel function
    
    Returns:
        mean: Predictive mean [N*,]
        var: Predictive variance [N*,]
    """
    from ..gp.kernels.params import KernelParams
    
    # Unpack hyperparameters
    lengthscale = jnp.exp(phi[0])
    outputscale = jnp.exp(phi[1])
    noise_var = jnp.exp(phi[2])
    
    kparams = KernelParams(lengthscale=lengthscale, variance=outputscale)
    
    # Ensure y is flat
    y = jnp.atleast_1d(y_train.squeeze())
    M = Z.shape[0]
    
    # K_zz with jitter
    K_zz = kernel_fn(Z, Z, kparams) + 1e-6 * jnp.eye(M)
    L_z = jnp.linalg.cholesky(K_zz)
    
    # K_xz
    K_xz = kernel_fn(X_train, Z, kparams)
    V = jnp.linalg.solve(L_z, K_xz.T)  # [M, N]
    
    # Diagonal correction: d_i = k(x_i, x_i) - q_ii + σ²
    K_xx_diag = jnp.full(X_train.shape[0], outputscale)
    d = K_xx_diag - (V ** 2).sum(axis=0) + noise_var
    d = jnp.maximum(d, 1e-12)  # Numerical stability
    
    # Solve for α (posterior mean coefficients)
    Dinv_sqrt = 1.0 / jnp.sqrt(d)
    A = V * Dinv_sqrt[None, :]  # [M, N]
    L_b = jnp.linalg.cholesky(jnp.eye(M) + A @ A.T)
    c = jnp.linalg.solve(L_b, jnp.linalg.solve(L_b.T, A @ (y * Dinv_sqrt)))
    alpha = (y / d) - (A.T @ c) * Dinv_sqrt
    
    # Prediction
    K_xz_star = kernel_fn(X_test, Z, kparams)
    T = jnp.linalg.solve(L_z, K_xz_star.T)  # [M, N*]
    
    # Predictive mean
    mean = (T.T @ V) @ alpha
    
    # Predictive variance
    K_ss_diag = jnp.full(X_test.shape[0], outputscale)
    W = jnp.linalg.solve(L_b, T)
    var = K_ss_diag - (T ** 2).sum(axis=0) + (T * W).sum(axis=0) + noise_var
    var = jnp.maximum(var, 1e-12)  # Numerical stability
    
    return mean, var


def sample_gp_hyperparams_prior(
    key: jax.random.PRNGKey,
    n_samples: int,
    prior_config: Dict[str, Tuple[float, float]],
) -> jnp.ndarray:
    """
    Sample GP hyperparameters from a log-normal prior.
    
    Args:
        key: PRNG key
        n_samples: Number of samples
        prior_config: Dict with keys 'log_lengthscale', 'log_outputscale', 'log_noise'
                      Each value is (loc, scale) for the normal distribution
    
    Returns:
        Samples [n_samples, 3] in log-space
    """
    keys = random.split(key, 3)
    
    # Sample from normal (these are already in log-space)
    log_ell = prior_config['log_lengthscale'][0] + prior_config['log_lengthscale'][1] * random.normal(keys[0], (n_samples,))
    log_sf2 = prior_config['log_outputscale'][0] + prior_config['log_outputscale'][1] * random.normal(keys[1], (n_samples,))
    log_sn2 = prior_config['log_noise'][0] + prior_config['log_noise'][1] * random.normal(keys[2], (n_samples,))
    
    return jnp.stack([log_ell, log_sf2, log_sn2], axis=-1)


def compute_metrics(
    mean: jnp.ndarray,
    var: jnp.ndarray,
    y_true: jnp.ndarray,
) -> Tuple[float, float]:
    """
    Compute RMSE and NLPD metrics.
    
    Args:
        mean: Predictive mean
        var: Predictive variance
        y_true: True values
    
    Returns:
        rmse: Root mean squared error
        nlpd: Negative log predictive density
    """
    y = jnp.atleast_1d(y_true.squeeze())
    
    rmse = jnp.sqrt(((mean - y) ** 2).mean())
    
    v = jnp.maximum(var, 1e-9)
    nlpd = 0.5 * ((y - mean) ** 2 / v + jnp.log(2 * jnp.pi * v)).mean()
    
    return float(rmse), float(nlpd)
