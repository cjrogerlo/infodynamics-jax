# infodynamics_jax/inference/optimisation/vfe.py
"""
Type-II variational objective (VFE surrogate) for optimisation-based inference.

This module provides VFE (Variational Free Energy) objective functions for type-II
maximum likelihood and variational inference methods. VFE and KL terms are
specific to these inference paradigms and do not belong in the gp/ or energy/
layers.

IMPORTANT: This is NOT a model energy. This is an optimisation surrogate
used only in type-II / VGA / SVGP inference contexts. For Bayesian inference
(SMC/MCMC), use energy/inertial.py instead.

The VFE objective uses SparsifiedKernel from gp.sparsify and can call ansatz
for non-conjugate cases.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ...gp.sparsify import SparsifiedKernel


def vfe_objective(phi, X, Y, *, kernel_fn: Callable, residual: str = "fitc") -> jnp.ndarray:
    """
    Type-II variational objective (collapsed ELBO surrogate) for Gaussian likelihood.
    
    This computes the VFE objective for type-II ML / variational inference:
        L(phi) = E_{q*(f | phi)}[ -log p(y | f, phi) ]
    
    where q*(f | phi) is the exact posterior under the sparsified kernel S_ff(φ).
    
    According to the sparsified kernel framework, for Gaussian likelihood, 
    the exact posterior is:
        μ(φ) = S_ff(φ) (S_ff(φ) + σ²I)^{-1} y
        Σ(φ) = S_ff(φ) - S_ff(φ) (S_ff(φ) + σ²I)^{-1} S_ff(φ)
    
    Args:
        phi: Structural parameters (must have .Z, .kernel_params, .likelihood_params)
        X: Input locations (N, Q)
        Y: Observations (N,) or (N, D)
        kernel_fn: Kernel function kernel_fn(X1, X2, kernel_params) -> (N1, N2)
        residual: Residual type ("fitc", "sor", "dtc")
    
    Returns:
        Scalar objective value (to be minimized in type-II optimisation).
    
    Notes:
        - This is an OPTIMISATION OBJECTIVE, not a model energy.
        - Uses SparsifiedKernel to get S_ff = Q + R
        - Computes exact posterior under S_ff (u already marginalised)
        - For non-conjugate cases, can call ansatz from gp.ansatz
        - For Bayesian inference (SMC/MCMC), use energy/inertial.py instead
    """
    Y = Y[:, None] if Y.ndim == 1 else Y  # (N, D)
    N, D = Y.shape
    
    # Extract parameters
    Z = phi.Z  # (M, Q)
    noise_var = phi.likelihood_params["noise_var"]  # scalar or (D,)
    noise_var = jnp.asarray(noise_var)
    if noise_var.ndim == 0:
        noise_var = noise_var[None]  # (1,)
    noise_var = jnp.broadcast_to(noise_var, (D,))  # (D,)
    
    # Get sparsified kernel S_ff = Q + R (u already marginalised)
    sk = SparsifiedKernel(kernel_fn=kernel_fn, residual=residual)
    S_ff = sk.S_ff(phi.kernel_params, X, Z, jitter=phi.jitter)  # (N, N)
    
    # For each output dimension, compute exact posterior and objective
    def objective_for_output_d(y_d, noise_var_d):
        """
        Compute objective for single output dimension using exact posterior under S_ff.
        
        Exact posterior under sparsified kernel:
            μ(φ) = S_ff(φ) (S_ff(φ) + σ²I)^{-1} y
            Σ(φ) = S_ff(φ) - S_ff(φ) (S_ff(φ) + σ²I)^{-1} S_ff(φ)
        
        Then compute: E_{q*(f)}[-log N(y; f, σ²)]
        """
        y_d = y_d[:, None]  # (N, 1)
        
        # S_ff + σ²I
        S_noise = S_ff + noise_var_d * jnp.eye(N, dtype=S_ff.dtype)
        
        # Cholesky of S_noise
        L_S = jnp.linalg.cholesky(S_noise)
        
        # μ = S_ff (S_ff + σ²I)^{-1} y
        #   = S_ff @ solve(S_noise, y)
        mu = S_ff @ jax.scipy.linalg.cho_solve((L_S, True), y_d)  # (N, 1)
        mu = mu[:, 0]  # (N,)
        
        # Σ = S_ff - S_ff (S_ff + σ²I)^{-1} S_ff
        # Diagonal: diag(Σ) = diag(S_ff) - diag(S_ff @ solve(S_noise, S_ff))
        S_ff_solve_S_noise = jax.scipy.linalg.cho_solve((L_S, True), S_ff)  # (N, N)
        var_f = jnp.diag(S_ff) - jnp.sum(S_ff * S_ff_solve_S_noise, axis=1)  # (N,)
        var_f = jnp.clip(var_f, a_min=0.0)
        
        # Compute E_{q*(f_i)}[-log N(y_i; f_i, sigma^2)]
        # = 0.5 * (log(2*pi*sigma^2) + ((y_i - mu_i)^2 + var_i) / sigma^2)
        objective = 0.5 * (
            jnp.log(2.0 * jnp.pi * noise_var_d)
            + ((y_d[:, 0] - mu) ** 2 + var_f) / noise_var_d
        )
        return jnp.sum(objective)
    
    # Sum over output dimensions
    total_objective = jnp.sum(
        jax.vmap(objective_for_output_d, in_axes=(1, 0))(Y, noise_var)
    )
    
    return total_objective


def make_vfe_objective(kernel_fn: Callable, residual: str = "fitc") -> Callable:
    """
    Factory function to create a VFE objective function bound to a kernel.
    
    This returns a callable with signature (phi, X, Y) -> scalar,
    suitable for use in type-II ML / variational inference methods (e.g. VGA).
    
    Args:
        kernel_fn: Kernel function kernel_fn(X1, X2, kernel_params) -> (N1, N2)
        residual: Residual type ("fitc", "sor", "dtc")
    
    Returns:
        Function (phi, X, Y) -> scalar objective (to be minimized)
    
    Notes:
        - This is an OPTIMISATION OBJECTIVE, not a model energy.
        - For Bayesian inference (SMC/MCMC), use energy/inertial.py instead.
    """
    return partial(vfe_objective, kernel_fn=kernel_fn, residual=residual)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "vfe_objective",
    "make_vfe_objective",
]
