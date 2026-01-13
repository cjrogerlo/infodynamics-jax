# infodynamics_jax/energy/vfe.py
"""
VFE (Variational Free Energy) energy for GP hyperparameter inference via SMC.

This module provides energy functions for sparse GP inference using the
FITC/VFE approximation (Titsias, 2009).
"""
from __future__ import annotations

from typing import Dict, Callable, Optional
import jax
import jax.numpy as jnp
from jax import jit

from .base import EnergyTerm


def titsias_vfe_energy(
    X: jnp.ndarray,
    y: jnp.ndarray,
    Z: jnp.ndarray,
    phi: jnp.ndarray,
    kernel_fn: Callable,
) -> float:
    """
    Compute negative VFE (Variational Free Energy) as energy term.
    
    This is the FITC/VFE approximation from Titsias (2009):
        F_VFE = log p(y|Z,φ) - (1/2σ²) Tr[K_ff - Q_ff]
    
    where Q_ff = K_fu K_uu^{-1} K_uf is the inducing representation.
    
    Args:
        X: Training inputs [N, D]
        y: Training outputs [N,] or [N, 1]
        Z: Inducing inputs [M, D]
        phi: Hyperparameters [lengthscale, outputscale, noise_variance]
        kernel_fn: Kernel function (X1, X2, params) -> K
    
    Returns:
        Negative VFE (energy = -log p(y|Z,φ) + trace correction)
    """
    from ..gp.kernels.params import KernelParams
    
    # Unpack hyperparameters (assuming log-space)
    lengthscale = jnp.exp(phi[0])
    outputscale = jnp.exp(phi[1])
    noise_var = jnp.exp(phi[2])
    
    # Ensure y is flat
    y = jnp.atleast_1d(y.squeeze())
    N = X.shape[0]
    M = Z.shape[0]
    
    # Create kernel params
    kparams = KernelParams(lengthscale=lengthscale, variance=outputscale)
    
    # K_uu: [M, M]
    K_uu = kernel_fn(Z, Z, kparams) + 1e-6 * jnp.eye(M)
    L_u = jnp.linalg.cholesky(K_uu)
    
    # K_fu: [N, M]
    K_fu = kernel_fn(X, Z, kparams)
    
    # Compute A = K_uu + σ^{-2} K_uf K_fu
    inv_noise = 1.0 / noise_var
    A = K_uu + inv_noise * (K_fu.T @ K_fu)
    L_A = jnp.linalg.cholesky(A)
    
    # Solve for α = A^{-1} K_uf y
    u_vec = K_fu.T @ y  # [M,]
    alpha = jnp.linalg.solve(L_A, jnp.linalg.solve(L_A.T, u_vec))
    
    # Quadratic term: y^T (σ^{-2}I - σ^{-4} K_fu A^{-1} K_uf) y
    quad = (y * (inv_noise * y - inv_noise**2 * (K_fu @ alpha))).sum()
    
    # Log determinant: |A| / |K_uu| * σ^{2N}
    logdet_A = 2 * jnp.log(jnp.diag(L_A)).sum()
    logdet_Kuu = 2 * jnp.log(jnp.diag(L_u)).sum()
    logdet = logdet_A - logdet_Kuu + N * jnp.log(noise_var)
    
    # Log likelihood term
    loglik = -0.5 * (quad + logdet + N * jnp.log(2 * jnp.pi))
    
    # Trace correction: Tr[K_ff - Q_ff] / (2σ²)
    # K_ff diagonal - for RBF this is just the variance
    K_ff_diag = jnp.full(N, outputscale)
    
    # Q_ff diagonal: diag(K_fu K_uu^{-1} K_uf)
    V = jnp.linalg.solve(L_u, K_fu.T)  # [M, N]
    Q_ff_diag = (V ** 2).sum(axis=0)  # [N,]
    
    trace_term = 0.5 * (K_ff_diag - Q_ff_diag).sum() / noise_var
    
    # VFE = log likelihood - trace correction
    vfe = loglik - trace_term
    
    # Return negative VFE as energy
    return -vfe


class VFEEnergy(EnergyTerm):
    """
    Energy term for GP hyperparameter inference using VFE approximation.
    
    This energy is suitable for use with particle methods (SMC, MCMC) to
    infer hyperparameters φ and inducing locations Z jointly.
    
    The energy is:
        E(φ, Z) = -log p(y|Z,φ) + trace_correction
    
    where the VFE provides a lower bound on the marginal likelihood.
    """
    
    def __init__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        kernel_fn: Callable,
    ):
        """
        Initialize VFE energy.
        
        Args:
            X: Training inputs [N, D]
            y: Training outputs [N,] or [N, 1]
            kernel_fn: Kernel function with signature (X1, X2, params) -> K
        """
        self.X = X
        self.y = jnp.atleast_1d(y.squeeze())
        self.kernel_fn = kernel_fn
    
    @jit
    def __call__(self, params: Dict[str, jnp.ndarray]) -> float:
        """
        Compute VFE energy for given parameters.
        
        Args:
            params: Dictionary with keys:
                - 'phi': Hyperparameters [lengthscale, outputscale, noise_var]
                - 'Z': Inducing locations [M, D]
        
        Returns:
            Energy value (negative VFE)
        """
        phi = params['phi']
        Z = params['Z']
        
        return titsias_vfe_energy(
            self.X,
            self.y,
            Z,
            phi,
            self.kernel_fn
        )
