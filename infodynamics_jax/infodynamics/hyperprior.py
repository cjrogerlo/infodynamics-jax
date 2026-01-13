# infodynamics_jax/infodynamics/hyperprior.py
"""
Hyperprior utilities for all inference methods.

Hyperpriors are priors on structural hyperparameters φ:
  - kernel_params: kernel hyperparameters
  - Z: inducing point locations
  - likelihood_params: likelihood hyperparameters

NOTE: Hyperpriors are NOT energy terms. They are regularization terms
that can be added to the objective in all inference methods:
  - MAP2: optimize E(phi) + hyperprior(phi)
  - HMC/NUTS/MALA: sample p(phi|y) ∝ exp(-E(phi) - hyperprior(phi))
  - Annealed SMC: use E(phi) + hyperprior(phi) as target
  - IBIS: use E(phi) + hyperprior(phi) as target

Different kernels/likelihoods use different parameter subsets, so these
functions allow selective application of priors to specific fields/keys.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Callable

import jax.numpy as jnp

from ..core.phi import Phi


def kernel_l2_hyperprior(phi: Phi, fields: Optional[list[str]] = None, lam: float = 1.0) -> jnp.ndarray:
    """
    L2 hyperprior on kernel parameters.
    
    Args:
        phi: Structural hyperparameters
        fields: List of field names to penalize (e.g., ["lengthscale", "variance"])
                If None, penalizes default fields
        lam: Penalty strength
    
    Returns:
        Scalar penalty value (-log p(phi) contribution)
    """
    if lam <= 0.0:
        return jnp.array(0.0)
    
    if fields is None:
        fields = ["lengthscale", "variance"]
    
    total = 0.0
    for field in fields:
        if hasattr(phi.kernel_params, field):
            val = getattr(phi.kernel_params, field)
            total += jnp.sum(val ** 2)
    
    return 0.5 * lam * total


def kernel_log_l2_hyperprior(phi: Phi, fields: Optional[list[str]] = None, 
                              lam: float = 1.0, mu: Optional[Dict[str, float]] = None) -> jnp.ndarray:
    """
    L2 hyperprior on log kernel parameters (log-normal prior).
    
    Useful for positive parameters like lengthscale, variance.
    
    Args:
        phi: Structural hyperparameters
        fields: List of field names to penalize
        lam: Penalty strength
        mu: Optional dict of mean values for each field (default: 0)
    
    Returns:
        Scalar penalty value
    """
    if lam <= 0.0:
        return jnp.array(0.0)
    
    if fields is None:
        fields = ["lengthscale", "variance"]
    
    if mu is None:
        mu = {}
    
    total = 0.0
    for field in fields:
        if hasattr(phi.kernel_params, field):
            val = getattr(phi.kernel_params, field)
            log_val = jnp.log(jnp.clip(val, a_min=1e-8))
            mu_val = mu.get(field, 0.0)
            total += jnp.sum((log_val - mu_val) ** 2)
    
    return 0.5 * lam * total


def z_l2_hyperprior(phi: Phi, lam: float = 1.0) -> jnp.ndarray:
    """
    L2 hyperprior on inducing point locations.
    
    Args:
        phi: Structural hyperparameters
        lam: Penalty strength
    
    Returns:
        Scalar penalty value
    """
    if lam <= 0.0:
        return jnp.array(0.0)
    return 0.5 * lam * jnp.sum(phi.Z ** 2)


def likelihood_l2_hyperprior(phi: Phi, keys: Optional[list[str]] = None, 
                             lam: float = 1.0) -> jnp.ndarray:
    """
    L2 hyperprior on likelihood parameters.
    
    Args:
        phi: Structural hyperparameters
        keys: List of keys to penalize (e.g., ["noise_var"])
               If None, penalizes default keys
        lam: Penalty strength
    
    Returns:
        Scalar penalty value
    """
    if lam <= 0.0:
        return jnp.array(0.0)
    
    if keys is None:
        keys = ["noise_var"]
    
    total = 0.0
    for key in keys:
        if key in phi.likelihood_params:
            val = phi.likelihood_params[key]
            if isinstance(val, jnp.ndarray):
                total += jnp.sum(val ** 2)
            else:
                total += float(val) ** 2
    
    return 0.5 * lam * total


def likelihood_log_l2_hyperprior(phi: Phi, keys: Optional[list[str]] = None,
                                 lam: float = 1.0, mu: Optional[Dict[str, float]] = None) -> jnp.ndarray:
    """
    L2 hyperprior on log likelihood parameters (log-normal prior).
    
    Useful for positive parameters like noise_var.
    
    Args:
        phi: Structural hyperparameters
        keys: List of keys to penalize
        lam: Penalty strength
        mu: Optional dict of mean values for each key (default: 0)
    
    Returns:
        Scalar penalty value
    """
    if lam <= 0.0:
        return jnp.array(0.0)
    
    if keys is None:
        keys = ["noise_var"]
    
    if mu is None:
        mu = {}
    
    total = 0.0
    for key in keys:
        if key in phi.likelihood_params:
            val = phi.likelihood_params[key]
            if isinstance(val, jnp.ndarray):
                log_val = jnp.log(jnp.clip(val, a_min=1e-8))
            else:
                log_val = jnp.log(jnp.clip(float(val), a_min=1e-8))
            mu_val = mu.get(key, 0.0)
            total += jnp.sum((log_val - mu_val) ** 2)
    
    return 0.5 * lam * total


def make_hyperprior(kernel_fields: Optional[list[str]] = None,
                    kernel_lambda: float = 0.0,
                    kernel_log_lambda: float = 0.0,
                    kernel_log_mu: Optional[Dict[str, float]] = None,
                    z_lambda: float = 0.0,
                    likelihood_keys: Optional[list[str]] = None,
                    likelihood_lambda: float = 0.0,
                    likelihood_log_lambda: float = 0.0,
                    likelihood_log_mu: Optional[Dict[str, float]] = None) -> Callable[[Phi, Any, Any, Any], jnp.ndarray]:
    """
    Factory function to create a hyperprior function.
    
    Returns a function (phi, X, Y, key=None) -> scalar that can be used
    in TargetEnergy.extra or wrapped around energy in all inference methods.
    
    Args:
        kernel_fields: Kernel parameter fields to penalize
        kernel_lambda: L2 penalty on kernel params
        kernel_log_lambda: Log-L2 penalty on kernel params
        kernel_log_mu: Optional dict of mean values for log kernel params
        z_lambda: L2 penalty on Z
        likelihood_keys: Likelihood parameter keys to penalize
        likelihood_lambda: L2 penalty on likelihood params
        likelihood_log_lambda: Log-L2 penalty on likelihood params
        likelihood_log_mu: Optional dict of mean values for log likelihood params
    
    Returns:
        Function (phi, X, Y, key=None) -> scalar hyperprior value
    
    Examples:
        >>> # RBF kernel + Gaussian likelihood
        >>> hyperprior = make_hyperprior(
        ...     kernel_log_lambda=0.1,
        ...     kernel_fields=["lengthscale", "variance"],
        ...     likelihood_log_lambda=0.1,
        ...     likelihood_keys=["noise_var"],
        ... )
        >>> 
        >>> # Use in TargetEnergy
        >>> target = TargetEnergy(
        ...     inertial=inertial_energy,
        ...     extra=[hyperprior],  # Add as extra term
        ... )
    """
    def hyperprior_fn(phi: Phi, X: Any = None, Y: Any = None, key: Any = None) -> jnp.ndarray:
        E = jnp.array(0.0)
        
        if kernel_lambda > 0.0:
            E += kernel_l2_hyperprior(phi, fields=kernel_fields, lam=kernel_lambda)
        
        if kernel_log_lambda > 0.0:
            E += kernel_log_l2_hyperprior(phi, fields=kernel_fields, lam=kernel_log_lambda, mu=kernel_log_mu)
        
        if z_lambda > 0.0:
            E += z_l2_hyperprior(phi, lam=z_lambda)
        
        if likelihood_lambda > 0.0:
            E += likelihood_l2_hyperprior(phi, keys=likelihood_keys, lam=likelihood_lambda)
        
        if likelihood_log_lambda > 0.0:
            E += likelihood_log_l2_hyperprior(phi, keys=likelihood_keys, lam=likelihood_log_lambda, mu=likelihood_log_mu)
        
        return E
    
    return hyperprior_fn



__all__ = [
    "kernel_l2_hyperprior",
    "kernel_log_l2_hyperprior",
    "z_l2_hyperprior",
    "likelihood_l2_hyperprior",
    "likelihood_log_l2_hyperprior",
    "make_hyperprior",
]
