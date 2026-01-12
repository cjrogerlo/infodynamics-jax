# infodynamics_jax/gp/__init__.py
"""
Gaussian Process components.

This package provides:
  - kernels: GP kernel functions and parameters
  - likelihoods: Likelihood functions for different observation models
  - ansatz: Ansatz-based estimators for non-conjugate likelihoods
  - sparsify: Sparse GP approximations (FITC, etc.)
"""
from .kernels import get as get_kernel
from .likelihoods import get as get_likelihood

__all__ = [
    "get_kernel",
    "get_likelihood",
]
