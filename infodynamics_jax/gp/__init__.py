# infodynamics_jax/gp/__init__.py
"""
Gaussian Process components.

This package provides:
  - kernels: GP kernel functions and parameters
  - likelihoods: Likelihood functions for different observation models
  - ansatz: Ansatz-based estimators for non-conjugate likelihoods
  - sparsify: Sparse GP approximations (FITC, etc.)
  - predict: Prediction utilities for Type-II inference
"""
from .kernels import get as get_kernel
from .likelihoods import get as get_likelihood
from .predict import predict_typeii

__all__ = [
    "get_kernel",
    "get_likelihood",
    "predict_typeii",
]
