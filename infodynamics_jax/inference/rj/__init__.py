# infodynamics_jax/inference/rj/__init__.py
"""
Reversible Jump MCMC methods for trans-dimensional inference.

This module provides:
- RJMCMC: For conjugate (Gaussian) likelihoods using VFE
- RJVMC: For non-conjugate likelihoods using InertialEnergy + VariationalState
"""
from .state import RJState
from .rjmcmc import RJMCMC, RJMCMCCFG, RJMCMCRun
from .rjvmc import RJVMC, RJVMCCFG, RJVMCRun

__all__ = [
    "RJState",
    "RJMCMC", "RJMCMCCFG", "RJMCMCRun",
    "RJVMC", "RJVMCCFG", "RJVMCRun",
]
