# infodynamics_jax/inference/optimisation/__init__.py
"""
Type-II optimisation methods.

This module provides:
- VGA: Variational Gaussian Approximation (optimisation algorithm)
- TypeII: Type-II optimiser (for ML-II and MAP-II)
- VFE objective: Type-II variational objective (optimisation target)

NOTE: These are for type-II / variational inference only.
For Bayesian inference (SMC/MCMC), use energy/inertial.py instead.
"""
from .vga import VGA, VGACFG, VGARun
from .typeii import TypeII, TypeIICFG, TypeIIRun
from .vfe import vfe_objective, make_vfe_objective

__all__ = [
    # Optimisation algorithms
    "VGA", "VGACFG", "VGARun",
    "TypeII", "TypeIICFG", "TypeIIRun",
    # Optimisation objectives
    "vfe_objective", "make_vfe_objective",
]
