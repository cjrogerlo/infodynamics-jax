# infodynamics_jax/inference/optimisation/__init__.py
"""
Type-II optimisation methods.

This module provides:
- VGA: Variational Gaussian Approximation (optimisation algorithm)
- MAP2: Maximum A Posteriori Type-II (general-purpose optimiser)
- VFE objective: Type-II variational objective (optimisation target)

NOTE: These are for type-II / variational inference only.
For Bayesian inference (SMC/MCMC), use energy/inertial.py instead.
"""
from .vga import VGA, VGACFG, VGARun
from .map2 import MAP2, MAP2CFG, MAP2Run
from .vfe import vfe_objective, make_vfe_objective

__all__ = [
    # Optimisation algorithms
    "VGA", "VGACFG", "VGARun",
    "MAP2", "MAP2CFG", "MAP2Run",
    # Optimisation objectives
    "vfe_objective", "make_vfe_objective",
]
