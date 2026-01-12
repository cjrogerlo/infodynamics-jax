# infodynamics_jax/inference/particle/__init__.py
"""
Particle-based inference methods.

This module provides particle-based methods:
  - annealed.py: β-annealed SMC (thermodynamic path)
  - ibis.py: IBIS (data streaming path, Bayesian filtering)
  - schedules.py: Beta scheduling strategies for annealed SMC
"""
from .annealed import AnnealedSMC, AnnealedSMCCFG, SMCRun
from .ibis import IBIS, IBISCFG, IBISRun
from .schedules import (
    linear_schedule,
    geometric_schedule,
    power_schedule,
    make_beta_schedule,
)

__all__ = [
    "AnnealedSMC", "AnnealedSMCCFG", "SMCRun",
    "IBIS", "IBISCFG", "IBISRun",
    "linear_schedule",
    "geometric_schedule",
    "power_schedule",
    "make_beta_schedule",
]
