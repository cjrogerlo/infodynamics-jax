# infodynamics_jax/inference/particle/__init__.py
"""
Particle-based inference methods.

This module provides particle-based methods:
  - annealed.py: β-annealed SMC (thermodynamic path)
  - ibis.py: IBIS (data streaming path, Bayesian filtering)
"""
from .annealed import AnnealedSMC, AnnealedSMCCFG, SMCRun
from .ibis import IBIS, IBISCFG, IBISRun

__all__ = [
    "AnnealedSMC", "AnnealedSMCCFG", "SMCRun",
    "IBIS", "IBISCFG", "IBISRun",
]
