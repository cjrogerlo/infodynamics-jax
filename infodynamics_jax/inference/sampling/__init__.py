# infodynamics_jax/inference/sampling/__init__.py
"""
MCMC sampling kernels.

This module provides MCMC transition kernels that can be used standalone
or composed into larger inference methods (e.g. SMC rejuvenation, RJ-MCMC).

All kernels follow the same interface:
    kernel_fn(phi_current, energy, *, key, ...) -> phi_next, diagnostics
"""
from .hmc import HMC, HMCCFG, HMCRun
from .nuts import NUTS, NUTSCFG, NUTSRun
from .mala import MALA, MALACFG, MALARun
from .slice import SliceSampler, SliceCFG, SliceRun

__all__ = [
    "HMC", "HMCCFG", "HMCRun",
    "NUTS", "NUTSCFG", "NUTSRun",
    "MALA", "MALACFG", "MALARun",
    "SliceSampler", "SliceCFG", "SliceRun",
]

