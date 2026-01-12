# infodynamics_jax/inference/__init__.py
from __future__ import annotations

"""
Inference layer (dynamics).

This package is intentionally reserved for *inference dynamics* (operators / kernels)
rather than end-to-end scripts. A concrete "method" (e.g. annealed SMC with HMC
rejuvenation, ML-II optimisation, RJ-MCMC, EGPF, etc.) should be implemented as a
*composition* of:
  - a model energy / log-density (energy.*),
  - a schedule / time parametrisation (schedules.*), and
  - one or more dynamics operators (inference.*).

Rationale:
  - keeps model components (kernels, likelihoods, sparsified operators) reusable across inference paradigms,
  - makes it possible to compare dynamics under the same energy landscape,
  - supports variable-dimension structure (e.g. RJ moves) without contaminating
    the structural state container (core.phi.Phi).
"""

from .base import InferenceMethod
from .optimisation import (
    VGA, VGACFG, VGARun,
    TypeII, TypeIICFG, TypeIIRun,
    vfe_objective, make_vfe_objective
)
from .sampling import (
    HMC, HMCCFG, HMCRun,
    NUTS, NUTSCFG, NUTSRun,
    MALA, MALACFG, MALARun,
    SliceSampler, SliceCFG, SliceRun
)
from .particle import (
    AnnealedSMC, AnnealedSMCCFG, SMCRun,
    IBIS, IBISCFG, IBISRun
)

__all__ = [
    "InferenceMethod",
    "VGA", "VGACFG", "VGARun",
    "TypeII", "TypeIICFG", "TypeIIRun",
    "vfe_objective", "make_vfe_objective",
    "HMC", "HMCCFG", "HMCRun",
    "NUTS", "NUTSCFG", "NUTSRun",
    "MALA", "MALACFG", "MALARun",
    "SliceSampler", "SliceCFG", "SliceRun",
    "AnnealedSMC", "AnnealedSMCCFG", "SMCRun",
    "IBIS", "IBISCFG", "IBISRun",
]
