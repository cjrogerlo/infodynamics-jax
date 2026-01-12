# infodynamics_jax/inference/base.py
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any

from ..energy.base import EnergyTerm


@runtime_checkable
class InferenceMethod(Protocol):
    """
    Protocol for inference methods.

    Design principles
    -----------------
    - An InferenceMethod consumes an EnergyTerm and performs inference.
    - It MUST treat EnergyTerm as a black box (no inspection of internal structure).
    - It MUST NOT assume any specific energy signature beyond EnergyTerm contract.
    - It MAY accept configuration for annealing, schedules, kernel parameters, etc.

    Canonical contract
    ------------------
    The exact `run` signature is method-specific, but all methods:
    - accept an EnergyTerm as input,
    - perform inference over the energy landscape,
    - return inference results (samples, weights, diagnostics, etc.).

    NOTE:
    - This protocol does NOT specify state structure, sampling strategy, or
      convergence criteria. These are implementation details of each method.
    - Optimisation-based methods (e.g. Type-II, VGA) are special cases where
      sampling degenerates to point estimates.
    - Annealed SMC, MCMC, RJMCMC, IBIS, etc. are all valid implementations.
    """

    def run(self, energy: EnergyTerm, *args, **kwargs) -> Any:
        """
        Run inference on the given energy.

        Parameters
        ----------
        energy : EnergyTerm
            The energy to perform inference on. Must be treated as a black box.

        Returns
        -------
        Any
            Inference results (method-specific: samples, weights, diagnostics, etc.).

        Notes
        -----
        - Method-specific arguments (initial state, PRNGKey, iterations, etc.)
          should be passed via *args or **kwargs.
        - This method MUST NOT inspect energy internals beyond calling energy(*args).
        """
        ...