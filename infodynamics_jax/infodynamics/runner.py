# infodynamics_jax/infodynamics/runner.py
"""
Orchestration layer: (energy + inference method) -> result.

This module provides a clean entry point that wires together:
  - Data (via energy_args/kwargs)
  - Energy functions (EnergyTerm)
  - Inference methods (InferenceMethod)
  - Standardized diagnostics

The runner is algorithm-agnostic: it doesn't assume SMC/MCMC/optimisation,
just delegates to the method's .run() and standardizes the output.

Design principles:
  - No assumptions about algorithm class (SMC/MCMC/VGA/etc)
  - energy_args/kwargs is the *only* sanctioned way to feed data into energy
  - Diagnostics are opportunistically extracted (don't rely on them elsewhere)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Dict

import jax
import jax.numpy as jnp

from ..energy.base import EnergyTerm
from ..inference.base import InferenceMethod


@dataclass(frozen=True)
class RunCFG:
    """
    Thin orchestration config.

    Notes:
    - `energy_args/kwargs` is the *only* sanctioned way to feed data into energy.
    - No assumptions about algorithm class (SMC/MCMC/VGA/etc).
    - JIT is owned by each method; runner does not wrap jit around unknown pytrees.
    """
    jit: bool = True


@dataclass
class RunOut:
    """
    Standardised output from a run.

    `result` is the method-specific run object (e.g. VGARun / SMCRun / HMCRun / SliceRun),
    but we also expose some common fields when available.
    """
    result: Any
    diagnostics: Dict[str, Any]


def run(
    *,
    key: jax.random.KeyArray,
    method: InferenceMethod,
    energy: EnergyTerm,
    phi_init: Any,
    energy_args: Tuple[Any, ...] = (),
    energy_kwargs: Optional[Dict[str, Any]] = None,
    hyperprior: Optional[Callable[[Any], jnp.ndarray]] = None,
    cfg: RunCFG = RunCFG(),
) -> RunOut:
    """
    One-shot runner: (energy + inference method) -> result.

    The method controls what it means to 'run':
      - optimisation: returns argmin (approx) + traces
      - MCMC: returns samples + diagnostics
      - SMC: returns particles/weights + schedules + diagnostics

    Args:
        key: PRNG key
        method: Inference method (VGA, HMC, NUTS, MALA, SliceSampler, AnnealedSMC, IBIS, etc.)
        energy: Energy term to optimize/sample from
        phi_init: Initial parameter state
        energy_args: Positional arguments for energy (typically (X, Y) for data)
        energy_kwargs: Keyword arguments for energy
        hyperprior: Optional function (phi) -> scalar hyperprior (-log p(phi))
                   This is NOT an energy term, but a regularization term.
                   Can be added to the objective: E(phi) + hyperprior(phi)
                   Works with all inference methods (MAP2, HMC, NUTS, SMC, IBIS, etc.)
        cfg: Runner configuration

    Returns:
        RunOut with method-specific result and standardized diagnostics

    Examples:
        >>> # Type-II optimisation (VGA) with VFE objective
        >>> from infodynamics_jax.infodynamics import run
        >>> from infodynamics_jax.inference.optimisation import VGA, VGACFG, make_vfe_objective
        >>> 
        >>> vfe_obj = make_vfe_objective(kernel_fn=kernel, residual="fitc")
        >>> method = VGA(cfg=VGACFG(steps=200, lr=1e-2))
        >>> 
        >>> out = run(
        ...     key=key,
        ...     method=method,
        ...     energy=vfe_obj,
        ...     phi_init=phi_init,
        ...     energy_args=(X, Y),
        ... )
        >>> print(out.diagnostics)
        >>> print(out.result)  # VGARun

        >>> # MCMC sampling (HMC)
        >>> from infodynamics_jax.inference.sampling import HMC, HMCCFG
        >>> from infodynamics_jax.energy import TargetEnergy, InertialEnergy, PriorEnergy
        >>> 
        >>> inertial = InertialEnergy(...)
        >>> prior = PriorEnergy([...])
        >>> target = TargetEnergy(inertial=inertial, prior=prior)
        >>> 
        >>> method = HMC(cfg=HMCCFG(step_size=1e-2, n_samples=256))
        >>> out = run(
        ...     key=key,
        ...     method=method,
        ...     energy=target,
        ...     phi_init=phi_init,
        ...     energy_args=(X, Y),
        ... )
        >>> print(out.result.samples)  # HMCRun.samples
    """
    if energy_kwargs is None:
        energy_kwargs = {}

    # Wrap energy with hyperprior if provided
    # This works for all inference methods (MAP2, HMC, SMC, IBIS, etc.)
    if hyperprior is not None:
        # Create a wrapper that adds hyperprior to energy
        original_energy = energy
        
        class EnergyWithHyperprior:
            """Wrapper that adds hyperprior to energy (not an EnergyTerm, but callable)."""
            def __call__(self, phi, *args, **kwargs):
                E = original_energy(phi, *args, **kwargs)
                # hyperprior only depends on phi
                E = E + hyperprior(phi)
                return E
        
        energy = EnergyWithHyperprior()

    # JIT is owned by each method; runner does not wrap jit around unknown pytrees.
    out = method.run(  # type: Any (method-specific Run object, e.g. VGARun, HMCRun, SMCRun)
        energy=energy,
        phi_init=phi_init,
        key=key,
        energy_args=energy_args,
        energy_kwargs=energy_kwargs,
    )

    diagnostics = {"method": method.__class__.__name__}
    # Opportunistically extract common diagnostics (don't rely on them elsewhere)
    if hasattr(out, "accept_rate"):
        diagnostics["accept_rate"] = float(out.accept_rate)
    if hasattr(out, "energy_trace") and len(out.energy_trace) > 0:
        diagnostics["final_energy"] = float(out.energy_trace[-1])
        diagnostics["initial_energy"] = float(out.energy_trace[0])
        diagnostics["energy_change"] = float(out.energy_trace[0] - out.energy_trace[-1])
    if hasattr(out, "grad_norm_trace") and len(out.grad_norm_trace) > 0:
        diagnostics["final_grad_norm"] = float(out.grad_norm_trace[-1])

    return RunOut(result=out, diagnostics=diagnostics)
