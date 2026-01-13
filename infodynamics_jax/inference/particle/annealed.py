# infodynamics_jax/inference/particle/annealed.py
"""
Annealed Sequential Monte Carlo (SMC) via β-annealing.

This module implements β-annealed SMC on a FIXED dataset:
  - Target: π_β(φ) ∝ p(φ) p(y|φ)^β, where β ∈ [0,1]
  - Evolution axis: inverse temperature (thermodynamic path)
  - Weight update: Δlogw = -Δβ * E(φ)

⚠️ This is NOT IBIS (Iterated Batch Importance Sampling):
  - IBIS evolves along data stream: p(φ | y_{1:t})
  - IBIS weight update: logw += log p(y_t | φ)
  - IBIS requires batching/streaming (see future inference/particle/ibis.py)

⚠️ This is NOT SVGP inference:
  - SVGP requires variational parameters η = (m, S) + mini-batch
  - This is collapsed Bayesian hyperparameter inference (u already marginalised)
  - Energy is deterministic (Gaussian likelihood case)

This is a clean, correct, provable baseline for thermodynamic integration.
It deliberately does NOT include:
  - Data batching/streaming
  - IBIS logic
  - SVGP-specific inference

This preserves theoretical freedom for future extensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.tree_util import tree_map

from ...energy.base import EnergyTerm
from ..base import InferenceMethod
from .resampling import multinomial_resample, effective_sample_size
from .rejuvenation import hmc_rejuvenate, mala_rejuvenate, nuts_rejuvenate
from .schedules import make_beta_schedule


@dataclass(frozen=True)
class AnnealedSMCCFG:
    """Configuration for annealed SMC."""
    n_particles: int = 128
    betas: Optional[jnp.ndarray] = None  # If provided, overrides schedule_type
    n_steps: int = 32
    schedule_type: str = "linear"  # "linear", "geometric", "power"
    schedule_kwargs: Optional[dict] = None  # Additional args for schedule (e.g., alpha, power)
    ess_threshold: float = 0.5
    rejuvenation: str = "hmc"  # "hmc", "mala", "nuts", or None
    rejuvenation_steps: int = 1
    step_size: float = 1e-2  # For HMC/MALA/NUTS rejuvenation
    n_leapfrog: int = 4  # For HMC rejuvenation
    jit: bool = True
    verbose: bool = False


@dataclass
class SMCRun:
    """Annealed SMC run results."""
    particles: Any  # pytree stacked [P, ...]
    logw: jnp.ndarray  # shape [P]
    ess_trace: jnp.ndarray  # shape [n_steps]
    logZ_est: float
    betas: jnp.ndarray  # shape [n_steps+1]

    @property
    def beta_trace(self) -> jnp.ndarray:
        """Backward-compatible alias for betas."""
        # Match ESS trace length (n_steps) for legacy plotting code.
        return self.betas[1:]


class AnnealedSMC(InferenceMethod):
    """
    Annealed Sequential Monte Carlo via β-annealing.
    
    Samples from tempered distributions:
        π_β(φ) ∝ p(φ) p(y|φ)^β
    
    where β ∈ [0,1] is the inverse temperature parameter.
    
    Evolution:
        - β = 0: prior p(φ)
        - β = 1: posterior p(φ|y)
        - Intermediate: tempered distributions
    
    Weight update (thermodynamic integration):
        Δlogw_i = -(β_t - β_{t-1}) * E(φ_i)
    
    This is a THERMODYNAMIC path, not a data streaming path.
    For IBIS (data streaming), see future inference/particle/ibis.py.
    
    Args:
        cfg: AnnealedSMCCFG configuration
    """
    
    def __init__(self, cfg: AnnealedSMCCFG = AnnealedSMCCFG()):
        self.cfg = cfg


    def run(
        self, 
        energy: EnergyTerm, 
        init_particles_fn: Callable[[jax.random.PRNGKey, int], Any], 
        *, 
        key, 
        energy_args=(), 
        energy_kwargs=None
    ) -> SMCRun:
        """
        Run annealed SMC.
        
        Args:
            energy: Energy term to sample from
            init_particles_fn: Function (key, n_particles) -> initial particles
            key: PRNG key
            energy_args: Additional arguments for energy
            energy_kwargs: Additional keyword arguments for energy
        
        Returns:
            SMCRun with particles, log weights, ESS trace, and logZ estimate
        """
        if energy_kwargs is None:
            energy_kwargs = {}

        cfg = self.cfg
        n_particles = cfg.n_particles
        n_steps = cfg.n_steps
        ess_threshold = cfg.ess_threshold
        rejuvenation = cfg.rejuvenation
        jit = cfg.jit

        # Generate beta schedule
        if cfg.betas is not None:
            betas = cfg.betas
        else:
            schedule_kwargs = cfg.schedule_kwargs or {}
            betas = make_beta_schedule(
                schedule_type=cfg.schedule_type,
                n_steps=n_steps,
                beta_min=0.0,
                beta_max=1.0,
                **schedule_kwargs
            )

        # Split key for initialization
        key, subkey = random.split(key)
        particles = init_particles_fn(subkey, n_particles)
        logw = jnp.zeros(n_particles)
        ess_trace = jnp.zeros(n_steps)

        logZ_est = 0.0

        def step_fn(carry, t):
            particles, logw, logZ_est, key = carry
            beta_prev = betas[t]
            beta_curr = betas[t + 1]
            delta_beta = beta_curr - beta_prev

            # Split keys for this step
            key, key_resample, key_rejuv = random.split(key, 3)

            # Incremental weights: Δlogw = -(beta_t - beta_{t-1}) * E(phi)
            # This is THERMODYNAMIC integration (path sampling), NOT data update.
            # For IBIS, weight update would be: logw += log p(y_t | phi)
            def energy_eval(phi):
                return energy(phi, *energy_args, **energy_kwargs)

            energies = jax.vmap(energy_eval)(particles)
            
            # Incremental weights: Δlogw = -(beta_t - beta_{t-1}) * E(phi)
            # This is the CORRECT incremental weight for this step
            delta_logw = -delta_beta * energies
            
            # Store normalized weights BEFORE resampling for logZ calculation
            # This is the correct way to compute incremental normalizer in annealed SMC
            max_logw_prev = jnp.max(logw)
            w_prev_norm = jnp.exp(logw - max_logw_prev)
            w_prev_norm = w_prev_norm / jnp.sum(w_prev_norm)
            
            # Update cumulative log weights
            logw = logw + delta_logw
            
            # Compute ESS from normalized weights
            max_logw = jnp.max(logw)
            w_norm = jnp.exp(logw - max_logw)
            w_norm = w_norm / jnp.sum(w_norm)
            ess = 1.0 / jnp.sum(w_norm ** 2)
            ess_trace = jnp.array(ess)

            # CORRECT logZ update for annealed SMC / AIS:
            # Incremental normalizer = log sum_i (w_{t-1,i} * exp(delta_logw_i))
            # This accounts for the weighted average of incremental weights
            # This is the standard AIS/SMC evidence estimator
            weighted_delta_logw = delta_logw + jnp.log(w_prev_norm + 1e-10)  # Add small epsilon for numerical stability
            max_weighted_delta = jnp.max(weighted_delta_logw)
            logZ_increment = max_weighted_delta + jnp.log(jnp.sum(jnp.exp(weighted_delta_logw - max_weighted_delta)))
            logZ_est = logZ_est + logZ_increment

            # Resample if ESS < threshold
            def resample_particles(particles, logw, key):
                from jax.tree_util import tree_map
                indices = multinomial_resample(key, logw, n_particles)
                particles_resampled = tree_map(lambda x: x[indices], particles)
                logw_resampled = jnp.zeros_like(logw)
                return particles_resampled, logw_resampled

            do_resample = ess < ess_threshold * n_particles
            particles, logw = jax.lax.cond(
                do_resample,
                lambda _: resample_particles(particles, logw, key_resample),
                lambda _: (particles, logw),
                operand=None,
            )

            # Rejuvenation step
            if rejuvenation == "hmc":
                # Target tempered distribution: U_beta = beta * E(phi)
                def energy_fn(phi):
                    return beta_curr * energy(phi, *energy_args, **energy_kwargs)
                particles = hmc_rejuvenate(
                    key_rejuv,
                    particles,
                    energy_fn,
                    step_size=cfg.step_size,
                    n_leapfrog=cfg.n_leapfrog,
                    n_steps=cfg.rejuvenation_steps,
                    jit=jit,
                )
            elif rejuvenation == "mala":
                # Target tempered distribution: U_beta = beta * E(phi)
                def energy_fn(phi):
                    return beta_curr * energy(phi, *energy_args, **energy_kwargs)
                particles = mala_rejuvenate(
                    key_rejuv,
                    particles,
                    energy_fn,
                    step_size=cfg.step_size,
                    n_steps=cfg.rejuvenation_steps,
                    jit=jit,
                )
            elif rejuvenation == "nuts":
                # Target tempered distribution: U_beta = beta * E(phi)
                def energy_fn(phi):
                    return beta_curr * energy(phi, *energy_args, **energy_kwargs)
                particles = nuts_rejuvenate(
                    key_rejuv,
                    particles,
                    energy_fn,
                    step_size=cfg.step_size,
                    max_tree_depth=getattr(cfg, "max_tree_depth", 10),
                    delta_max=getattr(cfg, "delta_max", 1000.0),
                    n_steps=cfg.rejuvenation_steps,
                    jit=jit,
                )

            return (particles, logw, logZ_est, key), ess_trace

        def iter_steps():
            if not cfg.verbose:
                return range(n_steps)
            try:
                from tqdm.auto import tqdm
                return tqdm(range(n_steps), total=n_steps, desc="AnnealedSMC")
            except Exception:
                return range(n_steps)

        if jit:
            (particles, logw, logZ_est, _), ess_trace = lax.scan(
                step_fn, (particles, logw, logZ_est, key), jnp.arange(n_steps)
            )
        else:
            ess_trace_list = []
            for t in iter_steps():
                (particles, logw, logZ_est, key), ess_t = step_fn((particles, logw, logZ_est, key), t)
                ess_trace_list.append(ess_t)
            ess_trace = jnp.array(ess_trace_list)

        return SMCRun(
            particles=particles,
            logw=logw,
            ess_trace=ess_trace,
            logZ_est=logZ_est,
            betas=betas,
        )
        
