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


@dataclass(frozen=True)
class AnnealedSMCCFG:
    """Configuration for annealed SMC."""
    n_particles: int = 128
    betas: Optional[jnp.ndarray] = None
    n_steps: int = 32
    ess_threshold: float = 0.5
    rejuvenation: str = "hmc"  # "hmc", "mala", "nuts", or None
    rejuvenation_steps: int = 1
    jit: bool = True


@dataclass
class SMCRun:
    """Annealed SMC run results."""
    particles: Any  # pytree stacked [P, ...]
    logw: jnp.ndarray  # shape [P]
    ess_trace: jnp.ndarray  # shape [n_steps]
    logZ_est: float
    betas: jnp.ndarray  # shape [n_steps+1]


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

        if cfg.betas is None:
            betas = jnp.linspace(0.0, 1.0, n_steps + 1)
        else:
            betas = cfg.betas

        particles = init_particles_fn(key, n_particles)
        logw = jnp.zeros(n_particles)
        ess_trace = jnp.zeros(n_steps)

        logZ_est = 0.0

        def step_fn(carry, t):
            particles, logw, logZ_est = carry
            beta_prev = betas[t]
            beta_curr = betas[t + 1]
            delta_beta = beta_curr - beta_prev

            # Incremental weights: Δlogw = -(beta_t - beta_{t-1}) * E(phi)
            # This is THERMODYNAMIC integration (path sampling), NOT data update.
            # For IBIS, weight update would be: logw += log p(y_t | phi)
            def energy_eval(phi):
                return energy(phi, *energy_args, **energy_kwargs)

            energies = jax.vmap(energy_eval)(particles)
            logw = logw - delta_beta * energies  # β-annealing: thermodynamic path

            max_logw = jnp.max(logw)
            w_norm = jnp.exp(logw - max_logw)
            w_norm = w_norm / jnp.sum(w_norm)
            ess = 1.0 / jnp.sum(w_norm ** 2)
            ess_trace = jnp.array(ess)

            # Update logZ_est (log normalizing constant estimate)
            logZ_est = logZ_est + max_logw + jnp.log(jnp.sum(jnp.exp(logw - max_logw))) - jnp.log(n_particles)

            # Resample if ESS < threshold
            def resample_particles(particles, logw, key):
                indices = multinomial_resample(key, logw, n_particles)
                particles_resampled = jax.tree_map(lambda x: x[indices], particles)
                logw_resampled = jnp.zeros_like(logw)
                return particles_resampled, logw_resampled

            key_resample = random.fold_in(key, t)
            do_resample = ess < ess_threshold * n_particles
            particles, logw = jax.lax.cond(
                do_resample,
                lambda _: resample_particles(particles, logw, key_resample),
                lambda _: (particles, logw),
                operand=None,
            )

            # Rejuvenation step
            if rejuvenation == "hmc":
                key_rejuv = random.fold_in(key, t + 10000)
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
                    jit=cfg.jit,
                )
            elif rejuvenation == "mala":
                key_rejuv = random.fold_in(key, t + 10000)
                # Target tempered distribution: U_beta = beta * E(phi)
                def energy_fn(phi):
                    return beta_curr * energy(phi, *energy_args, **energy_kwargs)
                particles = mala_rejuvenate(
                    key_rejuv,
                    particles,
                    energy_fn,
                    step_size=cfg.step_size,
                    n_steps=cfg.rejuvenation_steps,
                    jit=cfg.jit,
                )
            elif rejuvenation == "nuts":
                key_rejuv = random.fold_in(key, t + 10000)
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
                    jit=cfg.jit,
                )

            return (particles, logw, logZ_est), ess_trace

        if jit:
            (particles, logw, logZ_est), ess_trace = lax.scan(
                step_fn, (particles, logw, logZ_est), jnp.arange(n_steps)
            )
        else:
            ess_trace_list = []
            for t in range(n_steps):
                (particles, logw, logZ_est), ess_t = step_fn((particles, logw, logZ_est), t)
                ess_trace_list.append(ess_t)
            ess_trace = jnp.array(ess_trace_list)

        return SMCRun(
            particles=particles,
            logw=logw,
            ess_trace=ess_trace,
            logZ_est=logZ_est,
            betas=betas,
        )
        