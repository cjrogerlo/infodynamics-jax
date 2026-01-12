# infodynamics_jax/inference/particle/ibis.py
"""
Iterated Batch Importance Sampling (IBIS).

IBIS is a Bayesian filtering method that processes data sequentially:
  - Evolution axis: data stream (y_1, y_2, ..., y_T)
  - Target: p(φ | y_{1:t}) at each time t
  - Weight update: logw += log p(y_t | φ)

This is fundamentally different from β-annealed SMC:
  - Annealed SMC: thermodynamic path (β ∈ [0,1])
  - IBIS: data streaming path (t = 1, 2, ..., T)

IBIS corresponds to SVGP when using sparse GP with inducing points Z.

Key design:
  - Accepts data stream / batch iterator
  - Weight update is likelihood increment, NOT energy-based β-annealing
  - Can reuse HMC rejuvenation kernel, but targets p(φ | y_{1:t})
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Iterator, Union

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.tree_util import tree_map

from ...energy.base import EnergyTerm
from ...core.data import SupervisedData
from ..base import InferenceMethod
from .resampling import multinomial_resample, effective_sample_size
from .rejuvenation import hmc_rejuvenate, mala_rejuvenate, nuts_rejuvenate


@dataclass(frozen=True)
class IBISCFG:
    """Configuration for IBIS."""
    n_particles: int = 128
    ess_threshold: float = 0.5
    rejuvenation: str = "hmc"  # "hmc", "mala", "nuts", or None
    rejuvenation_steps: int = 1
    step_size: float = 1e-2
    n_leapfrog: int = 4
    jit: bool = True


@dataclass
class IBISRun:
    """IBIS run results."""
    particles: Any  # pytree stacked [P, ...]
    logw: jnp.ndarray  # shape [P]
    ess_trace: jnp.ndarray  # shape [T] (one ESS per time step)
    logZ_trace: jnp.ndarray  # shape [T] (cumulative log evidence)
    time_steps: int  # Number of data batches processed


class IBIS(InferenceMethod):
    """
    Iterated Batch Importance Sampling (IBIS).
    
    Processes data sequentially, updating posterior p(φ | y_{1:t}) at each step.
    
    Weight update (Bayesian filtering):
        logw_i^{(t)} = logw_i^{(t-1)} + log p(y_t | φ_i)
    
    This is a DATA STREAMING path, not a thermodynamic path.
    For β-annealing, see inference/particle/annealed.py.
    
    Args:
        cfg: IBISCFG configuration
    """
    
    def __init__(self, cfg: IBISCFG = IBISCFG()):
        self.cfg = cfg


    def _compute_log_likelihood_increment(
        self,
        energy: EnergyTerm,
        particles: Any,
        X_batch: jnp.ndarray,
        Y_batch: jnp.ndarray,
        log_likelihood_fn: Optional[Callable] = None,
        energy_kwargs: Optional[dict] = None,
    ) -> jnp.ndarray:
        """
        Compute log p(y_batch | φ) for each particle.
        
        This method handles the fact that IBIS needs log p(y|φ), which is different
        from the energy E[-log p(y|f,φ)] that energy layer provides.
        
        For non-conjugate likelihoods, computing log p(y|φ) requires calling ansatz
        (GH/MC) to compute log ∫ p(y|f,φ) q(f|φ) df. This is a different computation
        than E[-log p(y|f,φ)].
        
        Args:
            energy: EnergyTerm (typically InertialEnergy)
            particles: Stacked particles pytree [P, ...]
            X_batch: Batch inputs (B, Q)
            Y_batch: Batch observations (B,) or (B, D)
            log_likelihood_fn: Optional function (phi, X, Y, key=None) -> log p(Y | X, phi)
                If provided, used for accurate non-conjugate likelihood.
                For non-conjugate, this should internally call ansatz (GH/MC).
                If None, uses -energy as approximation.
            energy_kwargs: Optional kwargs for energy
        
        Returns:
            log_likelihoods: shape [P] - log p(y_batch | φ_i) for each particle
        
        Note:
            Algorithm layer differences:
            - sampling/ (HMC, NUTS): only need energy(phi) -> scalar (ansatz handled internally)
            - optimisation/ (MAP2): only need energy(phi) -> scalar (ansatz handled internally)
            - particle/annealed: only need energy(phi) -> scalar (ansatz handled internally)
            - particle/ibis: needs log p(y|φ), which for non-conjugate requires ansatz
            
            For Gaussian: -energy is accurate (up to constant that cancels in ratios).
            For non-conjugate: E[-log p(y|f,φ)] ≠ -log p(y|φ) due to Jensen's inequality.
            If accurate log likelihood is needed for non-conjugate, provide log_likelihood_fn
            that internally calls ansatz. Otherwise, -energy is used as approximation.
        """
        if log_likelihood_fn is not None:
            # Use provided log likelihood function (should call ansatz for non-conjugate)
            def loglik_eval(phi):
                # Pass key if available in energy_kwargs
                key = energy_kwargs.get("key", None) if energy_kwargs else None
                return log_likelihood_fn(phi, X_batch, Y_batch, key=key)
            log_likelihoods = jax.vmap(loglik_eval)(particles)  # [P]
            return log_likelihoods
        
        # Fallback: use -energy as approximation
        # This works for Gaussian (accurate) and non-conjugate (approximate)
        if energy_kwargs is None:
            energy_kwargs = {}
        
        # Evaluate energy for each particle on this batch
        def energy_eval(phi):
            return energy(phi, X_batch, Y_batch, **energy_kwargs)
        
        energies = jax.vmap(energy_eval)(particles)  # [P]
        
        # For Gaussian likelihood, -energy is proportional to log-likelihood
        # (up to normalization constant that cancels in weight ratios)
        # For non-Gaussian, this is an approximation (Jensen's inequality)
        # Note: For non-conjugate, accurate computation would require calling ansatz
        # to compute log ∫ p(y|f,φ) q(f|φ) df, which is different from E[-log p(y|f,φ)]
        log_likelihoods = -energies  # [P]
        
        return log_likelihoods

    def run(
        self,
        energy: EnergyTerm,
        init_particles_fn: Callable[[jax.random.PRNGKey, int], Any],
        data_stream: Union[Iterator[SupervisedData], list[SupervisedData]],
        *,
        key: jax.random.PRNGKey,
        log_likelihood_fn: Optional[Callable] = None,
        energy_kwargs: Optional[dict] = None,
    ) -> IBISRun:
        """
        Run IBIS on a data stream.
        
        Args:
            energy: EnergyTerm (typically InertialEnergy for data-dependent term)
            init_particles_fn: Function (key, n_particles) -> initial particles
            data_stream: Iterator or list of SupervisedData batches
            key: PRNG key
            log_likelihood_fn: Optional function (phi, X, Y, key=None) -> log p(Y | X, phi)
                For non-conjugate likelihoods, provide this for accurate weights.
                This function should internally call ansatz (GH/MC) to compute
                log ∫ p(y|f,φ) q(f|φ) df, which is different from E[-log p(y|f,φ)].
                If None, uses -energy(phi, X, Y) as approximation.
            energy_kwargs: Optional kwargs for energy
        
        Returns:
            IBISRun with particles, log weights, ESS trace, and logZ trace
        
        Note:
            Weight update: logw += log p(y_t | φ)
            This is DATA STREAMING, not β-annealing.
            
            Log likelihood computation:
            - Gaussian likelihood: -energy is accurate (up to constant that cancels in weight ratios)
            - Non-conjugate likelihood:
              * If log_likelihood_fn is provided: uses accurate log p(y|φ)
              * If None: uses -energy as approximation (Jensen's inequality)
                This approximation is acceptable for most IBIS use cases.
            
            Note: log_likelihood_fn is OPTIONAL. Default behavior (-energy) works for:
            - All Gaussian likelihood cases (accurate)
            - Most non-conjugate cases (approximate but acceptable)
        """
        if energy_kwargs is None:
            energy_kwargs = {}
        
        cfg = self.cfg
        n_particles = cfg.n_particles
        ess_threshold = cfg.ess_threshold
        rejuvenation = cfg.rejuvenation
        jit = cfg.jit
        
        # Initialize particles
        key, subkey = random.split(key)
        particles = init_particles_fn(subkey, n_particles)
        logw = jnp.zeros(n_particles)
        logZ_est = 0.0  # Initialize logZ_est, will be accumulated across steps
        accumulated_data = None  # Will accumulate data as we process batches
        
        # Convert data_stream to list if iterator
        if not isinstance(data_stream, list):
            data_batches = list(data_stream)
        else:
            data_batches = data_stream
        
        n_steps = len(data_batches)
        ess_trace = jnp.zeros(n_steps)
        logZ_trace = jnp.zeros(n_steps)
        
        def step_fn(carry, t_and_batch):
            particles, logw, logZ_est, accumulated_data, key = carry
            t, batch = t_and_batch
            
            # Split keys for this step
            key, key_resample, key_rejuv = random.split(key, 3)
            
            # IBIS weight update: logw += log p(y_t | φ)
            # This is the KEY difference from β-annealed SMC
            log_likelihood_increment = self._compute_log_likelihood_increment(
                energy=energy,
                particles=particles,
                X_batch=batch.X,
                Y_batch=batch.Y,
                log_likelihood_fn=log_likelihood_fn,
                energy_kwargs=energy_kwargs,
            )
            
            # Store normalized weights BEFORE resampling for logZ calculation
            # This is needed for correct IBIS evidence estimation
            max_logw_prev = jnp.max(logw)
            w_prev_norm = jnp.exp(logw - max_logw_prev)
            w_prev_norm = w_prev_norm / jnp.sum(w_prev_norm)
            
            logw = logw + log_likelihood_increment  # DATA UPDATE, not β-annealing
            
            # Accumulate data for rejuvenation (targets p(φ | y_{1:t}))
            # Concatenate current batch with accumulated data
            if accumulated_data is None:
                accumulated_X = batch.X
                accumulated_Y = batch.Y
            else:
                accumulated_X = jnp.concatenate([accumulated_data.X, batch.X], axis=0)
                accumulated_Y = jnp.concatenate([accumulated_data.Y, batch.Y], axis=0)
            accumulated_data = SupervisedData(accumulated_X, accumulated_Y)
            
            # Normalize and compute ESS
            max_logw = jnp.max(logw)
            w_norm = jnp.exp(logw - max_logw)
            w_norm = w_norm / jnp.sum(w_norm)
            ess = 1.0 / jnp.sum(w_norm ** 2)
            ess_trace = jnp.array(ess)
            
            # CORRECT logZ update for IBIS:
            # Incremental normalizer = log sum_i (w_{t-1,i} * exp(log_likelihood_increment_i))
            # This accounts for the weighted average of likelihood increments
            weighted_loglik = log_likelihood_increment + jnp.log(w_prev_norm + 1e-10)  # Add small epsilon for numerical stability
            max_weighted_loglik = jnp.max(weighted_loglik)
            logZ_increment = max_weighted_loglik + jnp.log(jnp.sum(jnp.exp(weighted_loglik - max_weighted_loglik)))
            logZ_est = logZ_est + logZ_increment  # Accumulate across steps
            logZ_trace = jnp.array(logZ_est)
            
            # Resample if ESS < threshold
            def resample_particles(particles, logw, key):
                indices = multinomial_resample(key, logw, n_particles)
                particles_resampled = jax.tree_util.tree_map(lambda x: x[indices], particles)
                logw_resampled = jnp.zeros_like(logw)
                return particles_resampled, logw_resampled
            
            do_resample = ess < ess_threshold * n_particles
            particles, logw = jax.lax.cond(
                do_resample,
                lambda _: resample_particles(particles, logw, key_resample),
                lambda _: (particles, logw),
                operand=None,
            )
            
            # Rejuvenation step (targets p(φ | y_{1:t}), not tempered distribution)
            if rejuvenation == "hmc":
                # Target full posterior: p(φ | y_{1:t})
                def energy_fn(phi):
                    return energy(phi, accumulated_data.X, accumulated_data.Y, **energy_kwargs)
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
                # Target full posterior: p(φ | y_{1:t})
                def energy_fn(phi):
                    return energy(phi, accumulated_data.X, accumulated_data.Y, **energy_kwargs)
                particles = mala_rejuvenate(
                    key_rejuv,
                    particles,
                    energy_fn,
                    step_size=cfg.step_size,
                    n_steps=cfg.rejuvenation_steps,
                    jit=cfg.jit,
                )
            elif rejuvenation == "nuts":
                # Target full posterior: p(φ | y_{1:t})
                def energy_fn(phi):
                    return energy(phi, accumulated_data.X, accumulated_data.Y, **energy_kwargs)
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
            
            return (particles, logw, logZ_est, accumulated_data, key), (ess_trace, logZ_trace)
        
        # Process data stream
        ess_trace_list = []
        logZ_trace_list = []
        for t, batch in enumerate(data_batches):
            (particles, logw, logZ_est, accumulated_data, key), (ess_t, logZ_t) = step_fn(
                (particles, logw, logZ_est, accumulated_data, key), (t, batch)
            )
            ess_trace_list.append(ess_t)
            logZ_trace_list.append(logZ_t)
        ess_trace = jnp.array(ess_trace_list)
        logZ_trace = jnp.array(logZ_trace_list)
        
        return IBISRun(
            particles=particles,
            logw=logw,
            ess_trace=ess_trace,
            logZ_trace=logZ_trace,
            time_steps=n_steps,
        )
