# infodynamics_jax/inference/sampling/mala.py
"""
Metropolis-Adjusted Langevin Algorithm (MALA).

MALA is a gradient-based MCMC method that uses Langevin dynamics to propose moves.
It is simpler than HMC (no momentum, single step) but more efficient than random-walk
Metropolis-Hastings by using gradient information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.tree_util import tree_map

from ...energy.base import EnergyTerm
from ..base import InferenceMethod


@dataclass(frozen=True)
class MALACFG:
    """Configuration for MALA kernel."""
    step_size: float = 1e-2
    n_samples: int = 256
    n_warmup: int = 256
    jit: bool = True


@dataclass
class MALARun:
    """MALA run results."""
    samples: Any  # pytree stacked on leading axis with shape [n_samples, ...]
    accept_rate: float
    energy_trace: jnp.ndarray  # shape [n_samples]


class MALA(InferenceMethod):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA).
    
    MALA uses Langevin dynamics to propose moves:
        q_proposed = q_current - (step_size / 2) * grad_U(q_current) + sqrt(step_size) * noise
    
    where noise ~ N(0, I). This uses gradient information to guide proposals
    towards higher probability regions, making it more efficient than random-walk
    Metropolis-Hastings.
    """
    
    def __init__(self, cfg: MALACFG = MALACFG()):
        self.cfg = cfg

    def _mala_step(self, state, energy_info):
        """Single MALA step.
        
        Args:
            state: (q_current, energy_current, key)
            energy_info: (energy, energy_args, energy_kwargs) tuple
        """
        q_current, energy_current, key = state
        energy, energy_args, energy_kwargs = energy_info
        cfg = self.cfg
        
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Compute gradient of energy (potential)
        def energy_fn(q_):
            return energy(q_, *energy_args, **energy_kwargs)
        
        grad_U = jax.grad(energy_fn)(q_current)
        
        # Propose new state using Langevin dynamics:
        # q_proposed = q_current - (step_size / 2) * grad_U + sqrt(step_size) * noise
        def sample_noise(q):
            return tree_map(lambda x: random.normal(subkey1, x.shape, dtype=x.dtype), q)
        
        noise = sample_noise(q_current)
        q_proposed = tree_map(
            lambda q, g, n: q - 0.5 * cfg.step_size * g + jnp.sqrt(cfg.step_size) * n,
            q_current, grad_U, noise
        )
        
        # Compute energy and gradient at proposed state
        energy_proposed = energy(q_proposed, *energy_args, **energy_kwargs)
        grad_U_proposed = jax.grad(energy_fn)(q_proposed)
        
        # Compute proposal probabilities (symmetric Gaussian proposals)
        # Forward: q_proposed | q_current ~ N(q_current - (step_size/2)*grad_U, step_size*I)
        # Backward: q_current | q_proposed ~ N(q_proposed - (step_size/2)*grad_U_proposed, step_size*I)
        
        def compute_proposal_log_prob(q_from, q_to, grad_U_from):
            """Compute log probability of proposing q_to from q_from."""
            # Mean of proposal distribution
            q_mean = tree_map(lambda q, g: q - 0.5 * cfg.step_size * g, q_from, grad_U_from)
            # Difference
            q_diff_flat, _ = jax.tree_flatten(tree_map(lambda a, b: a - b, q_to, q_mean))
            # Log probability: -0.5 * ||diff||^2 / step_size - 0.5 * log(2*pi*step_size) * n_dims
            log_prob = -0.5 * sum(jnp.sum(diff ** 2) for diff in q_diff_flat) / cfg.step_size
            # Normalization constant (can be omitted as it cancels in ratio)
            return log_prob
        
        log_q_forward = compute_proposal_log_prob(q_current, q_proposed, grad_U)
        log_q_backward = compute_proposal_log_prob(q_proposed, q_current, grad_U_proposed)
        
        # Metropolis acceptance ratio
        log_accept_ratio = (
            -energy_proposed + energy_current  # Energy difference
            + log_q_backward - log_q_forward   # Proposal probability ratio
        )
        
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        
        # Accept or reject
        u = random.uniform(subkey2)
        accept = u < accept_prob
        
        q_next = jax.lax.select(accept, q_proposed, q_current)
        energy_next = jax.lax.select(accept, energy_proposed, energy_current)
        
        return (q_next, energy_next, key), accept_prob

    def run(self, energy: EnergyTerm, phi_init, *, key, energy_args=(), energy_kwargs=None) -> MALARun:
        """
        Run MALA sampling.
        
        Args:
            energy: Energy term to sample from
            phi_init: Initial parameter state
            key: PRNG key
            energy_args: Additional arguments for energy
            energy_kwargs: Additional keyword arguments for energy
        
        Returns:
            MALARun with samples, accept_rate, and energy_trace
        """
        if energy_kwargs is None:
            energy_kwargs = {}

        cfg = self.cfg
        n_warmup = cfg.n_warmup
        n_samples = cfg.n_samples
        jit = cfg.jit

        # Pack energy info for passing to _mala_step
        energy_info = (energy, energy_args, energy_kwargs)

        # Initial energy
        energy_init = energy(phi_init, *energy_args, **energy_kwargs)
        state = (phi_init, energy_init, key)

        # Warmup
        if jit:
            def warmup_body(i, state):
                state, _ = self._mala_step(state, energy_info)
                return state
            state = lax.fori_loop(0, n_warmup, warmup_body, state)
        else:
            for _ in range(n_warmup):
                state, _ = self._mala_step(state, energy_info)

        # Sampling
        if jit:
            # Preallocate arrays for JIT
            q_flat, q_struct = jax.tree_flatten(state[0])
            shapes = [x.shape for x in q_flat]
            dtypes = [x.dtype for x in q_flat]
            samples_stack = [
                jnp.zeros((n_samples,) + shape, dtype=dtype) 
                for shape, dtype in zip(shapes, dtypes)
            ]
            energy_trace = jnp.zeros(n_samples)
            accept_probs = jnp.zeros(n_samples)

            def body_fun(i, val):
                state, samples_stack, energy_trace, accept_probs = val
                state, accept_prob = self._mala_step(state, energy_info)
                q, energy_val, _ = state
                q_flat, _ = jax.tree_flatten(q)
                samples_stack = [arr.at[i].set(x) for arr, x in zip(samples_stack, q_flat)]
                energy_trace = energy_trace.at[i].set(energy_val)
                accept_probs = accept_probs.at[i].set(accept_prob)
                return (state, samples_stack, energy_trace, accept_probs)

            state, samples_stack, energy_trace, accept_probs = lax.fori_loop(
                0, n_samples, body_fun, (state, samples_stack, energy_trace, accept_probs)
            )
            samples = jax.tree_unflatten(q_struct, samples_stack)
        else:
            samples = []
            energy_trace = jnp.zeros(n_samples)
            accept_probs = jnp.zeros(n_samples)
            for i in range(n_samples):
                state, accept_prob = self._mala_step(state, energy_info)
                q, energy_val, _ = state
                samples.append(q)
                energy_trace = energy_trace.at[i].set(energy_val)
                accept_probs = accept_probs.at[i].set(accept_prob)
            # Stack samples
            samples = jax.tree_map(lambda *xs: jnp.stack(xs), *samples)

        accept_rate = jnp.mean(accept_probs)
        return MALARun(samples=samples, accept_rate=accept_rate, energy_trace=energy_trace)
