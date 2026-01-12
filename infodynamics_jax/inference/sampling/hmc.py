# infodynamics_jax/inference/sampling/hmc.py
"""
Hamiltonian Monte Carlo (HMC) kernel.

HMC is a gradient-based MCMC method that uses Hamiltonian dynamics
to propose moves in parameter space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.tree_util import tree_map, tree_leaves

from ...energy.base import EnergyTerm
from ..base import InferenceMethod


@dataclass(frozen=True)
class HMCCFG:
    """Configuration for HMC kernel."""
    step_size: float = 1e-2
    n_leapfrog: int = 8
    n_samples: int = 256
    n_warmup: int = 256
    jit: bool = True


@dataclass
class HMCRun:
    """HMC run results."""
    samples: Any  # pytree stacked on leading axis with shape [n_samples, ...]
    accept_rate: float
    energy_trace: jnp.ndarray  # shape [n_samples]


def kinetic_energy(p):
    """Compute kinetic energy: K(p) = 0.5 * ||p||^2"""
    sq_sum = 0.0
    for leaf in tree_leaves(p):
        sq_sum += jnp.sum(leaf ** 2)
    return 0.5 * sq_sum


class HMC(InferenceMethod):
    """
    Hamiltonian Monte Carlo kernel.
    
    HMC uses Hamiltonian dynamics to propose moves, leading to more efficient
    exploration of the energy landscape compared to random-walk proposals.
    """
    
    def __init__(self, cfg: HMCCFG = HMCCFG()):
        self.cfg = cfg

    def _leapfrog(self, q, p, grad_U, step_size, n_steps, energy, energy_args, energy_kwargs):
        """Perform leapfrog integration for Hamiltonian dynamics."""
        def body_fn(i, val):
            q, p, grad_U = val
            p_half = tree_map(lambda p_, g: p_ - 0.5 * step_size * g, p, grad_U)
            q_new = tree_map(lambda q_, p_: q_ + step_size * p_, q, p_half)
            # Compute new grad_U at q_new
            def energy_fn(q_):
                return energy(q_, *energy_args, **energy_kwargs)
            grad_U_new = jax.grad(energy_fn)(q_new)
            p_new = tree_map(lambda p_, g: p_ - 0.5 * step_size * g, p_half, grad_U_new)
            return (q_new, p_new, grad_U_new)
        return lax.fori_loop(0, n_steps, body_fn, (q, p, grad_U))

    def _single_step(self, state, energy_info):
        """Single HMC step.
        
        Args:
            state: (q_current, energy_current, key)
            energy_info: (energy, energy_args, energy_kwargs) tuple
        """
        q_current, energy_current, key = state
        energy, energy_args, energy_kwargs = energy_info
        cfg = self.cfg
        step_size = cfg.step_size
        n_leapfrog = cfg.n_leapfrog

        key, subkey = random.split(key)
        # Sample momentum
        def sample_momentum(q):
            return tree_map(lambda x: random.normal(subkey, x.shape, dtype=x.dtype), q)

        p_current = sample_momentum(q_current)
        # Compute grad_U at current position
        grad_U_current = jax.grad(lambda q: energy(q, *energy_args, **energy_kwargs))(q_current)

        # Leapfrog integration
        q_proposed, p_proposed, grad_U_proposed = self._leapfrog(
            q_current, p_current, grad_U_current, step_size, n_leapfrog, 
            energy, energy_args, energy_kwargs
        )

        # Compute energies
        U_current = energy_current
        K_current = kinetic_energy(p_current)
        U_proposed = energy(q_proposed, *energy_args, **energy_kwargs)
        K_proposed = kinetic_energy(p_proposed)

        # Metropolis acceptance
        log_accept_ratio = -(U_proposed + K_proposed) + (U_current + K_current)
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        key, subkey_accept = random.split(key)
        u = random.uniform(subkey_accept)
        accept = u < accept_prob

        q_next = jax.lax.select(accept, q_proposed, q_current)
        energy_next = jax.lax.select(accept, U_proposed, U_current)

        return (q_next, energy_next, key), accept_prob

    def run(self, energy: EnergyTerm, phi_init, *, key, energy_args=(), energy_kwargs=None) -> HMCRun:
        """
        Run HMC sampling.
        
        Args:
            energy: Energy term to sample from
            phi_init: Initial parameter state
            key: PRNG key
            energy_args: Additional arguments for energy
            energy_kwargs: Additional keyword arguments for energy
        
        Returns:
            HMCRun with samples, accept_rate, and energy_trace
        """
        if energy_kwargs is None:
            energy_kwargs = {}

        cfg = self.cfg
        n_warmup = cfg.n_warmup
        n_samples = cfg.n_samples
        jit = cfg.jit

        # Pack energy info for passing to _single_step
        energy_info = (energy, energy_args, energy_kwargs)

        # Initial energy
        energy_init = energy(phi_init, *energy_args, **energy_kwargs)
        state = (phi_init, energy_init, key)

        # Warmup
        if jit:
            def warmup_body(i, state):
                state, _ = self._single_step(state, energy_info)
                return state
            state = lax.fori_loop(0, n_warmup, warmup_body, state)
        else:
            for _ in range(n_warmup):
                state, _ = self._single_step(state, energy_info)

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
                state, accept_prob = self._single_step(state, energy_info)
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
                state, accept_prob = self._single_step(state, energy_info)
                q, energy_val, _ = state
                samples.append(q)
                energy_trace = energy_trace.at[i].set(energy_val)
                accept_probs = accept_probs.at[i].set(accept_prob)
            # Stack samples
            samples = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *samples)

        accept_rate = jnp.mean(accept_probs)
        return HMCRun(samples=samples, accept_rate=accept_rate, energy_trace=energy_trace)
