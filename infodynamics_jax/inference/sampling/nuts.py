# infodynamics_jax/inference/sampling/nuts.py
"""
No-U-Turn Sampler (NUTS).

NUTS is an extension of HMC that automatically selects the number of leapfrog steps
by building a binary tree of trajectories and stopping when a U-turn is detected.
This eliminates the need to manually tune the number of steps.
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
class NUTSCFG:
    """Configuration for NUTS kernel."""
    step_size: float = 1e-2
    max_tree_depth: int = 10
    delta_max: float = 1000.0
    n_samples: int = 256
    n_warmup: int = 256
    jit: bool = True


@dataclass
class NUTSRun:
    """NUTS run results."""
    samples: Any  # pytree stacked on leading axis with shape [n_samples, ...]
    accept_rate: float
    energy_trace: jnp.ndarray  # shape [n_samples]
    tree_depth_trace: jnp.ndarray  # shape [n_samples]
    n_leapfrog_trace: jnp.ndarray  # shape [n_samples]


def kinetic_energy(p):
    """Compute kinetic energy: K(p) = 0.5 * ||p||^2"""
    sq_sum = 0.0
    for leaf in tree_leaves(p):
        sq_sum += jnp.sum(leaf ** 2)
    return 0.5 * sq_sum


def leapfrog_step(q, p, grad_U, step_size, energy, energy_args, energy_kwargs):
    """Single leapfrog step."""
    def energy_fn(q_):
        return energy(q_, *energy_args, **energy_kwargs)
    
    # Half step in momentum
    p_half = tree_map(lambda p_, g: p_ - 0.5 * step_size * g, p, grad_U)
    
    # Full step in position
    q_new = tree_map(lambda q_, p_: q_ + step_size * p_, q, p_half)
    
    # Compute new gradient
    grad_U_new = jax.grad(energy_fn)(q_new)
    
    # Half step in momentum
    p_new = tree_map(lambda p_, g: p_ - 0.5 * step_size * g, p_half, grad_U_new)
    
    return q_new, p_new, grad_U_new


class NUTS(InferenceMethod):
    """
    No-U-Turn Sampler (NUTS).
    
    NUTS automatically selects the number of leapfrog steps by building a binary
    tree of trajectories and stopping when a U-turn is detected. This eliminates
    the need to manually tune the number of steps.
    """
    
    def __init__(self, cfg: NUTSCFG = NUTSCFG()):
        self.cfg = cfg

    def _compute_dot_product(self, q1, p1, q2):
        """Compute (q1 - q2) · p1 (dot product for U-turn check)."""
        q_diff_flat, _ = jax.tree_flatten(tree_map(lambda a, b: a - b, q1, q2))
        p1_flat, _ = jax.tree_flatten(p1)
        dot = sum(jnp.sum(diff * p) for diff, p in zip(q_diff_flat, p1_flat))
        return dot

    def _nuts_step(self, state, energy_info):
        """Single NUTS step using iterative tree building.
        
        Args:
            state: (q_current, energy_current, key)
            energy_info: (energy, energy_args, energy_kwargs) tuple
        """
        q_current, energy_current, key = state
        energy, energy_args, energy_kwargs = energy_info
        cfg = self.cfg
        
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Sample momentum
        def sample_momentum(q):
            return tree_map(lambda x: random.normal(subkey1, x.shape, dtype=x.dtype), q)
        
        p_current = sample_momentum(q_current)
        
        # Compute initial gradient
        def energy_fn(q_):
            return energy(q_, *energy_args, **energy_kwargs)
        grad_U_current = jax.grad(energy_fn)(q_current)
        
        # Compute initial Hamiltonian
        H0 = energy_current + kinetic_energy(p_current)
        
        # Sample slice variable: u ~ Uniform(0, exp(-H0))
        u = random.uniform(subkey2) * jnp.exp(-H0)
        
        # Initialize tree boundaries
        q_minus = q_current
        p_minus = p_current
        grad_U_minus = grad_U_current
        q_plus = q_current
        p_plus = p_current
        grad_U_plus = grad_U_current
        
        # Initialize proposal
        q_proposed = q_current
        p_proposed = p_current
        n_proposed = 1
        s = 1
        alpha = 0.0
        n_alpha = 0
        
        # Build tree iteratively
        j = 0
        while j < cfg.max_tree_depth and s == 1:
            # Sample direction v ~ Uniform({-1, 1})
            key, subkey = random.split(key)
            v = random.choice(subkey, jnp.array([-1, 1]))
            
            if v == -1:
                # Build left subtree (backward)
                q_minus, p_minus, grad_U_minus = leapfrog_step(
                    q_minus, p_minus, grad_U_minus, -cfg.step_size,
                    energy, energy_args, energy_kwargs
                )
            else:
                # Build right subtree (forward)
                q_plus, p_plus, grad_U_plus = leapfrog_step(
                    q_plus, p_plus, grad_U_plus, cfg.step_size,
                    energy, energy_args, energy_kwargs
                )
            
            # Compute Hamiltonian for new states
            H_minus = energy(q_minus, *energy_args, **energy_kwargs) + kinetic_energy(p_minus)
            H_plus = energy(q_plus, *energy_args, **energy_kwargs) + kinetic_energy(p_plus)
            
            # Check slice condition
            s_minus = (u <= jnp.exp(-H_minus + H0)).astype(jnp.int32) * (u < self.cfg.delta_max * jnp.exp(-H_minus + H0)).astype(jnp.int32)
            s_plus = (u <= jnp.exp(-H_plus + H0)).astype(jnp.int32) * (u < self.cfg.delta_max * jnp.exp(-H_plus + H0)).astype(jnp.int32)
            
            # Check for U-turn
            # U-turn: (q_plus - q_minus) · p_minus <= 0 or (q_plus - q_minus) · p_plus <= 0
            dot_minus = self._compute_dot_product(q_plus, p_minus, q_minus)
            dot_plus = self._compute_dot_product(q_plus, p_plus, q_minus)
            
            s_uturn = (dot_minus > 0) * (dot_plus > 0)
            s = s_minus * s_plus * s_uturn
            
            # Update proposal with probability n_new / (n_proposed + n_new)
            if s == 1:
                # Accept new state with probability based on slice condition
                key, subkey = random.split(key)
                if v == -1:
                    n_new = s_minus
                    q_new, p_new = q_minus, p_minus
                else:
                    n_new = s_plus
                    q_new, p_new = q_plus, p_plus
                
                accept_proposal = random.uniform(subkey) < (n_new.astype(jnp.float32) / (n_proposed + n_new + 1e-10))
                q_proposed = jax.lax.select(accept_proposal, q_new, q_proposed)
                p_proposed = jax.lax.select(accept_proposal, p_new, p_proposed)
                n_proposed = n_proposed + n_new
                
                # Update alpha (for diagnostics)
                if v == -1:
                    alpha_new = jnp.minimum(1.0, jnp.exp(-H_minus + H0))
                else:
                    alpha_new = jnp.minimum(1.0, jnp.exp(-H_plus + H0))
                alpha = alpha + alpha_new
                n_alpha = n_alpha + 1
            
            j += 1
        
        # Accept or reject final proposal
        if n_proposed > 0:
            H_proposed = energy(q_proposed, *energy_args, **energy_kwargs) + kinetic_energy(p_proposed)
            log_accept_ratio = -H_proposed + H0
            accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
            
            key, subkey_accept = random.split(key)
            u_accept = random.uniform(subkey_accept)
            accept = u_accept < accept_prob
            
            q_next = jax.lax.select(accept, q_proposed, q_current)
            energy_next = jax.lax.select(
                accept,
                energy(q_proposed, *energy_args, **energy_kwargs),
                energy_current
            )
        else:
            # No valid proposal, stay at current state
            accept_prob = 0.0
            q_next = q_current
            energy_next = energy_current
        
        tree_depth = j
        n_leapfrog = 2 ** j - 1  # Approximate number of leapfrog steps
        
        return (q_next, energy_next, key), (accept_prob, tree_depth, n_leapfrog)

    def run(self, energy: EnergyTerm, phi_init, *, key, energy_args=(), energy_kwargs=None) -> NUTSRun:
        """
        Run NUTS sampling.
        
        Args:
            energy: Energy term to sample from
            phi_init: Initial parameter state
            key: PRNG key
            energy_args: Additional arguments for energy
            energy_kwargs: Additional keyword arguments for energy
        
        Returns:
            NUTSRun with samples, accept_rate, energy_trace, tree_depth_trace, and n_leapfrog_trace
        """
        if energy_kwargs is None:
            energy_kwargs = {}

        cfg = self.cfg
        n_warmup = cfg.n_warmup
        n_samples = cfg.n_samples
        jit = cfg.jit

        # Pack energy info for passing to _nuts_step
        energy_info = (energy, energy_args, energy_kwargs)

        # Initial energy
        energy_init = energy(phi_init, *energy_args, **energy_kwargs)
        state = (phi_init, energy_init, key)

        # Warmup
        if jit:
            def warmup_body(i, state):
                state, _ = self._nuts_step(state, energy_info)
                return state
            state = lax.fori_loop(0, n_warmup, warmup_body, state)
        else:
            for _ in range(n_warmup):
                state, _ = self._nuts_step(state, energy_info)

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
            tree_depth_trace = jnp.zeros(n_samples, dtype=jnp.int32)
            n_leapfrog_trace = jnp.zeros(n_samples, dtype=jnp.int32)

            def body_fun(i, val):
                state, samples_stack, energy_trace, accept_probs, tree_depth_trace, n_leapfrog_trace = val
                state, (accept_prob, tree_depth, n_leapfrog) = self._nuts_step(state, energy_info)
                q, energy_val, _ = state
                q_flat, _ = jax.tree_flatten(q)
                samples_stack = [arr.at[i].set(x) for arr, x in zip(samples_stack, q_flat)]
                energy_trace = energy_trace.at[i].set(energy_val)
                accept_probs = accept_probs.at[i].set(accept_prob)
                tree_depth_trace = tree_depth_trace.at[i].set(tree_depth)
                n_leapfrog_trace = n_leapfrog_trace.at[i].set(n_leapfrog)
                return (state, samples_stack, energy_trace, accept_probs, tree_depth_trace, n_leapfrog_trace)

            state, samples_stack, energy_trace, accept_probs, tree_depth_trace, n_leapfrog_trace = lax.fori_loop(
                0, n_samples, body_fun,
                (state, samples_stack, energy_trace, accept_probs, tree_depth_trace, n_leapfrog_trace)
            )
            samples = jax.tree_unflatten(q_struct, samples_stack)
        else:
            samples = []
            energy_trace = jnp.zeros(n_samples)
            accept_probs = jnp.zeros(n_samples)
            tree_depth_trace = jnp.zeros(n_samples, dtype=jnp.int32)
            n_leapfrog_trace = jnp.zeros(n_samples, dtype=jnp.int32)
            for i in range(n_samples):
                state, (accept_prob, tree_depth, n_leapfrog) = self._nuts_step(state, energy_info)
                q, energy_val, _ = state
                samples.append(q)
                energy_trace = energy_trace.at[i].set(energy_val)
                accept_probs = accept_probs.at[i].set(accept_prob)
                tree_depth_trace = tree_depth_trace.at[i].set(tree_depth)
                n_leapfrog_trace = n_leapfrog_trace.at[i].set(n_leapfrog)
            # Stack samples
            samples = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *samples)

        accept_rate = jnp.mean(accept_probs)
        return NUTSRun(
            samples=samples,
            accept_rate=accept_rate,
            energy_trace=energy_trace,
            tree_depth_trace=tree_depth_trace,
            n_leapfrog_trace=n_leapfrog_trace
        )
