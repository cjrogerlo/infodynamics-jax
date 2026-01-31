# infodynamics_jax/inference/particle/rejuvenation.py
"""
Rejuvenation kernels for particle-based inference methods.

Rejuvenation is used to refresh particles after resampling in SMC methods.
These kernels target either:
  - Tempered distribution: π_β(φ) ∝ p(φ) p(y|φ)^β (for Annealed SMC)
  - Full posterior: p(φ | y_{1:t}) (for IBIS)

All kernels operate on stacked particles pytree with shape [P, ...].

Non-Conjugate Support:
  Rejuvenation kernels work with both conjugate and non-conjugate likelihoods.
  For non-conjugate likelihoods, the energy_fn should use InertialEnergy with
  appropriate estimators (GH/MC), which internally handle ansatz computation.
  The rejuvenation kernel only needs the energy value and gradient, and doesn't
  care how the energy is computed internally.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.tree_util import tree_map, tree_leaves, tree_flatten




def kinetic_energy(p):
    """Compute kinetic energy: K(p) = 0.5 * ||p||^2"""
    sq_sum = 0.0
    for leaf in tree_leaves(p):
        sq_sum += jnp.sum(leaf ** 2)
    return 0.5 * sq_sum


def hmc_rejuvenate(
    key: jax.random.KeyArray,
    particles: Any,  # pytree stacked [P, ...]
    energy_fn: Callable[[Any], jnp.ndarray],  # function(phi) -> scalar
    step_size: float = 1e-2,
    n_leapfrog: int = 4,
    n_steps: int = 1,
    jit: bool = True,
) -> Any:
    """
    HMC rejuvenation kernel for particle-based methods.
    
    This kernel performs HMC steps on each particle to refresh them after resampling.
    It targets the distribution defined by `energy_fn`.
    
    Args:
        key: PRNG key
        particles: Stacked particles pytree with shape [P, ...]
        energy_fn: Energy function (phi) -> scalar energy
            For non-conjugate likelihoods, this should use InertialEnergy with
            appropriate estimators (GH/MC), which internally handle ansatz computation.
        step_size: HMC step size
        n_leapfrog: Number of leapfrog steps per HMC step
        n_steps: Number of HMC steps per particle
        jit: Whether to use JIT-compiled scan (faster but less flexible)
    
    Returns:
        rejuvenated_particles: pytree stacked [P, ...]
    
    Examples:
        >>> # For Annealed SMC (tempered distribution)
        >>> def energy_fn(phi):
        ...     return beta * energy(phi, *energy_args, **energy_kwargs)
        >>> particles = hmc_rejuvenate(key, particles, energy_fn, step_size=1e-2, n_leapfrog=4)
        
        >>> # For IBIS (full posterior)
        >>> def energy_fn(phi):
        ...     return energy(phi, *energy_args, **energy_kwargs)
        >>> particles = hmc_rejuvenate(key, particles, energy_fn, step_size=1e-2, n_leapfrog=4)
        
        >>> # Non-conjugate likelihood (energy internally uses GH/MC estimators)
        >>> # energy_fn automatically handles ansatz computation via InertialEnergy
        >>> def energy_fn(phi):
        ...     return energy(phi, X, Y, key=key)  # InertialEnergy handles GH/MC internally
        >>> particles = hmc_rejuvenate(key, particles, energy_fn, step_size=1e-2, n_leapfrog=4)
    
    Note:
        This kernel works with both conjugate and non-conjugate likelihoods.
        For non-conjugate, the energy_fn should use InertialEnergy with GH/MC estimators,
        which internally compute E[-log p(y|f,φ)] using ansatz (Gauss-Hermite or Monte Carlo).
        The rejuvenation kernel only needs the energy value and gradient, and doesn't
        care how the energy is computed internally.
    """
    grad_U = jax.grad(energy_fn)

    def leapfrog(q, p, step_size, n_steps):
        """Leapfrog integration for Hamiltonian dynamics."""
        def body_fn(i, val):
            q, p = val
            p = tree_map(lambda p_, g: p_ - 0.5 * step_size * g, p, grad_U(q))
            q = tree_map(lambda q_, p_: q_ + step_size * p_, q, p)
            p = tree_map(lambda p_, g: p_ - 0.5 * step_size * g, p, grad_U(q))
            return (q, p)

        return lax.fori_loop(0, n_steps, body_fn, (q, p))

    def hmc_step(carry, key):
        """Single HMC step."""
        q = carry
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Sample momentum
        p = tree_map(lambda x: random.normal(subkey1, x.shape, dtype=x.dtype), q)
        current_U = energy_fn(q)
        current_K = kinetic_energy(p)

        # Leapfrog integration
        q_new, p_new = leapfrog(q, p, step_size, n_leapfrog)

        # Metropolis-Hastings acceptance
        proposed_U = energy_fn(q_new)
        proposed_K = kinetic_energy(p_new)

        log_accept_ratio = -(proposed_U + proposed_K) + (current_U + current_K)
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        u = random.uniform(subkey2)
        accept = u < accept_prob

        # Use tree_map to select between pytrees based on accept flag
        q_next = tree_map(lambda new, old: jax.lax.select(accept, new, old), q_new, q)
        return q_next, accept_prob

    # Get number of particles from pytree structure
    particles_flat, _ = tree_flatten(particles)
    if len(particles_flat) == 0:
        raise ValueError("particles pytree is empty")
    n_particles = particles_flat[0].shape[0]
    
    # Split keys per particle, then per step (works across JAX key types)
    particle_keys = random.split(key, n_particles)
    keys = jax.vmap(lambda k: random.split(k, n_steps))(particle_keys)
    
    # vmap over particles
    def particle_rejuvenate(q, keys):
        """Apply n_steps HMC steps to a single particle."""
        def step(carry, key):
            return hmc_step(carry, key)
        
        if jit:
            q, probs = lax.scan(step, q, keys)
        else:
            probs_list = []
            for k in keys:
                q, p = hmc_step(q, k)
                probs_list.append(p)
            probs = jnp.array(probs_list)
        return q, probs

    particles, accept_probs = jax.vmap(particle_rejuvenate)(particles, keys)
    return particles, accept_probs


def mala_rejuvenate(
    key: jax.random.KeyArray,
    particles: Any,  # pytree stacked [P, ...]
    energy_fn: Callable[[Any], jnp.ndarray],  # function(phi) -> scalar
    step_size: float = 1e-2,
    n_steps: int = 1,
    jit: bool = True,
    drift_transform: Callable[[Any, Any, float], Any] = None,  # optional: g -> (I - C) g
    beta: float = 1.0,
) -> Any:
    """
    MALA (Metropolis-Adjusted Langevin Algorithm) rejuvenation kernel.
    
    MALA uses Langevin dynamics to propose moves, making it simpler than HMC
    (no momentum, single step) but still more efficient than random-walk
    Metropolis-Hastings by using gradient information.
    
    Args:
        key: PRNG key
        particles: Stacked particles pytree with shape [P, ...]
        energy_fn: Energy function (phi) -> scalar energy
            For non-conjugate likelihoods, this should use InertialEnergy with
            appropriate estimators (GH/MC), which internally handle ansatz computation.
        step_size: MALA step size
        n_steps: Number of MALA steps per particle
        jit: Whether to use JIT-compiled scan (faster but less flexible)
        drift_transform: Optional callable (q, grad_q, beta) -> modified_grad.
            Use this to inject solenoidal/circulation terms (Regime-1) without
            changing the energy. Defaults to identity (no curl). For drift = -g + Cg,
            pass modified_grad = g - Cg.
        beta: Scalar annealing parameter, forwarded to drift_transform.
    
    Returns:
        rejuvenated_particles: pytree stacked [P, ...]
    
    Note:
        MALA is simpler than HMC (no momentum, no leapfrog) but may require
        more steps to achieve similar mixing. It's useful when HMC is too expensive
        or when step size tuning is difficult.
    """
    grad_U = jax.grad(energy_fn)

    if drift_transform is None:
        def drift_transform(q, g, beta):
            return g

    def compute_proposal_log_prob(q_from, q_to, grad_drift_from, step_size):
        """Compute log probability of proposing q_to from q_from."""
        # Mean of proposal distribution: q_mean = q_from - (step_size/2) * grad_drift_from
        q_mean = tree_map(lambda q, g: q - 0.5 * step_size * g, q_from, grad_drift_from)
        # Difference: q_to - q_mean
        q_diff = tree_map(lambda a, b: a - b, q_to, q_mean)
        # Log probability: -0.5 * ||q_to - q_mean||^2 / step_size
        q_diff_flat, _ = tree_flatten(q_diff)
        log_prob = -0.5 * sum(jnp.sum(diff ** 2) for diff in q_diff_flat) / step_size
        return log_prob

    def mala_step(carry, key):
        """Single MALA step."""
        q = carry
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Compute gradient at current position
        grad_U_current = grad_U(q)
        grad_drift_current = drift_transform(q, grad_U_current, beta)
        current_U = energy_fn(q)
        
        # Propose new state using Langevin dynamics:
        # q_proposed = q_current - (step_size / 2) * grad + sqrt(step_size) * noise
        noise = tree_map(lambda x: random.normal(subkey1, x.shape, dtype=x.dtype), q)
        q_proposed = tree_map(
            lambda q, g, n: q - 0.5 * step_size * g + jnp.sqrt(step_size) * n,
            q, grad_drift_current, noise
        )
        
        # Compute energy and gradient at proposed state
        proposed_U = energy_fn(q_proposed)
        grad_U_proposed = grad_U(q_proposed)
        grad_drift_proposed = drift_transform(q_proposed, grad_U_proposed, beta)
        
        # Compute proposal probabilities (symmetric Gaussian proposals)
        log_q_forward = compute_proposal_log_prob(q, q_proposed, grad_drift_current, step_size)
        log_q_backward = compute_proposal_log_prob(q_proposed, q, grad_drift_proposed, step_size)
        
        # Metropolis acceptance ratio
        log_accept_ratio = (
            -proposed_U + current_U  # Energy difference
            + log_q_backward - log_q_forward  # Proposal probability ratio
        )
        
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        u = random.uniform(subkey2)
        accept = u < accept_prob
        
        q_next = tree_map(lambda new, old: jax.lax.select(accept, new, old), q_proposed, q)
        return q_next, accept_prob

    # Get number of particles from pytree structure
    particles_flat, _ = tree_flatten(particles)
    if len(particles_flat) == 0:
        raise ValueError("particles pytree is empty")
    n_particles = particles_flat[0].shape[0]
    
    # Split keys per particle, then per step (works across JAX key types)
    particle_keys = random.split(key, n_particles)
    keys = jax.vmap(lambda k: random.split(k, n_steps))(particle_keys)
    
    # vmap over particles
    def particle_rejuvenate(q, keys):
        """Apply n_steps MALA steps to a single particle."""
        def step(carry, key):
            return mala_step(carry, key)
        
        if jit:
            q, probs = lax.scan(step, q, keys)
        else:
            probs_list = []
            for k in keys:
                q, p = mala_step(q, k)
                probs_list.append(p)
            probs = jnp.array(probs_list)
        return q, probs

    particles, accept_probs = jax.vmap(particle_rejuvenate)(particles, keys)
    return particles, accept_probs


def _leapfrog_step_nuts(q, p, grad_U_val, step_size, grad_U_fn):
    """Single leapfrog step for NUTS."""
    # Half step in momentum
    p_half = tree_map(lambda p_, g: p_ - 0.5 * step_size * g, p, grad_U_val)
    # Full step in position
    q_new = tree_map(lambda q_, p_: q_ + step_size * p_, q, p_half)
    # Compute new gradient
    grad_U_new = grad_U_fn(q_new)
    # Half step in momentum
    p_new = tree_map(lambda p_, g: p_ - 0.5 * step_size * g, p_half, grad_U_new)
    return q_new, p_new, grad_U_new


def nuts_rejuvenate(
    key: jax.random.KeyArray,
    particles: Any,  # pytree stacked [P, ...]
    energy_fn: Callable[[Any], jnp.ndarray],  # function(phi) -> scalar
    step_size: float = 1e-2,
    max_tree_depth: int = 10,
    delta_max: float = 1000.0,
    n_steps: int = 1,
    jit: bool = True,
) -> Any:
    """
    NUTS (No-U-Turn Sampler) rejuvenation kernel.
    
    NUTS automatically selects the number of leapfrog steps by building a binary
    tree of trajectories and stopping when a U-turn is detected. This eliminates
    the need to manually tune the number of steps.
    
    Args:
        key: PRNG key
        particles: Stacked particles pytree with shape [P, ...]
        energy_fn: Energy function (phi) -> scalar energy
            For non-conjugate likelihoods, this should use InertialEnergy with
            appropriate estimators (GH/MC), which internally handle ansatz computation.
        step_size: NUTS step size
        max_tree_depth: Maximum depth of the binary tree
        delta_max: Maximum energy change for slice condition
        n_steps: Number of NUTS steps per particle (each step builds a tree)
        jit: Whether to use JIT-compiled scan (faster but less flexible)
    
    Returns:
        rejuvenated_particles: pytree stacked [P, ...]
    
    Note:
        NUTS is more sophisticated than HMC but also more expensive per step.
        It's useful when step size tuning is difficult or when you want automatic
        adaptation of the number of leapfrog steps.
    """
    grad_U = jax.grad(energy_fn)

    def _compute_dot_product(q1, p1, q2):
        """Compute (q1 - q2) · p1 (dot product for U-turn check)."""
        q_diff_flat, _ = tree_flatten(tree_map(lambda a, b: a - b, q1, q2))
        p1_flat, _ = tree_flatten(p1)
        dot = sum(jnp.sum(diff * p) for diff, p in zip(q_diff_flat, p1_flat))
        return dot

    def nuts_step(carry, key):
        """Single NUTS step using iterative tree building."""
        q = carry
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Sample momentum
        p = tree_map(lambda x: random.normal(subkey1, x.shape, dtype=x.dtype), q)
        current_U = energy_fn(q)
        current_K = kinetic_energy(p)
        
        # Compute initial gradient
        grad_U_current = grad_U(q)
        
        # Compute initial Hamiltonian
        H0 = current_U + current_K
        
        # Sample slice variable: u ~ Uniform(0, exp(-H0))
        u = random.uniform(subkey2) * jnp.exp(-H0)
        
        # Initialize tree boundaries
        q_minus = q
        p_minus = p
        grad_U_minus = grad_U_current
        q_plus = q
        p_plus = p
        grad_U_plus = grad_U_current
        
        # Initialize proposal
        q_proposed = q
        p_proposed = p
        n_proposed = 1
        s = 1
        alpha = 0.0
        n_alpha = 0
        
        # Build tree iteratively
        j = 0
        while j < max_tree_depth and s == 1:
            # Sample direction v ~ Uniform({-1, 1})
            key, subkey = random.split(key)
            v = random.choice(subkey, jnp.array([-1, 1]))
            
            if v == -1:
                # Build left subtree (backward)
                q_minus, p_minus, grad_U_minus = _leapfrog_step_nuts(
                    q_minus, p_minus, grad_U_minus, -step_size, grad_U
                )
            else:
                # Build right subtree (forward)
                q_plus, p_plus, grad_U_plus = _leapfrog_step_nuts(
                    q_plus, p_plus, grad_U_plus, step_size, grad_U
                )
            
            # Compute Hamiltonian for new states
            H_minus = energy_fn(q_minus) + kinetic_energy(p_minus)
            H_plus = energy_fn(q_plus) + kinetic_energy(p_plus)
            
            # Check slice condition
            s_minus = (u <= jnp.exp(-H_minus + H0)).astype(jnp.int32) * (
                u < delta_max * jnp.exp(-H_minus + H0)
            ).astype(jnp.int32)
            s_plus = (u <= jnp.exp(-H_plus + H0)).astype(jnp.int32) * (
                u < delta_max * jnp.exp(-H_plus + H0)
            ).astype(jnp.int32)
            
            # Check for U-turn
            dot_minus = _compute_dot_product(q_plus, p_minus, q_minus)
            dot_plus = _compute_dot_product(q_plus, p_plus, q_minus)
            u_turn = (dot_minus <= 0) | (dot_plus <= 0)
            
            s = s_minus * s_plus * (1 - u_turn.astype(jnp.int32))
            
            if s == 1:
                # Accept proposal with probability min(1, n_proposed / n_new)
                # For simplicity, we use uniform sampling from the tree
                key, subkey = random.split(key)
                accept_new = random.uniform(subkey) < (1.0 / n_proposed)
                q_proposed = jax.lax.select(accept_new, q_plus if v == 1 else q_minus, q_proposed)
                n_proposed += 1
                alpha += jnp.minimum(1.0, jnp.exp(-H_plus + H0) if v == 1 else jnp.exp(-H_minus + H0))
                n_alpha += 1
            
            j += 1
        
        # Final acceptance (simplified: always accept if tree was built)
        q_next = jax.lax.select(s == 1, q_proposed, q)
        return q_next, alpha / max(n_alpha, 1)

    # Get number of particles from pytree structure
    particles_flat, _ = tree_flatten(particles)
    if len(particles_flat) == 0:
        raise ValueError("particles pytree is empty")
    n_particles = particles_flat[0].shape[0]
    
    # Split keys per particle, then per step (works across JAX key types)
    particle_keys = random.split(key, n_particles)
    keys = jax.vmap(lambda k: random.split(k, n_steps))(particle_keys)
    
    # vmap over particles
    def particle_rejuvenate(q, keys):
        """Apply n_steps NUTS steps to a single particle."""
        def step(carry, key):
            return nuts_step(carry, key)
        
        if jit:
            q, probs = lax.scan(step, q, keys)
        else:
            probs_list = []
            for k in keys:
                q, p = nuts_step(q, k)
                probs_list.append(p)
            probs = jnp.array(probs_list)
        return q, probs

    particles, accept_probs = jax.vmap(particle_rejuvenate)(particles, keys)
    return particles, accept_probs


__all__ = [
    "kinetic_energy",
    "hmc_rejuvenate",
    "mala_rejuvenate",
    "nuts_rejuvenate",
]
