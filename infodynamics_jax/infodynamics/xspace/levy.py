"""
Latent X-space Lévy dynamics.

This module provides array-focused kernels for latent/image spaces.
Noise generation is delegated to inference.noise to avoid duplication.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import random, lax

from ...inference.noise import levy_noise


def levy_rejuvenate(
    key: jax.random.KeyArray,
    particles: Any,  # array [P, D]
    energy_fn: Callable[[Any], jnp.ndarray],  # function(phi) -> scalar
    step_size: float = 1e-2,
    n_steps: int = 8,
    alpha: float = 1.6,
    scale: float = 0.15,
    grad_clip: float = 30.0,
    jit: bool = True,
    drift_transform: Callable[[Any, Any, float], Any] = None,  # optional: modified grad (e.g., g - Cg)
    lam: float = 1.0,
) -> Any:
    """
    Lévy noise rejuvenation kernel for array latents.

    Args:
        key: PRNG key
        particles: Stacked particles with shape [P, D] (array, not pytree)
        energy_fn: Energy function (phi) -> scalar energy
        step_size: Step size dt for gradient descent
        n_steps: Number of Lévy steps per particle
        alpha: Stability parameter for α-stable distribution (0 < α ≤ 2)
        scale: Scale parameter σ for Lévy noise
        grad_clip: Gradient clipping threshold
        jit: Whether to use JIT (currently ignored, always uses lax.scan)
        drift_transform: Optional callable (q, grad_q, lam) -> modified_grad.
            For drift = -g + Cg, pass modified_grad = g - Cg.
        lam: Scalar stage parameter forwarded to drift_transform.
    """
    grad_U = jax.grad(energy_fn)

    def levy_step(carry, key):
        """Single Lévy dynamics step."""
        q = carry
        key, subkey = random.split(key)

        # Compute gradient
        g = grad_U(q)
        g = jnp.nan_to_num(g)
        g = jnp.clip(g, -grad_clip, grad_clip)
        if drift_transform is not None:
            g = drift_transform(q, g, lam)

        noise = levy_noise(
            subkey,
            q.shape,
            alpha=alpha,
            scale=scale,
            dt=step_size,
            dtype=q.dtype,
        )
        noise = jnp.nan_to_num(noise)

        # Update: X' = X - dt * grad + dt^{1/α} * noise
        q_new = q - step_size * g + noise
        q_new = jnp.clip(q_new, -8.0, 8.0)  # Prevent extreme values

        return q_new, jnp.array(1.0)  # Always "accept" in Lévy dynamics

    # Get number of particles
    n_particles = particles.shape[0]

    # Split keys per particle, then per step
    particle_keys = random.split(key, n_particles)
    keys = jax.vmap(lambda k: random.split(k, n_steps))(particle_keys)

    # Apply Lévy dynamics per particle
    def particle_rejuvenate(q, particle_keys):
        """Apply n_steps Lévy steps to a single particle."""
        q, probs = lax.scan(levy_step, q, particle_keys)
        return q, probs

    particles, accept_probs = jax.vmap(particle_rejuvenate)(particles, keys)
    return particles, accept_probs


__all__ = ["levy_rejuvenate"]
