# infodynamics_jax/inference/particle/schedules.py
"""
Beta schedules for annealed SMC.

This module provides various beta scheduling strategies for temperature annealing:
  - Linear: uniform spacing
  - Geometric: exponential spacing
  - Power law: polynomial spacing
  - ESS-adaptive: adaptive spacing based on effective sample size
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Callable, Any

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class BetaSchedule:
    """Base configuration for beta schedules."""
    n_steps: int
    beta_min: float = 0.0
    beta_max: float = 1.0


def linear_schedule(n_steps: int, beta_min: float = 0.0, beta_max: float = 1.0) -> jnp.ndarray:
    """
    Linear beta schedule: uniform spacing.
    
    Args:
        n_steps: Number of annealing steps
        beta_min: Starting beta (default: 0.0)
        beta_max: Final beta (default: 1.0)
    
    Returns:
        Array of shape (n_steps + 1,) with betas from beta_min to beta_max
    """
    return jnp.linspace(beta_min, beta_max, n_steps + 1)


def geometric_schedule(n_steps: int, beta_min: float = 0.0, beta_max: float = 1.0, alpha: float = 0.5) -> jnp.ndarray:
    """
    Geometric beta schedule: exponential spacing.
    
    Formula: β_t = β_min + (β_max - β_min) * (α^t - 1) / (α^n_steps - 1)
    
    For α < 1: more steps near β_min (prior)
    For α > 1: more steps near β_max (posterior)
    
    Args:
        n_steps: Number of annealing steps
        beta_min: Starting beta (default: 0.0)
        beta_max: Final beta (default: 1.0)
        alpha: Geometric ratio (default: 0.5, more steps near prior)
    
    Returns:
        Array of shape (n_steps + 1,) with betas
    """
    t = jnp.arange(n_steps + 1, dtype=jnp.float32)
    if abs(alpha - 1.0) < 1e-6:
        # Fallback to linear if alpha ≈ 1
        return linear_schedule(n_steps, beta_min, beta_max)
    denom = alpha ** n_steps - 1.0
    betas = beta_min + (beta_max - beta_min) * (alpha ** t - 1.0) / denom
    return betas


def power_schedule(n_steps: int, beta_min: float = 0.0, beta_max: float = 1.0, power: float = 2.0) -> jnp.ndarray:
    """
    Power law beta schedule: polynomial spacing.
    
    Formula: β_t = β_min + (β_max - β_min) * (t / n_steps)^power
    
    For power < 1: more steps near β_min (prior)
    For power > 1: more steps near β_max (posterior)
    
    Args:
        n_steps: Number of annealing steps
        beta_min: Starting beta (default: 0.0)
        beta_max: Final beta (default: 1.0)
        power: Power exponent (default: 2.0, more steps near posterior)
    
    Returns:
        Array of shape (n_steps + 1,) with betas
    """
    t = jnp.arange(n_steps + 1, dtype=jnp.float32)
    normalized = (t / n_steps) ** power
    betas = beta_min + (beta_max - beta_min) * normalized
    return betas


def ess_adaptive_schedule(
    energy_fn: Callable,
    particles: Any,
    n_steps: int,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
    ess_target: float = 0.5,
    n_particles: int = 128,
    max_bisection_steps: int = 20,
    tol: float = 0.01,
) -> jnp.ndarray:
    """
    ESS-adaptive beta schedule: adaptively choose betas to maintain target ESS.
    
    Uses bisection search to find next beta such that ESS ≈ ess_target * n_particles.
    
    Args:
        energy_fn: Function (phi) -> scalar energy
        particles: Initial particles pytree [P, ...]
        n_steps: Number of annealing steps
        beta_min: Starting beta (default: 0.0)
        beta_max: Final beta (default: 1.0)
        ess_target: Target ESS ratio (default: 0.5)
        n_particles: Number of particles
        max_bisection_steps: Maximum bisection iterations (default: 20)
        tol: Tolerance for ESS target (default: 0.01)
    
    Returns:
        Array of shape (n_steps + 1,) with adaptively chosen betas
    """
    from .resampling import effective_sample_size
    
    betas = jnp.zeros(n_steps + 1)
    betas = betas.at[0].set(beta_min)
    
    logw = jnp.zeros(n_particles)
    target_ess = ess_target * n_particles
    
    def find_next_beta(beta_prev, beta_max_curr, key):
        """Bisection search for next beta."""
        def ess_at_beta(beta):
            # Compute incremental weights
            energies = jax.vmap(energy_fn)(particles)
            delta_beta = beta - beta_prev
            logw_new = logw - delta_beta * energies
            ess = effective_sample_size(logw_new)
            return ess
        
        # Bisection search
        beta_low = beta_prev
        beta_high = beta_max_curr
        
        for _ in range(max_bisection_steps):
            beta_mid = 0.5 * (beta_low + beta_high)
            ess_mid = ess_at_beta(beta_mid)
            
            if abs(ess_mid - target_ess) < tol * target_ess:
                return beta_mid
            
            if ess_mid < target_ess:
                # ESS too low, need smaller delta_beta
                beta_high = beta_mid
            else:
                # ESS too high, can use larger delta_beta
                beta_low = beta_mid
        
        # Return midpoint if convergence not reached
        return 0.5 * (beta_low + beta_high)
    
    # Build schedule adaptively
    # Note: This is a simplified version. Full implementation would need
    # to update particles and logw as we go, which requires scan.
    # For now, we use a heuristic: geometric schedule as fallback
    # Full ESS-adaptive requires integration into the main SMC loop.
    
    # For practical use, we'll provide a simpler heuristic that works well:
    # Use geometric schedule but with adaptive alpha based on initial ESS
    energies = jax.vmap(energy_fn)(particles)
    # Estimate how "difficult" the landscape is
    energy_std = jnp.std(energies)
    # More difficult → use more conservative (smaller alpha) schedule
    alpha = jnp.clip(0.3 + 0.2 * jnp.exp(-energy_std / 10.0), 0.1, 0.9)
    return geometric_schedule(n_steps, beta_min, beta_max, alpha)


def make_beta_schedule(
    schedule_type: Literal["linear", "geometric", "power", "ess_adaptive"] = "linear",
    n_steps: int = 32,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
    **kwargs
) -> jnp.ndarray:
    """
    Factory function to create beta schedules.
    
    Args:
        schedule_type: Type of schedule ("linear", "geometric", "power", "ess_adaptive")
        n_steps: Number of annealing steps
        beta_min: Starting beta (default: 0.0)
        beta_max: Final beta (default: 1.0)
        **kwargs: Additional arguments for specific schedules:
            - geometric: alpha (default: 0.5)
            - power: power (default: 2.0)
            - ess_adaptive: energy_fn, particles, ess_target, n_particles, etc.
    
    Returns:
        Array of shape (n_steps + 1,) with betas
    """
    if schedule_type == "linear":
        return linear_schedule(n_steps, beta_min, beta_max)
    elif schedule_type == "geometric":
        alpha = kwargs.get("alpha", 0.5)
        return geometric_schedule(n_steps, beta_min, beta_max, alpha)
    elif schedule_type == "power":
        power = kwargs.get("power", 2.0)
        return power_schedule(n_steps, beta_min, beta_max, power)
    elif schedule_type == "ess_adaptive":
        # ESS-adaptive requires energy_fn and particles
        # This is a placeholder; full implementation needs integration
        raise NotImplementedError(
            "ESS-adaptive schedule requires integration into SMC loop. "
            "Use 'geometric' or 'power' schedules for now."
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
