# infodynamics_jax/inference/particle/resampling.py
"""
Resampling utilities for particle-based inference methods.

This module provides core resampling operations used by particle-based inference methods:
  - multinomial_resample: Resampling particles based on weights
  - effective_sample_size: Compute ESS to determine when to resample

These functions are used by:
  - AnnealedSMC: β-annealed SMC (thermodynamic path)
  - IBIS: Iterated Batch Importance Sampling (data streaming path)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def multinomial_resample(key, logw, n_particles):
    """
    Multinomial resampling of particles based on log weights.
    
    This is a core SMC operation used to resample particles when ESS drops below threshold.
    
    Args:
        key: PRNG key
        logw: Log weights (P,)
        n_particles: Number of particles
    
    Returns:
        indices: Resampling indices (P,)
    """
    w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
    indices = random.choice(key, n_particles, shape=(n_particles,), p=w, replace=True)
    return indices


def effective_sample_size(logw):
    """
    Compute effective sample size (ESS) from log weights.
    
    ESS is used to determine when to resample particles in SMC methods.
    ESS = 1 / sum(w^2), where w are normalized weights.
    
    Args:
        logw: Log weights (P,)
    
    Returns:
        ess: Effective sample size (scalar)
    """
    w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
    return 1.0 / jnp.sum(w ** 2)
