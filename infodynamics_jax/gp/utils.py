# infodynamics_jax/gp/utils.py
"""
Numerical stability utilities for GP computations.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def safe_cholesky(
    K: jnp.ndarray,
    jitter: float = 1e-6,
    max_jitter: float = 1e-2,
) -> jnp.ndarray:
    """
    Compute Cholesky decomposition with automatic jitter adjustment.
    
    This function attempts Cholesky decomposition with increasing jitter
    values until successful, ensuring numerical stability. JIT-compatible.
    
    Strategy: Try multiple jitter values in parallel, then select the first valid result.
    If all fail, use max_jitter as last resort.
    
    Args:
        K: Symmetric matrix to decompose (M, M)
        jitter: Initial jitter value (default: 1e-6)
        max_jitter: Maximum jitter value to try (default: 1e-2)
    
    Returns:
        L: Lower triangular Cholesky factor (M, M)
    """
    # Ensure symmetry
    K = 0.5 * (K + K.T)
    
    # Generate jitter values to try: [jitter, jitter*10, jitter*100, ..., max_jitter]
    n_attempts = 8
    jitter_scales = jnp.array([10.0 ** i for i in range(n_attempts)])
    jitter_values = jnp.minimum(jitter * jitter_scales, max_jitter)
    
    def try_cholesky(jit_val):
        """Try Cholesky with given jitter value."""
        K_safe = K + jit_val * jnp.eye(K.shape[0], dtype=K.dtype)
        L = jnp.linalg.cholesky(K_safe)
        # Check if valid (no NaN/Inf)
        is_finite = jnp.all(jnp.isfinite(L))
        # Return L and success flag
        return L, is_finite
    
    # Try all jitter values in parallel
    L_results, is_valid = jax.vmap(try_cholesky)(jitter_values)
    
    # Find first valid result using cumulative sum trick
    # valid_mask[i] = True if this is the first valid result
    cumsum_valid = jnp.cumsum(is_valid.astype(jnp.int32))
    valid_mask = (cumsum_valid == 1) & is_valid
    
    # Select first valid L (use nanmax to handle case where no result is valid)
    # We'll use a weighted sum where invalid results have weight 0
    weights = valid_mask.astype(L_results.dtype)[:, None, None]
    L_weighted = L_results * weights
    
    # Sum weighted results (only valid ones contribute)
    L_sum = jnp.sum(L_weighted, axis=0)
    weight_sum = jnp.sum(weights, axis=0)
    
    # Normalize (if no valid result, weight_sum will be 0, and we'll use fallback)
    L = jnp.where(weight_sum > 0, L_sum / weight_sum, jnp.nan)
    
    # Fallback: if no valid result, use max_jitter
    L_fallback = jnp.linalg.cholesky(K + max_jitter * jnp.eye(K.shape[0], dtype=K.dtype))
    L = jnp.where(jnp.all(jnp.isfinite(L)), L, L_fallback)
    
    return L
