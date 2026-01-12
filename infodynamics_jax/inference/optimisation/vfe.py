# infodynamics_jax/inference/optimisation/vfe.py
"""
Type-II variational objective (VFE surrogate) for optimisation-based inference.

This module provides VFE (Variational Free Energy) objective functions for type-II
maximum likelihood and variational inference methods. VFE and KL terms are
specific to these inference paradigms and do not belong in the gp/ or energy/
layers.

IMPORTANT: This is NOT a model energy. This is an optimisation surrogate
used only in type-II / VGA / SVGP inference contexts. For Bayesian inference
(SMC/MCMC), use energy/inertial.py instead.

The VFE objective uses low-rank computation via matrix determinant lemma and
Woodbury formula, avoiding construction of N×N matrices. Complexity is O(NM²+M³)
instead of O(N³).
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ...gp.sparsify import diag_Q_ff
from ...gp.utils import safe_cholesky


def vfe_objective(phi, X, Y, *, kernel_fn: Callable, residual: str = "fitc", key=None) -> jnp.ndarray:
    """
    Type-II variational objective (Titsias VFE) for Gaussian likelihood.
    
    This computes the VFE objective for type-II ML / variational inference:
        L(φ) = NLL(y; 0, Q + σ²I) + (1/(2σ²)) * tr(K_ff - Q_ff)
    
    where:
    - Q = K_xz @ K_zz^{-1} @ K_xz^T (low-rank, never materialized as N×N)
    - The NLL and trace penalty are computed using low-rank operations only
    - Complexity: O(NM² + M³), not O(N³)
    
    Uses matrix determinant lemma and Woodbury formula to avoid building N×N matrices.
    
    Args:
        phi: Structural parameters (must have .Z, .kernel_params, .likelihood_params)
        X: Input locations (N, Q)
        Y: Observations (N,) or (N, D)
        kernel_fn: Kernel function kernel_fn(X1, X2, kernel_params) -> (N1, N2)
        residual: Residual type ("fitc", "sor", "dtc") - affects trace penalty
        key: Optional PRNG key (not used for Gaussian likelihood, accepted for compatibility)
    
    Returns:
        Scalar objective value (to be minimized in type-II optimisation).
    
    Notes:
        - This is an OPTIMISATION OBJECTIVE, not a model energy.
        - Does NOT form N×N matrices (uses low-rank operations only)
        - Includes VFE trace penalty term (FITC does not have this)
        - For Bayesian inference (SMC/MCMC), use energy/inertial.py instead
    """
    Y = Y[:, None] if Y.ndim == 1 else Y  # (N, D)
    N, D = Y.shape
    
    # Extract parameters
    Z = phi.Z  # (M, Q)
    noise_var = phi.likelihood_params["noise_var"]  # scalar or (D,)
    noise_var = jnp.asarray(noise_var)
    if noise_var.ndim == 0:
        noise_var = noise_var[None]  # (1,)
    noise_var = jnp.broadcast_to(noise_var, (D,))  # (D,)
    
    # Compute kernel components (only low-rank, no N×N matrices)
    K_xz = kernel_fn(X, Z, phi.kernel_params)  # (N, M)
    K_zz = kernel_fn(Z, Z, phi.kernel_params)  # (M, M)
    K_xx_diag = jnp.diag(kernel_fn(X, X, phi.kernel_params))  # (N,) - only diagonal
    
    # Ensure K_zz symmetry
    K_zz = 0.5 * (K_zz + K_zz.T)
    
    # Compute diag(Q_ff) efficiently (for trace penalty)
    diag_Q = diag_Q_ff(K_xz, K_zz, jitter=phi.jitter)  # (N,)
    
    # For each output dimension, compute VFE objective using low-rank operations
    def objective_for_output_d(y_d, noise_var_d):
        """
        Compute VFE objective for single output dimension using low-rank operations.
        
        VFE = NLL(y; 0, Q + σ²I) + (1/(2σ²)) * tr(K_ff - Q_ff)
        
        Using matrix determinant lemma and Woodbury formula:
        - log|σ²I + Q| = N*log(σ²) + log|I + (1/σ²) K_xz @ K_zz^{-1} @ K_xz^T|
                       = N*log(σ²) + log|K_zz + (1/σ²) K_xz^T @ K_xz| - log|K_zz|
        
        - y^T (σ²I + Q)^{-1} y = (1/σ²) [y^T y - y^T K_xz (σ² K_zz + K_xz^T K_xz)^{-1} K_xz^T y]
        """
        y_d = y_d.flatten()  # (N,)
        
        # Ensure numerical stability
        noise_var_safe = jnp.maximum(noise_var_d, 1e-6)
        
        # Cholesky of K_zz (M×M)
        L_zz = safe_cholesky(K_zz, jitter=phi.jitter, max_jitter=1e-2)
        
        # Compute A = K_zz^{-1} @ K_xz^T  (via solve)
        # A shape: (M, N)
        A = jax.scipy.linalg.cho_solve((L_zz, True), K_xz.T)  # (M, N)
        
        # For Woodbury: we need σ² K_zz + K_xz^T @ K_xz
        # This is M×M, so we can afford to form it
        K_xz_T_K_xz = K_xz.T @ K_xz  # (M, M)
        
        # B = σ² K_zz + K_xz^T @ K_xz  (M×M)
        B = noise_var_safe * K_zz + K_xz_T_K_xz
        
        # Ensure symmetry
        B = 0.5 * (B + B.T)
        
        # Cholesky of B (for Woodbury solve and determinant)
        L_B = safe_cholesky(B, jitter=phi.jitter, max_jitter=1e-2)
        
        # ====================================================================
        # 1. Compute log determinant: log|σ²I + Q|
        # ====================================================================
        # Using matrix determinant lemma:
        # |σ²I + K_xz @ K_zz^{-1} @ K_xz^T| = |K_zz + (1/σ²) K_xz^T K_xz| |K_zz^{-1}| |σ²I|
        #                                     = |K_zz + (1/σ²) K_xz^T K_xz| / |K_zz| * σ²^N
        #
        # Note: B = σ² K_zz + K_xz^T K_xz = σ² (K_zz + (1/σ²) K_xz^T K_xz)
        # So: |B| = σ²^M |K_zz + (1/σ²) K_xz^T K_xz|
        # Therefore: log|K_zz + (1/σ²) K_xz^T K_xz| = log|B| - M*log(σ²)
        #
        # Final result:
        # log|σ²I + Q| = log|B| - M*log(σ²) - log|K_zz| + N*log(σ²)
        #               = log|B| - log|K_zz| + (N-M)*log(σ²)
        
        log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diag(L_B)))
        log_det_Kzz = 2.0 * jnp.sum(jnp.log(jnp.diag(L_zz)))
        M = K_zz.shape[0]
        log_det_Q_plus_noise = log_det_B - log_det_Kzz + (N - M) * jnp.log(noise_var_safe)
        
        # ====================================================================
        # 2. Compute quadratic form: y^T (σ²I + Q)^{-1} y
        # ====================================================================
        # Using Woodbury formula:
        # (σ²I + Q)^{-1} = (1/σ²) I - (1/σ²²) K_xz @ (K_zz + (1/σ²) K_xz^T K_xz)^{-1} @ K_xz^T
        #
        # Note: K_zz + (1/σ²) K_xz^T K_xz = (1/σ²) (σ² K_zz + K_xz^T K_xz) = (1/σ²) B
        # So: (K_zz + (1/σ²) K_xz^T K_xz)^{-1} = σ² B^{-1}
        #
        # Therefore:
        # (σ²I + Q)^{-1} = (1/σ²) I - (1/σ²²) K_xz @ (σ² B^{-1}) @ K_xz^T
        #                 = (1/σ²) I - (1/σ²) K_xz @ B^{-1} @ K_xz^T
        #
        # So: y^T (σ²I + Q)^{-1} y = (1/σ²) y^T y - (1/σ²) y^T K_xz @ B^{-1} @ K_xz^T y
        
        # Compute K_xz^T @ y  (M,)
        K_xz_T_y = K_xz.T @ y_d  # (M,)
        
        # Solve B @ v = K_xz^T @ y  →  v = B^{-1} @ K_xz^T @ y
        v = jax.scipy.linalg.cho_solve((L_B, True), K_xz_T_y)  # (M,)
        
        # y^T K_xz @ B^{-1} @ K_xz^T y = (K_xz^T @ y)^T @ v = y^T K_xz @ v
        # But we already have v = B^{-1} @ K_xz^T @ y, so:
        # y^T K_xz @ B^{-1} @ K_xz^T y = y^T K_xz @ v = (K_xz^T @ y)^T @ v = K_xz_T_y @ v
        yT_Q_inv_y = (y_d @ y_d - K_xz_T_y @ v) / noise_var_safe
        
        # ====================================================================
        # 3. Compute NLL: NLL(y; 0, Q + σ²I)
        # ====================================================================
        # NLL = 0.5 * [N*log(2π) + log|σ²I + Q| + y^T (σ²I + Q)^{-1} y]
        nll = 0.5 * (N * jnp.log(2.0 * jnp.pi) + log_det_Q_plus_noise + yT_Q_inv_y)
        
        # ====================================================================
        # 4. Compute trace penalty: (1/(2σ²)) * tr(K_ff - Q_ff)
        # ====================================================================
        # This is the key difference between VFE and FITC
        if residual.lower() == "fitc":
            # For FITC: tr(K_ff - Q_ff) = tr(R) = sum(K_xx_diag - diag_Q)
            # Ensure diag_Q doesn't exceed K_xx_diag (numerical stability)
            diag_Q_safe = jnp.minimum(diag_Q, K_xx_diag * (1.0 + 1e-6))
            trace_correction = jnp.sum(jnp.maximum(K_xx_diag - diag_Q_safe, 0.0))
        else:
            # For SoR/DTC: R = 0, so tr(K_ff - Q_ff) = tr(K_ff) - tr(Q_ff)
            trace_correction = jnp.sum(K_xx_diag - diag_Q)
        
        # Trace penalty term: (1/(2σ²)) * tr(K_ff - Q_ff)
        trace_term = 0.5 * trace_correction / noise_var_safe
        
        # Ensure trace_term is finite (handle edge cases)
        trace_term = jnp.where(jnp.isfinite(trace_term), trace_term, 0.0)
        
        # ====================================================================
        # 5. Total VFE objective
        # ====================================================================
        total = nll + trace_term
        
        # Ensure total is finite
        total = jnp.where(jnp.isfinite(total), total, jnp.inf)
        
        return total
    
    # Sum over output dimensions
    total_objective = jnp.sum(
        jax.vmap(objective_for_output_d, in_axes=(1, 0))(Y, noise_var)
    )
    
    return total_objective


def make_vfe_objective(kernel_fn: Callable, residual: str = "fitc") -> Callable:
    """
    Factory function to create a VFE objective function bound to a kernel.
    
    This returns a callable with signature (phi, X, Y) -> scalar,
    suitable for use in type-II ML / variational inference methods (e.g. VGA).
    
    Args:
        kernel_fn: Kernel function kernel_fn(X1, X2, kernel_params) -> (N1, N2)
        residual: Residual type ("fitc", "sor", "dtc")
    
    Returns:
        Function (phi, X, Y) -> scalar objective (to be minimized)
    
    Notes:
        - This is an OPTIMISATION OBJECTIVE, not a model energy.
        - Uses low-rank operations only (O(NM²+M³) complexity)
        - For Bayesian inference (SMC/MCMC), use energy/inertial.py instead.
    """
    return partial(vfe_objective, kernel_fn=kernel_fn, residual=residual)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "vfe_objective",
    "make_vfe_objective",
]
