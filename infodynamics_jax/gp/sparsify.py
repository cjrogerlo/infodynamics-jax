# infodynamics_jax/gp/sparsify.py
"""
Sparsified kernel implementation.

This module provides:
- Q_ff: Nyström low-rank projection
- Residual functions: FITC, SoR, DTC
- SparsifiedKernel: Main abstraction S_ff = Q + R

According to the sparsified kernel framework:
- u is already marginalised out in S_ff = Q + R
- The sparsified GP is simply a GP with kernel restricted to this family
- Different sparsification methods correspond to different residual structures
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# ============================================================================
# 1. Nyström projection: Q_ff
# ============================================================================

def Q_ff(K_xz, K_zz):
    """
    Compute Q_ff = K_xz @ K_zz^{-1} @ K_xz^T
    
    Args:
        K_xz: (N, M) kernel matrix between X and Z
        K_zz: (M, M) kernel matrix between Z and Z
    
    Returns:
        Q_ff: (N, N) approximate kernel matrix
    """
    L_zz = jnp.linalg.cholesky(K_zz)
    A = jax.scipy.linalg.cho_solve((L_zz, True), K_xz.T)  # (M, N)
    Q_ff = K_xz @ A  # (N, N)
    return Q_ff


def diag_Q_ff(K_xz, K_zz):
    """
    Compute only diagonal of Q_ff = diag(K_xz @ K_zz^{-1} @ K_xz^T)
    
    More efficient: q_ii = ||A[i,:]||^2 where A = K_xz @ K_zz^{-1}
    
    Args:
        K_xz: (N, M)
        K_zz: (M, M)
    
    Returns:
        diag_Q: (N,) diagonal of Q_ff
    """
    L_zz = jnp.linalg.cholesky(K_zz)
    A = jax.scipy.linalg.cho_solve((L_zz, True), K_xz.T).T  # (N, M)
    diag_Q = jnp.sum(A**2, axis=1)  # (N,)
    return diag_Q


# ============================================================================
# 2. Residual functions: R structure (only diag)
# ============================================================================

def fitc_diag(K_xx_diag, diag_Q):
    """
    FITC diagonal residual: R = diag(K_xx_diag - diag_Q)
    
    Args:
        K_xx_diag: (N,) diagonal of K_xx
        diag_Q: (N,) diagonal of Q
    
    Returns:
        r_diag: (N,) diagonal of residual matrix
    """
    return K_xx_diag - diag_Q


def sor_diag(K_xx_diag, diag_Q):
    """
    SoR residual: R = 0 (zero residual)
    """
    return jnp.zeros_like(diag_Q)


def dtc_diag(K_xx_diag, diag_Q):
    """
    DTC residual: R = 0 (same as SoR)
    """
    return jnp.zeros_like(diag_Q)


# Map residual names to functions
RESIDUAL_FUNCS = {
    "fitc": fitc_diag,
    "sor": sor_diag,
    "dtc": dtc_diag,
}


# ============================================================================
# 3. SparsifiedKernel: Main abstraction S_ff = Q + R
# ============================================================================

@dataclass
class SparsifiedKernel:
    """
    Sparsified kernel S_ff = Q + R.
    
    This is the main abstraction: u is already marginalised out.
    The sparsified GP is simply a GP with kernel restricted to this family.
    
    Different sparsification methods correspond to different residual structures:
    - FITC: R = diag(K_xx - Q_ff)
    - SoR/DTC: R = 0
    """
    kernel_fn: callable
    residual: str = "fitc"

    def S_ff(self, params, X, Z, jitter=1e-8):
        """
        Compute sparsified kernel S_ff = Q + R
        
        Args:
            params: kernel parameters
            X: (N, Q) input locations
            Z: (M, Q) inducing locations
            jitter: numerical stability jitter
        
        Returns:
            S_ff: (N, N) sparsified kernel matrix
        """
        # Compute cross and inducing kernels
        K_xz = self.kernel_fn(params, X, Z)  # (N, M)
        K_zz = self.kernel_fn(params, Z, Z)  # (M, M)
        
        # Add jitter to K_zz for numerical stability
        K_zz = 0.5 * (K_zz + K_zz.T) + jitter * jnp.eye(K_zz.shape[0], dtype=K_zz.dtype)
        
        # Compute Q
        Q = Q_ff(K_xz, K_zz)  # (N, N)
        
        # Compute residual (only need diagonal for FITC)
        residual_func = RESIDUAL_FUNCS.get(self.residual.lower(), fitc_diag)
        
        if self.residual.lower() == "fitc":
            # Only compute diagonal of K_xx (don't materialise full matrix)
            K_xx_diag = jnp.diag(self.kernel_fn(params, X, X))  # (N,)
            diag_Q = diag_Q_ff(K_xz, K_zz)  # (N,) - more efficient
            r_diag = residual_func(K_xx_diag, diag_Q)  # (N,)
            R = jnp.diag(r_diag)  # (N, N)
        else:
            # SoR/DTC: zero residual
            diag_Q = diag_Q_ff(K_xz, K_zz)
            r_diag = residual_func(jnp.zeros_like(diag_Q), diag_Q)
            R = jnp.diag(r_diag)
        
        return Q + R


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Nyström
    "Q_ff",
    "diag_Q_ff",
    # Residuals
    "fitc_diag",
    "sor_diag",
    "dtc_diag",
    # Main abstraction
    "SparsifiedKernel",
]