# infodynamics_jax/energy/q_marginals.py
from __future__ import annotations
import jax.numpy as jnp
import jax.scipy.linalg as jsp


def qfi_from_qu_full(phi, X, kernel_fn, m_u, S_u_full, jitter: float = 1e-8):
    """
    Build 1D marginals q(f_i|phi) induced by:
      q(u|phi)=N(m_u, S_u)
      p(f|u,phi) = N(A u, Lambda)  with A=K_xz K_zz^{-1}

    We return per-datum (mu_i, var_i) for factorised likelihood handling.

    Inputs:
      X: (N,Q)
      phi.Z: (M,Q)
      m_u: (M,D) or (M,) -> (M,1)
      S_u_full: (M,M) or (D,M,M) (shared or per-output)
      kernel_fn(X1,X2,phi.theta) -> (N,M) or (M,M)

    Returns:
      mu:  (N,D)
      var: (N,D) marginal variances
    """
    Z = phi.Z
    if m_u.ndim == 1:
        m_u = m_u[:, None]
    M, D = m_u.shape

    Kzz = kernel_fn(Z, Z, phi.theta)
    Kxz = kernel_fn(X, Z, phi.theta)
    Kxx_diag = jnp.diag(kernel_fn(X, X, phi.theta))  # (N,)

    Kzz = 0.5 * (Kzz + Kzz.T) + jitter * jnp.eye(M, dtype=Kzz.dtype)
    L = jnp.linalg.cholesky(Kzz)

    # A = Kxz Kzz^{-1} without explicit inverse:
    # Solve Kzz^{-1} v via cho_solve.
    A_T = jsp.cho_solve((L, True), Kxz.T)  # (M,N) = Kzz^{-1} Kzx
    A = A_T.T                               # (N,M)

    mu = A @ m_u                             # (N,D)

    # var_i = k_ii + a_i^T (S_u - Kzz) a_i
    # handle S_u_full shared or per-output
    if S_u_full.ndim == 2:
        Su_minus_K = S_u_full - Kzz          # (M,M)
        # compute diag(A Su_minus_K A^T) efficiently
        AS = A @ Su_minus_K                  # (N,M)
        corr = jnp.sum(AS * A, axis=1)       # (N,)
        var = (Kxx_diag + corr)[:, None] * jnp.ones((1, D), dtype=mu.dtype)
    else:
        # (D,M,M)
        def one_d(d):
            Su_minus_Kd = S_u_full[d] - Kzz
            ASd = A @ Su_minus_Kd
            corr_d = jnp.sum(ASd * A, axis=1)
            return Kxx_diag + corr_d
        var = jnp.stack([one_d(d) for d in range(D)], axis=1)  # (N,D)

    var = jnp.maximum(var, 0.0)
    return mu, var