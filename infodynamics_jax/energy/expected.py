# infodynamics_jax/energy/expected.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from .gh import GaussHermite


Estimator = Literal["gh", "mc"]


import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass
class VariationalState:
    m_u: jnp.ndarray
    L_u: jnp.ndarray | None = None
    s_u_diag: jnp.ndarray | None = None
    cov_type: str = "full"   # static

    def tree_flatten(self):
        # ONLY numerical parameters go into children
        children = (self.m_u, self.L_u, self.s_u_diag)
        aux_data = self.cov_type
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        m_u, L_u, s_u_diag = children
        return cls(
            m_u=m_u,
            L_u=L_u,
            s_u_diag=s_u_diag,
            cov_type=aux_data,
        )


def _as_2d(Y: jnp.ndarray) -> jnp.ndarray:
    return Y[:, None] if Y.ndim == 1 else Y


def qfi_from_qu_full(phi, X: jnp.ndarray, kernel_fn: Callable, m_u: jnp.ndarray, L_u: jnp.ndarray
                    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute per-datum variational marginals q(f_i|phi) induced by q(u|phi) with FULL covariance.

    Using sparse GP conditional:
        f_i | u,phi ~ N(a_i^T u,  k_ii - q_ii)   (SoR/DTC)
    And with FITC-style residual, you can add residual back at SK layer.
    Here we implement the generic:
        mu_i = a_i^T m_u
        var_i = (k_ii - q_ii) + a_i^T S_u a_i
    where a_i^T = k_i^T K_uu^{-1}.

    Shapes:
    - X: (N,Q)
    - Z: (M,Q) inside phi
    - m_u: (M,D)
    - L_u: (D,M,M) or (M,M) if D=1

    Returns:
    - mu_f: (N,D)
    - var_f: (N,D)
    """
    Z = phi.Z
    Kuu = kernel_fn(Z, Z, phi.kernel_params)
    Kuu = 0.5 * (Kuu + Kuu.T) + phi.jitter * jnp.eye(Kuu.shape[0], dtype=Kuu.dtype)
    L = jnp.linalg.cholesky(Kuu)

    Kxu = kernel_fn(X, Z, phi.kernel_params)          # (N,M)
    Kxx_diag = jnp.diag(kernel_fn(X, X, phi.kernel_params))  # (N,)

    # Solve for A = Kxu Kuu^{-1} = (solve(Kuu, Kux))^T but stably via chol
    # We need a_i row-wise. Compute V = L^{-1} Kux = L^{-1} Kxu^T => (M,N)
    V = jax.scipy.linalg.solve_triangular(L, Kxu.T, lower=True)  # (M,N)
    # q_ii = k_i^T Kuu^{-1} k_i = ||V[:,i]||^2
    q_diag = jnp.sum(V**2, axis=0)  # (N,)
    # a_i = Kuu^{-1} k_i = L^{-T} L^{-1} k_i ; but we can use:
    # a_i^T u = k_i^T Kuu^{-1} u = (L^{-1}k_i)^T (L^{-1}u) = V[:,i]^T (L^{-1}u)
    # We'll compute A explicitly for covariance term a_i^T S a_i:
    A = jax.scipy.linalg.cho_solve((L, True), Kxu.T).T  # (N,M)

    m_u = _as_2d(m_u)  # (M,D)
    mu_f = A @ m_u     # (N,D)

    # conditional residual term (k_ii - q_ii)
    cond = (Kxx_diag - q_diag)[:, None]  # (N,1)
    cond = jnp.clip(cond, a_min=0.0)

    # a_i^T S a_i where S = L_u L_u^T
    if L_u.ndim == 2:
        # D=1
        S = L_u @ L_u.T  # (M,M)
        quad = jnp.einsum("nm,mk,nk->n", A, S, A)[:, None]  # (N,1)
    else:
        # (D,M,M)
        def quad_d(Ld, md_unused):
            Sd = Ld @ Ld.T
            return jnp.einsum("nm,mk,nk->n", A, Sd, A)
        quad = jax.vmap(quad_d, in_axes=(0, 0))(L_u, jnp.zeros((L_u.shape[0],)))  # (D,N)
        quad = quad.T  # (N,D)

    var_f = cond + quad
    var_f = jnp.clip(var_f, a_min=0.0)
    return mu_f, var_f


def expected_nll_factorised_gh(phi, X, Y, kernel_fn, state: VariationalState,
                              nll_1d_fn, gh: GaussHermite) -> jnp.ndarray:
    """
    Deterministic approximation:
        sum_i E_{q(f_i|phi)}[ nll(y_i, f_i, phi) ]
    where q(f_i|phi) is induced by q(u|phi).

    Assumes likelihood factorises across i and across D (independent outputs).
    """
    Y = _as_2d(Y)

    if state.cov_type != "full" or state.L_u is None:
        raise ValueError("expected_nll_factorised_gh currently expects full-cov state with L_u.")

    mu_f, var_f = qfi_from_qu_full(phi, X, kernel_fn, state.m_u, state.L_u)  # (N,D)

    def one_dim(y, mu, var):
        return gh.expect_nll_1d(y, mu, var, phi, nll_1d_fn)

    # vmap over (N,D)
    val = jax.vmap(
        lambda yrow, murow, varrow: jnp.sum(jax.vmap(one_dim)(yrow, murow, varrow)),
        in_axes=(0, 0, 0),
    )(Y, mu_f, var_f)

    return jnp.sum(val)


def expected_nll_factorised_mc(phi, X, Y, kernel_fn, state: VariationalState,
                              nll_1d_fn, key: jax.random.KeyArray,
                              n_samples: int = 16) -> jnp.ndarray:
    """
    Monte Carlo estimator for:
        sum_i E_{q(f_i|phi)}[ nll(y_i, f_i, phi) ].

    Advantage: no GH tables / special functions.
    Disadvantage: variance; but jit/vmap-friendly and works universally.

    Implementation: sample eps ~ N(0,1), f = mu + sqrt(var)*eps.
    """
    Y = _as_2d(Y)

    if state.cov_type != "full" or state.L_u is None:
        raise ValueError("expected_nll_factorised_mc currently expects full-cov state with L_u.")

    mu_f, var_f = qfi_from_qu_full(phi, X, kernel_fn, state.m_u, state.L_u)  # (N,D)
    var_f = jnp.clip(var_f, a_min=0.0)

    N, D = Y.shape
    eps = jax.random.normal(key, shape=(n_samples, N, D), dtype=Y.dtype)
    f_samps = mu_f[None, :, :] + jnp.sqrt(var_f[None, :, :]) * eps  # (S,N,D)

    # average over samples
    def nll_of_sample(f):
        # sum over N,D
        return jnp.sum(jax.vmap(lambda yi, fi: jnp.sum(jax.vmap(lambda yijd, fijd: nll_1d_fn(yijd, fijd, phi))(yi, fi)))(Y, f))

    vals = jax.vmap(nll_of_sample)(f_samps)  # (S,)
    return jnp.mean(vals)
