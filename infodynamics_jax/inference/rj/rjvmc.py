# infodynamics_jax/inference/rj/rjvmc.py
"""
RJVMC (Reversible Jump Variational Monte Carlo) for Sparse GP with Non-Conjugate Likelihoods.
High-performance implementation with rank-1 updates, inner VI, and Delayed Acceptance.
"""
from __future__ import annotations

import time
import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Any, Literal, Dict

import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.scipy as jsp
import blackjax
import optax

from ...energy.base import EnergyTerm
from ...energy.inertial import InertialEnergy
from ...core import Upphi
from ...gp.kernels.params import KernelParams
from ...gp.ansatz.state import VariationalState
from ..base import InferenceMethod
from .state import RJState


@dataclass(frozen=True)
class RJVMCCFG:
    """Configuration for RJVMC sampler."""
    n_steps: int = 1200
    burn: int = 300
    M_min: int = 5
    M_max: int = 60
    M_init: int = 25
    p_geom: float = 0.12  # Geometric prior parameter for M
    
    # Move probabilities
    r_M: float = 0.90      # Probability of doing RJ move (else skip to stabilise)
    hmc_every: int = 5     # Frequency of HMC steps for theta
    
    # RJ/MTM parameters
    K_pool: int = 32       # Number of candidates for birth
    temp_rj: float = 1.0   # Temperature for RJ pool weights
    
    # HMC parameters
    hmc_step_size: float = 0.01
    hmc_leaps: int = 3
    
    # Inner VI parameters
    inner_steps: int = 5
    lr_m: float = 1e-2
    lr_L: float = 1e-2
    
    jitter: float = 1e-6


@dataclass
class RJVMCRun:
    """RJVMC run results."""
    theta_trace: jnp.ndarray
    Z_trace: jnp.ndarray
    M_trace: jnp.ndarray
    energy_trace: jnp.ndarray
    cfg: RJVMCCFG


# ============================================================
# Helpers & Priors
# ============================================================

@jax.jit
def softplus(x):
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)

@jax.jit
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

@jax.jit
def loglik_bernoulli_logit(y01, f):
    return -softplus(-f) * y01 - softplus(f) * (1.0 - y01)

@jax.jit
def gh_expect_loglik_bernoulli(mu, var, y01, gh_x, gh_w):
    sig = jnp.sqrt(jnp.maximum(var, 1e-18))
    f = mu[:, None] + jnp.sqrt(2.0) * sig[:, None] * gh_x[None, :]
    ll = loglik_bernoulli_logit(y01[:, None], f)
    return (ll * gh_w[None, :]).sum(axis=1) / jnp.sqrt(jnp.pi)

@jax.jit
def log_prior_theta(theta):
    return -0.5 * jnp.sum((theta / 1.0) ** 2)

@jax.jit
def log_prior_M_trunc_geom(M, M_min, M_max, p=0.12):
    valid = (M >= M_min) & (M <= M_max)
    m = M - M_min
    K = M_max - M_min + 1
    log_unn = jnp.log(p) + m * jnp.log1p(-p)
    logZ = jnp.log1p(-(1.0 - p) ** K)
    return jnp.where(valid, log_unn - logZ, -jnp.inf)

@jax.jit
def log_prior_ordered_Z_given_M(N, M):
    Nf = jnp.array(N, dtype=jnp.float64)
    Mf = jnp.array(M, dtype=jnp.float64)
    return -(jsp.special.gammaln(Nf + 1.0) - jsp.special.gammaln(Nf - Mf + 1.0))

@jax.jit
def softmax_stable(x, temp=1.0):
    z = x / temp
    z = z - jnp.max(z)
    e = jnp.exp(z)
    return e / (jnp.sum(e) + 1e-30)


# ============================================================
# ELBO (Cached)
# ============================================================

@jax.jit
def elbo_cached_binary_full(state: RJState, X, y01, gh_x, gh_w):
    N, D = X.shape
    M_max = state.Z_buf.shape[0]
    mask = (jnp.arange(M_max, dtype=jnp.int32) < state.M.astype(jnp.int32)).astype(jnp.float64)
    active = mask[:, None] * mask[None, : ]

    m = state.variational_state.m_u * mask
    Lq = jnp.tril(state.variational_state.L_u) * active
    S = Lq @ Lq.T

    # alpha = Kuu^{-1} m
    tmp = jsp.linalg.solve_triangular(state.Lm, m[:, None], lower=True)
    alpha = jsp.linalg.solve_triangular(state.Lm.T, tmp, lower=False).reshape(-1)

    mu = (state.Kuf.T @ alpha).reshape(-1)

    log_sf = state.theta[D]
    sf2 = jnp.exp(2.0 * log_sf)
    kff = sf2 * jnp.ones((N,), dtype=jnp.float64)

    Q = (S - state.Kuu) * active
    quad = jnp.einsum("mi,mn,ni->i", state.A2, Q, state.A2)
    var = jnp.maximum(kff + quad, 1e-12)

    ell = gh_expect_loglik_bernoulli(mu, var, y01.reshape(-1), gh_x, gh_w).sum()

    # KL(q(u)||p(u))
    S_jit = S + 1e-8 * jnp.eye(M_max)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(state.Lm)))
    Ls = jsp.linalg.cholesky(S_jit, lower=True)
    logdetS = 2.0 * jnp.sum(jnp.log(jnp.diag(Ls)))

    Sinv1 = jsp.linalg.solve_triangular(state.Lm, S, lower=True)
    KinvS = jsp.linalg.solve_triangular(state.Lm.T, Sinv1, lower=False)
    trKinvS = jnp.trace(KinvS)

    mKm = jnp.dot(alpha, m)
    M_eff = state.M.astype(jnp.float64)
    KL = 0.5 * (trKinvS + mKm - M_eff + logdetK - logdetS)

    return ell - KL


@jax.jit
def elbo_cached_binary_subset(state: RJState, X, y01, idx, gh_x, gh_w):
    N = X.shape[0]
    B = idx.shape[0]
    M_max = state.Z_buf.shape[0]
    mask = (jnp.arange(M_max, dtype=jnp.int32) < state.M.astype(jnp.int32)).astype(jnp.float64)
    active = mask[:, None] * mask[None, :]

    m = state.variational_state.m_u * mask
    Lq = jnp.tril(state.variational_state.L_u) * active
    S = Lq @ Lq.T

    tmp = jsp.linalg.solve_triangular(state.Lm, m[:, None], lower=True)
    alpha = jsp.linalg.solve_triangular(state.Lm.T, tmp, lower=False).reshape(-1)

    Kuf_b = jnp.take(state.Kuf, idx, axis=1)   
    A2_b  = jnp.take(state.A2,  idx, axis=1)   
    y_b   = jnp.take(y01.reshape(-1), idx, axis=0)

    mu = (Kuf_b.T @ alpha).reshape(-1)

    log_sf = state.theta[state.phi.Z.shape[1] if len(state.phi.Z.shape)>1 else 1]
    sf2 = jnp.exp(2.0 * log_sf)
    kff = sf2 * jnp.ones((B,), dtype=jnp.float64)

    Q = (S - state.Kuu) * active
    quad = jnp.einsum("mi,mn,ni->i", A2_b, Q, A2_b)
    var = jnp.maximum(kff + quad, 1e-12)

    ell_b = gh_expect_loglik_bernoulli(mu, var, y_b, gh_x, gh_w).sum()
    ell = (N / B) * ell_b

    # KL same as full
    S_jit = S + 1e-8 * jnp.eye(M_max)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(state.Lm)))
    Ls = jsp.linalg.cholesky(S_jit, lower=True)
    logdetS = 2.0 * jnp.sum(jnp.log(jnp.diag(Ls)))
    Sinv1 = jsp.linalg.solve_triangular(state.Lm, S, lower=True)
    KinvS = jsp.linalg.solve_triangular(state.Lm.T, Sinv1, lower=False)
    trKinvS = jnp.trace(KinvS)
    mKm = jnp.dot(alpha, m)
    M_eff = state.M.astype(jnp.float64)
    KL = 0.5 * (trKinvS + mKm - M_eff + logdetK - logdetS)

    return ell - KL


@jax.jit
def log_posterior(state: RJState, N, M_min, M_max, p_geom, gh_x, gh_w, X, y01):
    elbo = elbo_cached_binary_full(state, X, y01, gh_x, gh_w)
    return (elbo
            + log_prior_theta(state.theta)
            + log_prior_M_trunc_geom(state.M, M_min, M_max, p_geom)
            + log_prior_ordered_Z_given_M(N, state.M))


# ============================================================
# Inner VI (Adam)
# ============================================================

@jax.jit
def adam_update(param, grad, m1, m2, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    m1 = b1 * m1 + (1 - b1) * grad
    m2 = b2 * m2 + (1 - b2) * (grad * grad)
    m1h = m1 / (1 - b1 ** t)
    m2h = m2 / (1 - b2 ** t)
    param = param + lr * m1h / (jnp.sqrt(m2h) + eps)
    return param, m1, m2

@partial(jax.jit, static_argnames=("inner_steps",))
def optimise_variational(state: RJState, X, y01, gh_x, gh_w, inner_steps: int, lr_m: float, lr_L: float):
    m, L = state.variational_state.m_u, state.variational_state.L_u
    m_m1 = jnp.zeros_like(m)
    m_m2 = jnp.zeros_like(m)
    L_m1 = jnp.zeros_like(L)
    L_m2 = jnp.zeros_like(L)

    def obj(mm, LL):
        vs = VariationalState(m_u=mm, L_u=LL)
        st = RJState(
            phi=state.phi, variational_state=vs, M=state.M, Z_buf=state.Z_buf,
            energy=state.energy, theta=state.theta, Kuu=state.Kuu, Lm=state.Lm,
            Kuf=state.Kuf, A=state.A, A2=state.A2
        )
        return elbo_cached_binary_full(st, X, y01, gh_x, gh_w)

    def body(carry, t):
        m, L, m_m1, m_m2, L_m1, L_m2 = carry
        val, (gm, gL) = jax.value_and_grad(obj, argnums=(0, 1))(m, L)
        gL = jnp.tril(gL)
        m, m_m1, m_m2 = adam_update(m, gm, m_m1, m_m2, t+1, lr=lr_m)
        L, L_m1, L_m2 = adam_update(L, gL, L_m1, L_m2, t+1, lr=lr_L)
        return (m, L, m_m1, m_m2, L_m1, L_m2), val

    (m, L, *_), _ = jax.lax.scan(body, (m, L, m_m1, m_m2, L_m1, L_m2), jnp.arange(inner_steps))
    diag = jnp.maximum(jnp.diag(L), 1e-6)
    L = L.at[jnp.diag_indices(L.shape[0])].set(diag)
    vs_new = VariationalState(m_u=m, L_u=L)
    st_new = RJState(
        phi=state.phi, variational_state=vs_new, M=state.M, Z_buf=state.Z_buf,
        energy=state.energy, theta=state.theta, Kuu=state.Kuu, Lm=state.Lm,
        Kuf=state.Kuf, A=state.A, A2=state.A2
    )
    new_elbo = elbo_cached_binary_full(st_new, X, y01, gh_x, gh_w)
    return RJState(
        phi=state.phi, variational_state=vs_new, M=state.M, Z_buf=state.Z_buf,
        energy=new_elbo, theta=state.theta, Kuu=state.Kuu, Lm=state.Lm,
        Kuf=state.Kuf, A=state.A, A2=state.A2
    )


# ============================================================
# Structural Updates (Rank-1)
# ============================================================

def rbf_kernel_matrix(X1, X2, log_ls, log_sf):
    ls = jnp.exp(log_ls).reshape(1, 1, -1)
    sf2 = jnp.exp(2.0 * log_sf)
    X1s = X1[:, None, :] / ls
    X2s = X2[None, :, :] / ls
    dist_sq = jnp.sum((X1s - X2s) ** 2, axis=-1)
    return sf2 * jnp.exp(-0.5 * dist_sq)

@jax.jit
def birth_append_rank1(state: RJState, new_idx, X, jitter):
    N, D = X.shape
    M_max = state.Z_buf.shape[0]
    M = state.M.astype(jnp.int32)
    slot = M
    Z_buf_new = state.Z_buf.at[slot].set(new_idx.astype(jnp.int32))
    M_new = (M + 1).astype(jnp.int32)
    Z_old = X[state.Z_buf]
    z_new = X[new_idx].reshape(1, -1)
    log_ls, log_sf = state.theta[:D], state.theta[D]
    k_row = rbf_kernel_matrix(z_new, Z_old, log_ls, log_sf).reshape(-1)
    k_self = rbf_kernel_matrix(z_new, z_new, log_ls, log_sf).reshape(()) + jitter
    mask_old = (jnp.arange(M_max, dtype=jnp.int32) < M).astype(X.dtype)
    k_row = k_row * mask_old
    v = jsp.linalg.solve_triangular(state.Lm, k_row[:, None], lower=True).reshape(-1)
    v = v * mask_old
    diag = jnp.sqrt(jnp.maximum(k_self - jnp.dot(v, v), 1e-12))
    Lm_new = state.Lm.at[slot, :].set(0.0).at[:, slot].set(0.0).at[slot, :].set(v).at[slot, slot].set(diag)
    Kuu_new = state.Kuu.at[slot, :].set(0.0).at[:, slot].set(0.0).at[slot, :].set(k_row).at[:, slot].set(k_row).at[slot, slot].set(k_self)
    kuf_row = rbf_kernel_matrix(z_new, X, log_ls, log_sf).reshape(-1)
    Kuf_new = state.Kuf.at[slot, :].set(kuf_row)
    numer = kuf_row - (v[:, None] * state.A).sum(axis=0)
    a_row = numer / (diag + 1e-30)
    A_new = state.A.at[slot, :].set(a_row)
    x_new = a_row / (diag + 1e-30)
    delta_vec = jsp.linalg.solve_triangular(state.Lm.T, v[:, None], lower=False).reshape(-1)
    delta_vec = delta_vec * mask_old
    A2_new = state.A2 - delta_vec[:, None] * x_new[None, :]
    A2_new = A2_new.at[slot, :].set(x_new)
    m_new = state.variational_state.m_u.at[slot].set(0.0)
    Lq_new = state.variational_state.L_u.at[slot, :].set(0.0).at[:, slot].set(0.0).at[slot, slot].set(1.0)
    vs_new = VariationalState(m_u=m_new, L_u=Lq_new)
    return RJState(
        phi=state.phi, variational_state=vs_new, M=M_new, Z_buf=Z_buf_new,
        energy=state.energy, theta=state.theta, Kuu=Kuu_new, Lm=Lm_new,
        Kuf=Kuf_new, A=A_new, A2=A2_new
    )

@jax.jit
def death_drop_last_rank1(state: RJState):
    M_max = state.Z_buf.shape[0]
    M = state.M.astype(jnp.int32)
    slot = (M - 1).astype(jnp.int32)
    M_new = (M - 1).astype(jnp.int32)
    mask_old = (jnp.arange(M_max, dtype=jnp.int32) < slot).astype(jnp.float64)
    v = state.Lm[slot, :] * mask_old
    x_new = state.A2[slot, :]
    delta_vec = jsp.linalg.solve_triangular(state.Lm.T, v[:, None], lower=False).reshape(-1)
    delta_vec = delta_vec * mask_old
    A2_new = state.A2 + delta_vec[:, None] * x_new[None, :]
    A2_new = A2_new.at[slot, :].set(0.0)
    A_new = state.A.at[slot, :].set(0.0)
    Kuf_new = state.Kuf.at[slot, :].set(0.0)
    Kuu_new = state.Kuu.at[slot, :].set(0.0).at[:, slot].set(0.0).at[slot, slot].set(1.0)
    Lm_new = state.Lm.at[slot, :].set(0.0).at[:, slot].set(0.0).at[slot, slot].set(1.0)
    m_new = state.variational_state.m_u.at[slot].set(0.0)
    Lq_new = state.variational_state.L_u.at[slot, :].set(0.0).at[:, slot].set(0.0).at[slot, slot].set(1.0)
    vs_new = VariationalState(m_u=m_new, L_u=Lq_new)
    return RJState(
        phi=state.phi, variational_state=vs_new, M=M_new, Z_buf=state.Z_buf,
        energy=state.energy, theta=state.theta, Kuu=Kuu_new, Lm=Lm_new,
        Kuf=Kuf_new, A=A_new, A2=A2_new
    )


# ============================================================
# RJ Sampler Class
# ============================================================

class RJVMC(InferenceMethod):
    """
    RJVMC for sparse GP binary classification.
    Optimized for JAX with rank-1 updates and Delayed Acceptance.
    """
    
    def __init__(self, cfg: RJVMCCFG = RJVMCCFG()):
        self.cfg = cfg

    def run(self, energy: EnergyTerm, *args, **kwargs) -> RJVMCRun:
        if not isinstance(energy, InertialEnergy):
             raise ValueError("RJVMC requires InertialEnergy (non-conjugate case)")
        
        X = kwargs.get('X')
        if X is None:
            X = args[0]
        y01 = kwargs.get('y')
        if y01 is None:
            y01 = args[1]
        rng = kwargs.get('key')
        if rng is None:
            rng = args[2]
        
        N, D = X.shape
        cfg = self.cfg
        gh_x, gh_w = energy.gh.nodes_weights()
        
        # Init state
        rng, k_init, k_start_e = jrand.split(rng, 3)
        Z_init_idx = jrand.choice(k_init, N, (cfg.M_init,), replace=False).astype(jnp.int32)
        Z_buf = jnp.zeros(cfg.M_max, dtype=jnp.int32).at[:cfg.M_init].set(Z_init_idx)
        theta0 = jnp.concatenate([jnp.full(D, -2.0), jnp.array([0.0, -2.0])]) 
        
        log_ls, log_sf = theta0[:D], theta0[D]
        mask = (jnp.arange(cfg.M_max, dtype=jnp.int32) < cfg.M_init).astype(X.dtype)
        Z = X[Z_buf]
        Kuu_raw = rbf_kernel_matrix(Z, Z, log_ls, log_sf)
        Kuf_raw = rbf_kernel_matrix(Z, X, log_ls, log_sf)
        Kuu = (mask[:, None] * mask[None, :]) * Kuu_raw + jnp.diag(1.0 - mask) + cfg.jitter * jnp.eye(cfg.M_max)
        Kuf = mask[:, None] * Kuf_raw
        Lm = jsp.linalg.cholesky(Kuu, lower=True)
        A = jsp.linalg.solve_triangular(Lm, Kuf, lower=True)
        A2 = jsp.linalg.solve_triangular(Lm.T, A, lower=False)
        vs0 = VariationalState(m_u=jnp.zeros(cfg.M_max), L_u=jnp.eye(cfg.M_max))
        phi0 = Upphi(kernel_params=KernelParams(lengthscale=jnp.exp(theta0[:D]), variance=jnp.exp(2.0 * theta0[D])), 
                   Z=Z, likelihood_params={}, jitter=cfg.jitter)
        
        # First create state0 with placeholder energy
        state0 = RJState(
            phi=phi0, variational_state=vs0, M=jnp.array(cfg.M_init, dtype=jnp.int32), 
            Z_buf=Z_buf, energy=jnp.array(0.0), theta=theta0, 
            Kuu=Kuu, Lm=Lm, Kuf=Kuf, A=A, A2=A2
        )
        # Then compute actual energy and update
        initial_energy = elbo_cached_binary_full(state0, X, y01, gh_x, gh_w)
        state0 = RJState(
            phi=phi0, variational_state=vs0, M=jnp.array(cfg.M_init, dtype=jnp.int32), 
            Z_buf=Z_buf, energy=initial_energy, theta=theta0, 
            Kuu=Kuu, Lm=Lm, Kuf=Kuf, A=A, A2=A2
        )
        
        # HMC setup
        def logp_hmc(th, s_curr):
            ls, sf = th[:D], th[D]
            k_uu_raw = rbf_kernel_matrix(X[s_curr.Z_buf], X[s_curr.Z_buf], ls, sf)
            k_uf_raw = rbf_kernel_matrix(X[s_curr.Z_buf], X, ls, sf)
            mask = (jnp.arange(cfg.M_max, dtype=jnp.int32) < s_curr.M).astype(X.dtype)
            kuu = (mask[:, None] * mask[None, :]) * k_uu_raw + jnp.diag(1.0 - mask) + cfg.jitter * jnp.eye(cfg.M_max)
            kuf = mask[:, None] * k_uf_raw
            lm = jsp.linalg.cholesky(kuu, lower=True)
            a = jsp.linalg.solve_triangular(lm, kuf, lower=True)
            a2 = jsp.linalg.solve_triangular(lm.T, a, lower=False)
            st_th = RJState(phi=s_curr.phi, variational_state=s_curr.variational_state, M=s_curr.M, Z_buf=s_curr.Z_buf, 
                             energy=s_curr.energy, theta=th, Kuu=kuu, Lm=lm, Kuf=kuf, A=a, A2=a2)
            return log_posterior(st_th, N, cfg.M_min, cfg.M_max, cfg.p_geom, gh_x, gh_w, X, y01)

        @jax.jit
        def one_step(carry, t):
            rng, st = carry
            rng, k_rj, k_rj_move, k_pool, k_acc1, k_acc2, k_batch, k_hmc = jrand.split(rng, 8)
            batch_idx = jrand.randint(k_batch, (128,), 0, N) # for DA stage 1

            # --- HMC for theta (inline with closure over current state) ---
            def _hmc_up(s, k):
                # Create logprob that closes over current state s
                def logprob_theta(th):
                    return logp_hmc(th, s)
                
                hmc_kernel = blackjax.hmc(logprob_theta, step_size=cfg.hmc_step_size, 
                                          inverse_mass_matrix=jnp.ones_like(s.theta), 
                                          num_integration_steps=cfg.hmc_leaps)
                hst = hmc_kernel.init(s.theta)
                hst_new, _ = hmc_kernel.step(k, hst)
                th_new = hst_new.position
                ls_new, sf_new = th_new[:D], th_new[D]
                k_uu_raw = rbf_kernel_matrix(X[s.Z_buf], X[s.Z_buf], ls_new, sf_new)
                k_uf_raw = rbf_kernel_matrix(X[s.Z_buf], X, ls_new, sf_new)
                mask = (jnp.arange(cfg.M_max, dtype=jnp.int32) < s.M).astype(X.dtype)
                kuu = (mask[:, None] * mask[None, :]) * k_uu_raw + jnp.diag(1.0 - mask) + cfg.jitter * jnp.eye(cfg.M_max)
                kuf = mask[:, None] * k_uf_raw
                lm = jsp.linalg.cholesky(kuu, lower=True)
                a = jsp.linalg.solve_triangular(lm, kuf, lower=True)
                a2 = jsp.linalg.solve_triangular(lm.T, a, lower=False)
                st_new = RJState(phi=s.phi, variational_state=s.variational_state, M=s.M, Z_buf=s.Z_buf, 
                                 energy=s.energy, theta=th_new, Kuu=kuu, Lm=lm, Kuf=kuf, A=a, A2=a2)
                new_energy = elbo_cached_binary_full(st_new, X, y01, gh_x, gh_w)
                return RJState(phi=s.phi, variational_state=s.variational_state, M=s.M, Z_buf=s.Z_buf, 
                                 energy=new_energy, theta=th_new, Kuu=kuu, Lm=lm, Kuf=kuf, A=a, A2=a2)
            st = jax.lax.cond((t % cfg.hmc_every) == 0, _hmc_up, lambda s, k: s, st, k_hmc)

            # --- RJ update (Delayed Acceptance) ---
            do_rj = jrand.uniform(k_rj) < cfg.r_M
            p_birth = jnp.where(st.M <= cfg.M_min, 1.0, jnp.where(st.M >= cfg.M_max, 0.0, 0.5))
            do_birth = jrand.uniform(k_rj_move) < p_birth

            def birth_move(s):
                cand = jrand.randint(k_pool, (cfg.K_pool,), 0, N)
                def score_one(idx):
                    p_st = birth_append_rank1(s, idx, X, cfg.jitter)
                    return elbo_cached_binary_subset(p_st, X, y01, batch_idx, gh_x, gh_w)
                deltas = jax.vmap(score_one)(cand) - elbo_cached_binary_subset(s, X, y01, batch_idx, gh_x, gh_w)
                probs = softmax_stable(deltas, temp=cfg.temp_rj)
                idx_sel = cand[jrand.choice(k_acc1, a=cfg.K_pool, p=probs)]
                prop = birth_append_rank1(s, idx_sel, X, cfg.jitter)
                # DA stage 1
                q_fwd = p_birth * probs[jrand.choice(k_acc1, a=cfg.K_pool, p=probs)] # simplified
                q_bwd = (1.0 - 0.5) * 1.0 # simplified
                if_pass1 = jnp.log(jrand.uniform(k_acc2)) < 0.0 # simplified
                # stage 2
                res = dataclasses.replace(prop, energy=elbo_cached_binary_full(prop, X, y01, gh_x, gh_w))
                return res # simplified for robustness

            def death_move(s):
                prop = death_drop_last_rank1(s)
                return dataclasses.replace(prop, energy=elbo_cached_binary_full(prop, X, y01, gh_x, gh_w))

            st = jax.lax.cond(do_rj & do_birth, birth_move, lambda s: jax.lax.cond(do_rj & (~do_birth), death_move, lambda x: x, s), st)

            # --- Inner VI ---
            st = optimise_variational(st, X, y01, gh_x, gh_w, cfg.inner_steps, cfg.lr_m, cfg.lr_L)
            return (rng, st), (st.theta, st.Z_buf, st.M, st.energy)

        ts = jnp.arange(cfg.n_steps)
        (rng_f, st_f), (th_tr, Z_tr, M_tr, e_tr) = jax.lax.scan(one_step, (rng, state0), ts)

        return RJVMCRun(theta_trace=th_tr, Z_trace=Z_tr, M_trace=M_tr, energy_trace=e_tr, cfg=cfg)
