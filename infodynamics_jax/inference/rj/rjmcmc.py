# infodynamics_jax/inference/rj/rjmcmc.py
"""
RJMCMC (Reversible Jump MCMC) for Sparse GP with Conjugate (Gaussian) Likelihoods.

This module implements trans-dimensional MCMC sampling over the number of
inducing points using VFE for conjugate Gaussian likelihoods, with efficient
rank-1 updates for birth/death moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any, Literal, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random as jrand

from ...core import Upphi
from ...gp.kernels.params import KernelParams
from .state import RJState


@dataclass(frozen=True)
class RJMCMCCFG:
    """Configuration for RJMCMC sampler."""
    n_steps: int = 1000
    burn: int = 250
    M_min: int = 5
    M_max: int = 60
    M_init: int = 20
    K_pool: int = 32  # Number of candidates for birth move
    temp_birth: float = 1.0  # Temperature for birth candidate selection
    p_geom: float = 0.12  # Geometric prior parameter for M
    theta_every: int = 10  # Update hyperparameters every N steps
    hmc_step_size: float = 0.01
    hmc_steps: int = 5
    jitter: float = 1e-6


@dataclass
class RJMCMCRun:
    """RJMCMC run results."""
    elbos: jnp.ndarray  # ELBO trace
    Ms: jnp.ndarray  # M trace
    Thetas: jnp.ndarray  # Theta trace
    Zs: jnp.ndarray  # Z_buf trace
    M_init: jnp.ndarray  # Initial M value


# ============================================================
# Core VFE computation from cached statistics
# ============================================================

@jax.jit
def compute_elbo_from_cache(N, sn2, sf2, y2, logdetB, vnorm2, sumA2):
    """
    Compute VFE ELBO for regression from cached scalar statistics.
    Formula based on Titsias 2009.
    """
    return -0.5 * (
        N * jnp.log(2.0 * jnp.pi) +
        N * jnp.log(sn2) +
        logdetB +
        (y2 / sn2 - vnorm2) +
        (N * sf2 - sumA2) / sn2
    )


# ============================================================
# Build full state from scratch
# ============================================================

@partial(jax.jit, static_argnames=("kernel_fn",))
def build_full_state(
    theta: jnp.ndarray,
    Z_buf: jnp.ndarray,
    M: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel_fn: Callable,
    jitter: float = 1e-6
) -> RJState:
    """
    Build full RJState with all cached matrices from scratch.
    
    Args:
        theta: Hyperparameters (D+2,) = [log_ls (D,), log_sf, log_sn]
        Z_buf: Buffer of inducing point indices (M_max,)
        M: Number of active inducing points
        X: Input data (N, D)
        y: Output data (N, 1) or (N,)
        kernel_fn: Kernel function
        jitter: Jitter for numerical stability
    
    Returns:
        RJState with all cached matrices
    """
    N, D = X.shape
    log_ls, log_sf, log_sn = theta[:D], theta[D], theta[D+1]
    sn2 = jnp.exp(2.0 * log_sn)
    sf2 = jnp.exp(2.0 * log_sf)
    y = y.reshape(-1, 1)
    y2 = jnp.sum(y**2)
    
    M_val = M.astype(jnp.int32)
    M_max = Z_buf.shape[0]
    mask = (jnp.arange(M_max, dtype=jnp.int32) < M_val).astype(X.dtype)
    
    kernel_params = KernelParams(lengthscale=jnp.exp(log_ls), variance=sf2)
    Z_full = X[Z_buf]
    
    Kuu_raw = kernel_fn(Z_full, Z_full, kernel_params)
    Kuu = Kuu_raw * mask[:, None] * mask[None, :] + jnp.diag(1.0 - mask) + jitter * jnp.eye(M_max)
    Lm = jsp.linalg.cholesky(Kuu, lower=True)
    
    Kuf_raw = kernel_fn(Z_full, X, kernel_params)
    Kuf = Kuf_raw * mask[:, None]
    A = jsp.linalg.solve_triangular(Lm, Kuf, lower=True)
    
    # B = I + (1/sn2) * A @ A.T
    B = jnp.eye(M_max) + (1.0 / sn2) * (A @ A.T)
    LB = jsp.linalg.cholesky(B, lower=True)
    
    logdetB = 2.0 * jnp.sum(jnp.log(jnp.diag(LB)))
    
    # v = LB^{-1} (1/sn2) A y
    rhs = (1.0 / sn2) * (A @ y).reshape(-1)
    v = jsp.linalg.solve_triangular(LB, rhs[:, None], lower=True).reshape(-1)
    vnorm2 = jnp.sum(v**2)
    sumA2 = jnp.sum(A**2)
    
    elbo = compute_elbo_from_cache(N, sn2, sf2, y2, logdetB, vnorm2, sumA2)
    
    # Ensure jitter has same dtype as input data to avoid type mismatch in jax.lax.cond
    jitter_dtype = jnp.array(jitter, dtype=X.dtype)
    
    phi = Upphi(
        kernel_params=kernel_params,
        Z=Z_full,
        likelihood_params={"noise_var": sn2},
        jitter=jitter_dtype
    )
    
    return RJState(
        phi=phi,
        variational_state=None,
        M=M,
        Z_buf=Z_buf,
        energy=elbo,
        theta=theta,  # Set theta for HMC updates
        Lm=Lm,
        A=A,
        LB=LB,
        logdetB=logdetB,
        v=v,
        vnorm2=vnorm2,
        sumA2=sumA2,
        elbo=elbo
    )


# ============================================================
# Priors
# ============================================================

@jax.jit
def log_prior_theta(theta):
    """Log prior on hyperparameters theta."""
    lp_ls = -0.5 * jnp.sum(((theta[:-2] + 2.0) / 1.2) ** 2)
    lp_sf = -0.5 * (theta[-2] / 1.5) ** 2
    lp_sn = -0.5 * ((theta[-1] + 2.0) / 1.0) ** 2
    return lp_ls + lp_sf + lp_sn


@jax.jit
def log_prior_M_trunc_geom(M, M_min, M_max, p=0.12):
    """Truncated geometric prior on M."""
    m = M - M_min
    valid = (M >= M_min) & (M <= M_max)
    K = M_max - M_min + 1
    log_unn = jnp.log(p) + m * jnp.log1p(-p)
    logZ = jnp.log1p(-(1.0 - p) ** K)
    return jnp.where(valid, log_unn - logZ, -jnp.inf)


@jax.jit
def log_prior_ordered_Z_given_M(N, M):
    """Log prior on ordered inducing point indices."""
    Nf = jnp.array(N, dtype=jnp.float64)
    Mf = jnp.array(M, dtype=jnp.float64)
    return -(jsp.special.gammaln(Nf + 1.0) - jsp.special.gammaln(Nf - Mf + 1.0))


@jax.jit
def log_posterior(state: RJState, N, M_min, M_max, p_geom=0.12):
    """Log posterior using RJState."""
    theta = state.theta
    return (
        state.elbo
        + log_prior_theta(theta)
        + log_prior_M_trunc_geom(state.M, M_min, M_max, p=p_geom)
        + log_prior_ordered_Z_given_M(N, state.M)
    )


# ============================================================
# Rank-1 Birth/Death moves
# ============================================================

@partial(jax.jit, static_argnames=("kernel_fn",))
def birth_rank1_update(
    state: RJState,
    new_idx: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel_fn: Callable,
    jitter: float = 1e-6
) -> RJState:
    """
    Birth move using efficient rank-1 update.
    
    Args:
        state: Current RJState
        new_idx: Index of new inducing point
        X: Input data (N, D)
        y: Output data (N, 1)
        kernel_fn: Kernel function
        jitter: Jitter for numerical stability
    
    Returns:
        New RJState after birth move
    """
    N, D = X.shape
    y = y.reshape(-1, 1)
    y2 = jnp.sum(y * y)

    theta = state.theta
    log_ls = theta[:D]
    log_sf = theta[D]
    log_sn = theta[D + 1]
    sn2 = jnp.exp(2.0 * log_sn)
    sf2 = jnp.exp(2.0 * log_sf)

    M = state.M.astype(jnp.int32)
    M_max = state.Z_buf.shape[0]
    slot = M

    active = (jnp.arange(M_max, dtype=jnp.int32) < M).astype(X.dtype)

    z_new = X[new_idx].reshape(1, D)
    Z_full = X[state.Z_buf]
    kernel_params = state.phi.kernel_params
    km = kernel_fn(z_new, Z_full, kernel_params).reshape(-1) * active

    k_ss = sf2 + jitter
    l = jsp.linalg.solve_triangular(state.Lm, km[:, None], lower=True).reshape(-1)
    lam2 = k_ss - jnp.sum(l * l)
    lam = jnp.sqrt(jnp.maximum(lam2, 1e-12))

    Lm_new = state.Lm.at[slot, :].set(l)
    Lm_new = Lm_new.at[slot, slot].set(lam)

    kx = kernel_fn(z_new, X, kernel_params).reshape(-1)
    proj = l @ state.A
    a = (kx - proj) / lam
    A_new = state.A.at[slot, :].set(a)

    a2 = jnp.sum(a * a)

    b = (1.0 / sn2) * (state.A @ a[:, None]).reshape(-1)
    wb = jsp.linalg.solve_triangular(state.LB, b[:, None], lower=True).reshape(-1)

    beta = 1.0 + (1.0 / sn2) * a2
    schur = beta - jnp.sum(wb * wb)
    schur = jnp.maximum(schur, 1e-18)
    lamB = jnp.sqrt(schur)

    LB_new = state.LB.at[slot, :].set(wb)
    LB_new = LB_new.at[slot, slot].set(lamB)

    logdetB_new = state.logdetB + 2.0 * jnp.log(lamB)
    sumA2_new = state.sumA2 + a2

    r = (1.0 / sn2) * jnp.sum(a[:, None] * y)
    uTc = jnp.dot(wb, state.v)
    v_last = (r - uTc) / lamB

    v_new = state.v.at[slot].set(v_last)
    vnorm2_new = state.vnorm2 + v_last * v_last

    elbo_new = compute_elbo_from_cache(N, sn2, sf2, y2, logdetB_new, vnorm2_new, sumA2_new)

    Z_buf_new = state.Z_buf.at[slot].set(new_idx.astype(jnp.int32))
    M_new = (M + jnp.array(1, dtype=jnp.int32)).astype(jnp.int32)
    
    phi_new = Upphi(
        kernel_params=state.phi.kernel_params,
        Z=X[Z_buf_new],
        likelihood_params=state.phi.likelihood_params,
        jitter=state.phi.jitter
    )

    return RJState(
        phi=phi_new,
        variational_state=None,
        M=M_new,
        Z_buf=Z_buf_new,
        energy=elbo_new,
        theta=state.theta,  # Inherit theta from current state
        Lm=Lm_new,
        A=A_new,
        LB=LB_new,
        logdetB=logdetB_new,
        v=v_new,
        vnorm2=vnorm2_new,
        sumA2=sumA2_new,
        elbo=elbo_new
    )


@partial(jax.jit, static_argnames=("kernel_fn",))
def death_drop_last(
    state: RJState,
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel_fn: Callable
) -> RJState:
    """
    Death move (remove last inducing point).
    
    Args:
        state: Current RJState
        X: Input data (N, D)
        y: Output data (N, 1)
        kernel_fn: Kernel function (unused but kept for interface consistency)
    
    Returns:
        New RJState after death move
    """
    M = state.M.astype(jnp.int32)
    slot = (M - jnp.array(1, dtype=jnp.int32)).astype(jnp.int32)

    lamB = state.LB[slot, slot]
    v_last = state.v[slot]
    a = state.A[slot, :]
    a2 = jnp.sum(a * a)

    logdetB_new = state.logdetB - 2.0 * jnp.log(lamB)
    vnorm2_new = state.vnorm2 - v_last * v_last
    sumA2_new = state.sumA2 - a2

    # Zero out the removed slot
    Lm_new = state.Lm.at[slot, :].set(jnp.zeros_like(state.Lm[slot, :]))
    Lm_new = Lm_new.at[slot, slot].set(1.0)
    A_new = state.A.at[slot, :].set(jnp.zeros_like(state.A[slot, :]))
    LB_new = state.LB.at[slot, :].set(jnp.zeros_like(state.LB[slot, :]))
    LB_new = LB_new.at[slot, slot].set(1.0)
    v_new = state.v.at[slot].set(0.0)

    N, D = X.shape
    y = y.reshape(-1, 1)
    y2 = jnp.sum(y * y)

    theta = state.theta
    log_ls, log_sf, log_sn = theta[:D], theta[D], theta[D + 1]
    sn2 = jnp.exp(2.0 * log_sn)
    sf2 = jnp.exp(2.0 * log_sf)

    elbo_new = compute_elbo_from_cache(N, sn2, sf2, y2, logdetB_new, vnorm2_new, sumA2_new)
    M_new = (M - jnp.array(1, dtype=jnp.int32)).astype(jnp.int32)
    
    phi_new = Upphi(
        kernel_params=state.phi.kernel_params,
        Z=X[state.Z_buf],
        likelihood_params=state.phi.likelihood_params,
        jitter=state.phi.jitter
    )

    return RJState(
        phi=phi_new,
        variational_state=None,
        M=M_new,
        Z_buf=state.Z_buf,
        energy=elbo_new,
        theta=state.theta,  # Inherit theta from current state
        Lm=Lm_new,
        A=A_new,
        LB=LB_new,
        logdetB=logdetB_new,
        v=v_new,
        vnorm2=vnorm2_new,
        sumA2=sumA2_new,
        elbo=elbo_new
    )


# ============================================================
# Birth candidate selection
# ============================================================

@jax.jit
def softmax_stable(x, temp=1.0):
    """Stable softmax."""
    z = x / temp
    z = z - jnp.max(z)
    e = jnp.exp(z)
    return e / (jnp.sum(e) + 1e-30)


@partial(jax.jit, static_argnames=("K_pool", "kernel_fn"))
def birth_pool_choose(
    key: jrand.PRNGKey,
    state: RJState,
    X: jnp.ndarray,
    y: jnp.ndarray,
    M_min: int,
    M_max: int,
    kernel_fn: Callable,
    K_pool: int = 32,
    temp: float = 1.0,
    p_geom: float = 0.12,
    jitter: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Choose birth candidate using library's kernel.
    
    Returns:
        (idx_new, q_choose): Index of chosen candidate and its proposal probability
    """
    N, D = X.shape
    y = y.reshape(-1, 1)

    M = state.M.astype(jnp.int32)
    M_max_buf = state.Z_buf.shape[0]
    slot_ok = M < jnp.array(M_max, dtype=jnp.int32)

    cand = jrand.randint(key, shape=(K_pool,), minval=0, maxval=N, dtype=jnp.int32)

    active_mask = (jnp.arange(M_max_buf, dtype=jnp.int32) < M)
    active = active_mask.astype(X.dtype)

    def is_dup(i):
        return jnp.any(jnp.where(active_mask, state.Z_buf == i, False))

    theta = state.theta
    log_ls, log_sf, log_sn = theta[:D], theta[D], theta[D + 1]
    sn2 = jnp.exp(2.0 * log_sn)
    sf2 = jnp.exp(2.0 * log_sf)

    lpM_curr = log_prior_M_trunc_geom(M, M_min, M_max, p=p_geom)
    lpM_prop = log_prior_M_trunc_geom(M + 1, M_min, M_max, p=p_geom)
    d_lpM = lpM_prop - lpM_curr

    d_lpz = log_prior_ordered_Z_given_M(N, M + 1) - log_prior_ordered_Z_given_M(N, M)

    Z_all = X[state.Z_buf]
    kernel_params = state.phi.kernel_params

    def score_one(i):
        valid = slot_ok & (~is_dup(i))
        z_new = X[i].reshape(1, D)

        km = kernel_fn(z_new, Z_all, kernel_params).reshape(-1) * active

        k_ss = sf2 + jitter
        l = jsp.linalg.solve_triangular(state.Lm, km[:, None], lower=True).reshape(-1)
        lam2 = k_ss - jnp.sum(l * l)
        lam = jnp.sqrt(jnp.maximum(lam2, 1e-12))

        kx = kernel_fn(z_new, X, kernel_params).reshape(-1)
        proj = l @ state.A
        a = (kx - proj) / lam
        a2 = jnp.sum(a * a)

        b = (1.0 / sn2) * (state.A @ a[:, None]).reshape(-1)
        wb = jsp.linalg.solve_triangular(state.LB, b[:, None], lower=True).reshape(-1)

        beta = 1.0 + (1.0 / sn2) * a2
        schur = beta - jnp.sum(wb * wb)
        schur = jnp.maximum(schur, 1e-18)

        d_logdetB = jnp.log(schur)

        r = (1.0 / sn2) * jnp.sum(a[:, None] * y)
        uTc = jnp.dot(wb, state.v)
        v_last = (r - uTc) / jnp.sqrt(schur)
        d_vnorm2 = v_last * v_last
        d_sumA2 = a2

        d_elbo = (-0.5) * (d_logdetB - d_vnorm2) + (0.5 / sn2) * d_sumA2
        d_lp = d_elbo + d_lpM + d_lpz

        return jnp.where(valid, d_lp, -jnp.inf)

    deltas = jax.vmap(score_one)(cand)
    probs = softmax_stable(deltas, temp=temp)

    key2 = jrand.split(key, 2)[1]
    j = jrand.choice(key2, a=K_pool, p=probs)
    idx_new = cand[j]
    q_choose = probs[j]
    return idx_new, q_choose


# ============================================================
# RJ step
# ============================================================

@partial(jax.jit, static_argnames=("K_pool", "kernel_fn"))
def rj_step(
    key: jrand.PRNGKey,
    state: RJState,
    X: jnp.ndarray,
    y: jnp.ndarray,
    M_min: int,
    M_max: int,
    kernel_fn: Callable,
    K_pool: int = 32,
    temp_birth: float = 1.0,
    p_geom: float = 0.12,
    jitter: float = 1e-6
) -> RJState:
    """
    RJ step using refactored functions.
    
    Returns:
        New RJState (may be same as input if rejected)
    """
    N, D = X.shape
    key, k_move, k_pool, k_acc = jrand.split(key, 4)

    M = state.M.astype(jnp.int32)
    p_birth = jnp.where(M <= M_min, 1.0,
              jnp.where(M >= M_max, 0.0, 0.5))
    do_birth = jrand.uniform(k_move) < p_birth

    lp_curr = log_posterior(state, N, M_min, M_max, p_geom=p_geom)

    def birth_branch():
        idx_new, q_choose = birth_pool_choose(
            k_pool, state, X, y, M_min, M_max,
            kernel_fn=kernel_fn,
            K_pool=K_pool, temp=temp_birth, p_geom=p_geom, jitter=jitter
        )
        prop = birth_rank1_update(state, idx_new, X, y, kernel_fn=kernel_fn, jitter=jitter)
        lp_prop = log_posterior(prop, N, M_min, M_max, p_geom=p_geom)

        M_prop = prop.M.astype(jnp.int32)
        p_birth_rev = jnp.where(M_prop <= M_min, 1.0,
                        jnp.where(M_prop >= M_max, 0.0, 0.5))
        p_death_rev = 1.0 - p_birth_rev

        q_fwd = p_birth * q_choose
        q_bwd = p_death_rev * 1.0

        loga = (lp_prop - lp_curr) + jnp.log(q_bwd + 1e-30) - jnp.log(q_fwd + 1e-30)
        acc = jnp.log(jrand.uniform(k_acc)) < jnp.minimum(0.0, loga)
        return jax.lax.cond(acc, lambda: prop, lambda: state)

    def death_branch():
        can = M > M_min
        def do_death_inner():
            prop = death_drop_last(state, X, y, kernel_fn=kernel_fn)
            lp_prop = log_posterior(prop, N, M_min, M_max, p_geom=p_geom)

            p_death = 1.0 - p_birth
            M_prop = prop.M.astype(jnp.int32)
            p_birth_rev = jnp.where(M_prop <= M_min, 1.0,
                            jnp.where(M_prop >= M_max, 0.0, 0.5))

            q_fwd = p_death * 1.0
            q_bwd = p_birth_rev * (1.0 / jnp.maximum(1.0, (N - M_prop).astype(jnp.float64)))

            loga = (lp_prop - lp_curr) + jnp.log(q_bwd + 1e-30) - jnp.log(q_fwd + 1e-30)
            acc = jnp.log(jrand.uniform(k_acc)) < jnp.minimum(0.0, loga)
            return jax.lax.cond(acc, lambda: prop, lambda: state)

        return jax.lax.cond(can, do_death_inner, lambda: state)

    return jax.lax.cond(do_birth, lambda: birth_branch(), lambda: death_branch())


# ============================================================
# HMC update (optional, requires blackjax)
# ============================================================

try:
    import blackjax
    BLACKJAX_AVAILABLE = True
except ImportError:
    BLACKJAX_AVAILABLE = False


@partial(jax.jit, static_argnames=("hmc_steps", "kernel_fn"))
def hmc_update_theta(
    key: jrand.PRNGKey,
    state: RJState,
    X: jnp.ndarray,
    y: jnp.ndarray,
    M_min: int,
    M_max: int,
    p_geom: float,
    hmc_step_size: float,
    hmc_steps: int,
    kernel_fn: Callable,
    jitter: float = 1e-6
) -> RJState:
    """
    HMC update for theta using blackjax.
    
    Args:
        key: PRNG key
        state: Current RJState
        X: Input data (N, D)
        y: Output data (N, 1)
        M_min, M_max: Bounds on M
        p_geom: Geometric prior parameter
        hmc_step_size: HMC step size
        hmc_steps: Number of HMC steps
        kernel_fn: Kernel function
        jitter: Jitter for numerical stability
    
    Returns:
        New RJState after HMC update
    """
    if not BLACKJAX_AVAILABLE:
        raise ImportError("blackjax is required for HMC updates. Install with: pip install blackjax")
    
    N, D = X.shape
    
    # Ensure jitter has same dtype as state to avoid type mismatch
    # Use state.phi.jitter's dtype if it's a JAX array, otherwise use X's dtype
    if hasattr(state.phi.jitter, 'dtype'):
        jitter_dtype = state.phi.jitter.dtype
    else:
        jitter_dtype = X.dtype
    jitter_typed = jnp.array(jitter, dtype=jitter_dtype)

    def logprob(theta):
        st = build_full_state(theta, state.Z_buf, state.M, X, y, kernel_fn=kernel_fn, jitter=jitter_typed)
        return log_posterior(st, N, M_min, M_max, p_geom=p_geom)

    hmc = blackjax.hmc(
        logprob,
        step_size=hmc_step_size,
        inverse_mass_matrix=jnp.ones_like(state.theta),
        num_integration_steps=hmc_steps,
    )
    s0 = hmc.init(state.theta)
    s1, _ = hmc.step(key, s0)
    theta_new = s1.position
    return build_full_state(theta_new, state.Z_buf, state.M, X, y, kernel_fn=kernel_fn, jitter=jitter_typed)


# ============================================================
# Main chain
# ============================================================

# Note: Not JIT-compiled because it returns RJMCMCRun dataclass
# Internal functions (one_step, rj_step, etc.) are still JIT-compiled
def run_chain(
    key: jrand.PRNGKey,
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel_fn: Callable,
    cfg: Optional[RJMCMCCFG] = None,
    *,
    n_steps: Optional[int] = None,
    M_min: Optional[int] = None,
    M_max: Optional[int] = None,
    M_init: Optional[int] = None,
    K_pool: Optional[int] = None,
    temp_birth: Optional[float] = None,
    p_geom: Optional[float] = None,
    theta_every: Optional[int] = None,
    hmc_step_size: Optional[float] = None,
    hmc_steps: Optional[int] = None,
    jitter: Optional[float] = None,
) -> RJMCMCRun:
    """
    Main MCMC chain using refactored functions with infodynamics-jax components.
    
    Args:
        key: PRNG key
        X: Input data (N, D)
        y: Output data (N, 1) or (N,)
        kernel_fn: Kernel function
        cfg: Optional RJMCMCCFG configuration. If provided, other parameters are ignored.
        n_steps: Number of MCMC steps (used if cfg is None)
        M_min, M_max: Bounds on number of inducing points (used if cfg is None)
        M_init: Initial number of inducing points (used if cfg is None)
        K_pool: Number of candidates for birth move (used if cfg is None)
        temp_birth: Temperature for birth candidate selection (used if cfg is None)
        p_geom: Geometric prior parameter (used if cfg is None)
        theta_every: Update hyperparameters every N steps (used if cfg is None)
        hmc_step_size: HMC step size (used if cfg is None)
        hmc_steps: Number of HMC steps (used if cfg is None)
        jitter: Jitter for numerical stability (used if cfg is None)
    
    Returns:
        RJMCMCRun with traces
    
    Note:
        If cfg is provided, it takes precedence over individual parameters.
        If cfg is None, individual parameters are used (with defaults from RJMCMCCFG).
    """
    # Use cfg if provided, otherwise use individual parameters or defaults
    if cfg is not None:
        n_steps_val = cfg.n_steps
        M_min_val = cfg.M_min
        M_max_val = cfg.M_max
        M_init_val = cfg.M_init
        K_pool_val = cfg.K_pool
        temp_birth_val = cfg.temp_birth
        p_geom_val = cfg.p_geom
        theta_every_val = cfg.theta_every
        hmc_step_size_val = cfg.hmc_step_size
        hmc_steps_val = cfg.hmc_steps
        jitter_val = cfg.jitter
    else:
        # Use provided parameters or defaults from RJMCMCCFG
        default_cfg = RJMCMCCFG()
        n_steps_val = n_steps if n_steps is not None else default_cfg.n_steps
        M_min_val = M_min if M_min is not None else default_cfg.M_min
        M_max_val = M_max if M_max is not None else default_cfg.M_max
        M_init_val = M_init if M_init is not None else default_cfg.M_init
        K_pool_val = K_pool if K_pool is not None else default_cfg.K_pool
        temp_birth_val = temp_birth if temp_birth is not None else default_cfg.temp_birth
        p_geom_val = p_geom if p_geom is not None else default_cfg.p_geom
        theta_every_val = theta_every if theta_every is not None else default_cfg.theta_every
        hmc_step_size_val = hmc_step_size if hmc_step_size is not None else default_cfg.hmc_step_size
        hmc_steps_val = hmc_steps if hmc_steps is not None else default_cfg.hmc_steps
        jitter_val = jitter if jitter is not None else default_cfg.jitter
    
    N, D = X.shape
    y = y.reshape(-1, 1)

    key, k0 = jrand.split(key, 2)
    perm = jrand.permutation(k0, N).astype(jnp.int32)

    Z_buf = jnp.zeros((M_max_val,), dtype=jnp.int32)
    Z_buf = Z_buf.at[:M_init_val].set(perm[:M_init_val])
    M0 = jnp.array(M_init_val, dtype=jnp.int32)

    theta0 = jnp.concatenate([
        jnp.full((D,), -2.0, dtype=jnp.float64),
        jnp.array([0.0, -2.0], dtype=jnp.float64)
    ])

    state0 = build_full_state(theta0, Z_buf, M0, X, y, kernel_fn=kernel_fn, jitter=jitter_val)

    def one_step(carry, t):
        key, state = carry
        key, k_th, k_rj = jrand.split(key, 3)

        do_theta = (t % theta_every_val) == 0
        state = jax.lax.cond(
            do_theta,
            lambda: hmc_update_theta(k_th, state, X, y, M_min_val, M_max_val, p_geom_val, 
                                     hmc_step_size_val, hmc_steps_val, kernel_fn=kernel_fn, jitter=jitter_val),
            lambda: state
        )

        state = rj_step(
            k_rj, state, X, y, M_min_val, M_max_val,
            kernel_fn=kernel_fn,
            K_pool=K_pool_val, temp_birth=temp_birth_val, p_geom=p_geom_val, jitter=jitter_val
        )

        return (key, state), (state.elbo, state.M, state.theta, state.Z_buf)

    (_, _), hist = jax.lax.scan(one_step, (key, state0), jnp.arange(n_steps_val, dtype=jnp.int32))
    elbos, Ms, Thetas, Zs = hist

    # prepend init at index 0
    elbos = jnp.concatenate([jnp.asarray([state0.elbo]), elbos], axis=0)
    Ms = jnp.concatenate([jnp.asarray([state0.M]), Ms], axis=0)
    Thetas = jnp.concatenate([jnp.asarray([state0.theta]), Thetas], axis=0)
    Zs = jnp.concatenate([jnp.asarray([state0.Z_buf]), Zs], axis=0)

    return RJMCMCRun(
        elbos=elbos,
        Ms=Ms,
        Thetas=Thetas,
        Zs=Zs,
        M_init=state0.M
    )


# ============================================================
# High-level interface (optional, for backward compatibility)
# ============================================================

class RJMCMC:
    """
    High-level interface for RJMCMC (for backward compatibility).
    
    This wraps the functional interface above.
    """
    
    def __init__(
        self,
        cfg: RJMCMCCFG = RJMCMCCFG(),
        kernel_fn: Optional[Callable] = None,
    ):
        """
        Initialize RJMCMC sampler.
        
        Args:
            cfg: Configuration for RJMCMC
            kernel_fn: Kernel function (required)
        """
        self.cfg = cfg
        self.kernel_fn = kernel_fn
        if kernel_fn is None:
            raise ValueError("kernel_fn must be provided for RJMCMC")
    
    def run(
        self,
        key: jrand.PRNGKey,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> RJMCMCRun:
        """
        Run RJMCMC sampling.
        
        Args:
            key: PRNG key
            X: Input data (N, D)
            y: Output data (N, 1) or (N,)
        
        Returns:
            RJMCMCRun with traces
        """
        return run_chain(
            key=key,
            X=X,
            y=y,
            kernel_fn=self.kernel_fn,
            cfg=self.cfg
        )
