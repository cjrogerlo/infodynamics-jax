# -*- coding: utf-8 -*-
"""Particle Feynman (Annealed SMC + HMC rejuvenation) using infodynamics-jax.

Why your JAX version was *much* slower than GPyTorch:
  - You were effectively doing an N×N Cholesky per particle per anneal step
    by materialising S_ff and factoring (S_ff + σ² I). That is O(P*T*N^3).

This script uses the FITC/Woodbury identities so the dominant linear algebra
is M×M (inducing), matching the fast PyTorch baseline.

It also fixes NLPD: for SMC BMA, NLPD must be computed from the *Gaussian
mixture* predictive density (log-sum-exp), not from a moment-matched Gaussian.

Run from the repo root:
  python particle_feynman_infodynamics_jax.py

If `infodynamics_jax` is not installed, this script auto-adds the repo root to
`sys.path`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make local package importable when running as a script.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from infodynamics_jax.core.phi import Phi
from infodynamics_jax.gp.kernels import get as get_kernel
from infodynamics_jax.gp.kernels.params import KernelParams
from infodynamics_jax.inference.particle.annealed import AnnealedSMC, AnnealedSMCCFG


# ---------------------------------------------------------------------------
# 1) Data
# ---------------------------------------------------------------------------

def make_1d_data(key, f: Callable[[jnp.ndarray], jnp.ndarray], x_min=-3.0, x_max=3.0, n=300, noise_std=0.2):
    x = jnp.linspace(x_min, x_max, n)[:, None]
    y_clean = f(x)[:, 0]
    y = y_clean + noise_std * jax.random.normal(key, (n,))
    return x, y, y_clean


# ---------------------------------------------------------------------------
# 2) FITC Gaussian log-evidence + prediction via Woodbury (M×M only)
# ---------------------------------------------------------------------------

def _chol(K: jnp.ndarray, jitter: float) -> jnp.ndarray:
    K = 0.5 * (K + K.T)
    return jnp.linalg.cholesky(K + jitter * jnp.eye(K.shape[0], dtype=K.dtype))


def fitc_log_evidence_rbf(
    kernel_fn: Callable,
    params: KernelParams,
    X: jnp.ndarray,
    y: jnp.ndarray,
    Z: jnp.ndarray,
    noise_var: jnp.ndarray,
    jitter: float,
) -> jnp.ndarray:
    """Return log p(y|X,phi) under FITC (Gaussian) using Woodbury.

    Matches the standard fast derivation:
      Σ = D + Q,  Q = Kxz Kzz^{-1} Kzx,  D = diag(Kxx - diag(Q)) + σ²I
    """
    y = y.reshape(-1)
    N = X.shape[0]

    Kzz = kernel_fn(params, Z, Z)
    Lz = _chol(Kzz, jitter)
    Kxz = kernel_fn(params, X, Z)  # (N,M)

    # V = Lz^{-1} Kzx = Lz^{-1} Kxz^T  => (M,N)
    V = jax.scipy.linalg.solve_triangular(Lz, Kxz.T, lower=True)

    diagQ = jnp.sum(V * V, axis=0)              # (N,)
    diagK = jnp.full((N,), params.variance)     # RBF: k(x,x)=variance
    d = jnp.clip(diagK - diagQ + noise_var, a_min=1e-12)

    A = V / jnp.sqrt(d)[None, :]                # (M,N)
    B = jnp.eye(A.shape[0], dtype=A.dtype) + A @ A.T  # (M,M)
    Lb = _chol(B, jitter)

    b = y / jnp.sqrt(d)
    Ab = A @ b
    c = jax.scipy.linalg.cho_solve((Lb, True), Ab)

    quad = jnp.dot(b, b) - jnp.dot(c, Ab)
    logdet = jnp.sum(jnp.log(d)) + 2.0 * jnp.sum(jnp.log(jnp.diag(Lb)))
    const = N * jnp.log(2.0 * jnp.pi)
    return -0.5 * (quad + logdet + const)


def fitc_posterior_predict_rbf(
    kernel_fn: Callable,
    params: KernelParams,
    X: jnp.ndarray,
    y: jnp.ndarray,
    Z: jnp.ndarray,
    noise_var: jnp.ndarray,
    X_star: jnp.ndarray,
    jitter: float,
    include_obs_noise: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return predictive mean/variance at X_star under FITC (Gaussian)."""
    y = y.reshape(-1)
    N = X.shape[0]

    Kzz = kernel_fn(params, Z, Z)
    Lz = _chol(Kzz, jitter)
    Kxz = kernel_fn(params, X, Z)  # (N,M)

    V = jax.scipy.linalg.solve_triangular(Lz, Kxz.T, lower=True)  # (M,N)
    diagQ = jnp.sum(V * V, axis=0)
    diagK = jnp.full((N,), params.variance)
    d = jnp.clip(diagK - diagQ + noise_var, a_min=1e-12)

    A = V / jnp.sqrt(d)[None, :]
    B = jnp.eye(A.shape[0], dtype=A.dtype) + A @ A.T
    Lb = _chol(B, jitter)

    # alpha = Σ^{-1} y, with Σ = diag(d) + V^T V
    b = y / jnp.sqrt(d)
    Ab = A @ b
    c = jax.scipy.linalg.cho_solve((Lb, True), Ab)
    alpha = (y / d) - (A.T @ c) / jnp.sqrt(d)

    # cross: q_* = Q_{X, *} = V^T v_*, where v_* = Lz^{-1} K_{Z,*}
    Ksz = kernel_fn(params, X_star, Z)          # (N*,M)
    v_star = jax.scipy.linalg.solve_triangular(Lz, Ksz.T, lower=True)  # (M,N*)
    q_star = V.T @ v_star                        # (N,N*)

    mean = (q_star.T @ alpha)                    # (N*,)

    # variance: k** - q_*^T Σ^{-1} q_*
    def _sigma_inv_times_q(q_col):
        t = q_col / d
        tmp = A @ (q_col / jnp.sqrt(d))
        w = jax.scipy.linalg.cho_solve((Lb, True), tmp)
        return t - (A.T @ w) / jnp.sqrt(d)

    sigma_inv_q = jax.vmap(_sigma_inv_times_q, in_axes=1, out_axes=1)(q_star)  # (N,N*)
    qT_Sinv_q = jnp.sum(q_star * sigma_inv_q, axis=0)  # (N*,)

    k_ss = jnp.full((X_star.shape[0],), params.variance)
    var_latent = jnp.clip(k_ss - qT_Sinv_q, a_min=1e-12)
    var = var_latent + (noise_var if include_obs_noise else 0.0)
    return mean, var


# ---------------------------------------------------------------------------
# 3) Priors and annealed target: log π_β(φ) = log p(φ) + β log p(y|φ)
# ---------------------------------------------------------------------------

def log_prior_phi(phi: Phi) -> jnp.ndarray:
    # weakly-informative priors on log-lengthscale/log-variance/log-noise
    # (tune as needed)
    log_ls = jnp.log(phi.kernel_params.lengthscale)
    log_var = jnp.log(phi.kernel_params.variance)
    log_nv = jnp.log(phi.likelihood_params["noise_var"])
    mu = jnp.array([0.0, 0.0, jnp.log(0.2**2)])
    sd = jnp.array([1.0, 1.0, 0.5])
    z = jnp.array([log_ls, log_var, log_nv])
    return jnp.sum(-0.5 * ((z - mu) / sd) ** 2 - jnp.log(sd) - 0.5 * jnp.log(2.0 * jnp.pi))


def log_prior_Z(Z: jnp.ndarray, sigma_Z: float = 3.0) -> jnp.ndarray:
    return jnp.sum(-0.5 * (Z / sigma_Z) ** 2 - jnp.log(sigma_Z) - 0.5 * jnp.log(2.0 * jnp.pi))


def make_loglik_fn(kernel_fn: Callable):
    def _ll(phi: Phi, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        nv = jnp.asarray(phi.likelihood_params["noise_var"])
        return fitc_log_evidence_rbf(kernel_fn, phi.kernel_params, X, y, phi.Z, nv, phi.jitter)
    return _ll


def make_energy_fn(kernel_fn: Callable):
    ll = make_loglik_fn(kernel_fn)

    def energy(phi: Phi, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # energy = -log p(phi,Z) - log p(y|phi,Z)  (up to constant)
        return -(log_prior_phi(phi) + log_prior_Z(phi.Z) + ll(phi, X, y))

    return energy


# ---------------------------------------------------------------------------
# 4) Mixture NLPD (correct) and moment-matched mean/std (for plots)
# ---------------------------------------------------------------------------

def mixture_logpdf(y: jnp.ndarray, mus: jnp.ndarray, vars_: jnp.ndarray, logw: jnp.ndarray) -> jnp.ndarray:
    """log p(y) for 1D Gaussian mixture at each point.

    y: (N,)
    mus: (P,N)
    vars_: (P,N)
    logw: (P,) unnormalised ok
    returns: (N,)
    """
    logw = logw - jax.scipy.special.logsumexp(logw)
    y = y[None, :]
    vars_ = jnp.clip(vars_, a_min=1e-12)
    logN = -0.5 * (jnp.log(2.0 * jnp.pi * vars_) + (y - mus) ** 2 / vars_)
    return jax.scipy.special.logsumexp(logw[:, None] + logN, axis=0)


def moment_match(logw: jnp.ndarray, mus: jnp.ndarray, vars_: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logw = logw - jax.scipy.special.logsumexp(logw)
    w = jnp.exp(logw)[:, None]
    mean = jnp.sum(w * mus, axis=0)
    second = jnp.sum(w * (vars_ + mus**2), axis=0)
    var = jnp.clip(second - mean**2, a_min=1e-12)
    return mean, jnp.sqrt(var)


# ---------------------------------------------------------------------------
# 5) Run one benchmark function
# ---------------------------------------------------------------------------

@dataclass
class RunCFG:
    n: int = 300
    m: int = 25
    n_particles: int = 64
    n_steps: int = 24
    ess_threshold: float = 0.5
    noise_std: float = 0.2
    seed: int = 0


def run_one(spec_name: str, f: Callable[[jnp.ndarray], jnp.ndarray], x_min: float, x_max: float, cfg: RunCFG):
    key = jax.random.key(cfg.seed)
    key_data, key_split, key_Z, key_smc = jax.random.split(key, 4)

    X, y, y_clean = make_1d_data(key_data, f, x_min=x_min, x_max=x_max, n=cfg.n, noise_std=cfg.noise_std)

    # train/test split
    perm = jax.random.permutation(key_split, cfg.n)
    n_tr = int(0.8 * cfg.n)
    idx_tr, idx_te = perm[:n_tr], perm[n_tr:]
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]

    # initial inducing points from train
    idx_Z = jax.random.permutation(key_Z, n_tr)[: cfg.m]
    Z0 = X_tr[idx_Z]

    # initial phi (single)
    phi_init = Phi(
        kernel_params=KernelParams(lengthscale=jnp.array(1.0), variance=jnp.array(1.0)),
        Z=Z0,
        likelihood_params={"noise_var": jnp.array(cfg.noise_std**2)},
        jitter=1e-6,
    )

    kernel_fn = get_kernel("rbf")
    energy = make_energy_fn(kernel_fn)

    # particles init (vectorised)
    def init_particles_fn(key, n_particles):
        key_l, key_v, key_n, key_z = jax.random.split(key, 4)
        ls = jnp.exp(jax.random.normal(key_l, (n_particles,)) * 0.5)
        var = jnp.exp(jax.random.normal(key_v, (n_particles,)) * 0.5)
        nv = jnp.exp(jnp.log(cfg.noise_std**2) + jax.random.normal(key_n, (n_particles,)) * 0.5)
        Z = phi_init.Z[None, :, :] + 0.2 * jax.random.normal(key_z, (n_particles, *phi_init.Z.shape))

        def pack(ls_i, var_i, nv_i, Z_i):
            return Phi(
                kernel_params=KernelParams(lengthscale=ls_i, variance=var_i),
                Z=Z_i,
                likelihood_params={"noise_var": nv_i},
                jitter=phi_init.jitter,
            )

        return jax.vmap(pack)(ls, var, nv, Z)

    smc = AnnealedSMC(
        cfg=AnnealedSMCCFG(
            n_particles=cfg.n_particles,
            n_steps=cfg.n_steps,
            ess_threshold=cfg.ess_threshold,
            rejuvenation="hmc",
            rejuvenation_steps=3,
            step_size=5e-3,
            n_leapfrog=8,
            jit=True,
        )
    )

    energy_jit = jax.jit(lambda phi, X_, y_: energy(phi, X_, y_))

    smc_res = smc.run(
        energy=energy_jit,
        init_particles_fn=init_particles_fn,
        key=key_smc,
        energy_args=(X_tr, y_tr),
    )

    # predictions per particle
    def predict_particle(phi: Phi, X_star: jnp.ndarray):
        nv = jnp.asarray(phi.likelihood_params["noise_var"])
        return fitc_posterior_predict_rbf(kernel_fn, phi.kernel_params, X_tr, y_tr, phi.Z, nv, X_star, phi.jitter)

    mus_te, vars_te = jax.vmap(lambda p: predict_particle(p, X_te))(smc_res.particles)
    mus_plot, vars_plot = jax.vmap(lambda p: predict_particle(p, X))(smc_res.particles)

    # mixture metrics
    logp_te = mixture_logpdf(y_te.reshape(-1), mus_te, vars_te, smc_res.logw)
    nlpd = float(-jnp.mean(logp_te))

    mean_te, std_te = moment_match(smc_res.logw, mus_te, vars_te)
    rmse = float(jnp.sqrt(jnp.mean((mean_te - y_te.reshape(-1)) ** 2)))

    # plotting
    mean_plot, std_plot = moment_match(smc_res.logw, mus_plot, vars_plot)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].scatter(np.array(X_tr[:, 0]), np.array(y_tr), s=10, alpha=0.5, label="train")
    axs[0].scatter(np.array(X_te[:, 0]), np.array(y_te), s=10, alpha=0.5, label="test")
    axs[0].plot(np.array(X[:, 0]), np.array(y_clean), "k--", lw=2, label="true f")
    axs[0].plot(np.array(X[:, 0]), np.array(mean_plot), lw=2, label="SMC BMA mean")
    axs[0].fill_between(
        np.array(X[:, 0]),
        np.array(mean_plot - 2 * std_plot),
        np.array(mean_plot + 2 * std_plot),
        alpha=0.15,
        label="SMC ±2σ (moment-match)",
    )
    axs[0].set_title(f"{spec_name}: prediction")
    axs[0].legend(fontsize=8)

    axs[1].plot(np.array(smc_res.betas), marker="o")
    axs[1].set_title("β schedule")
    axs[1].set_xlabel("SMC step")

    axs[2].plot(np.array(smc_res.ess_trace), marker="o")
    axs[2].set_title("ESS")
    axs[2].set_xlabel("SMC step")

    plt.tight_layout()
    plt.show()

    print(f"[{spec_name}] RMSE={rmse:.4f} | NLPD(mixt)={nlpd:.4f}")
    print(f"SMC std range: [{float(std_plot.min()):.4f}, {float(std_plot.max()):.4f}]")


def main():
    cfg = RunCFG()

    specs = [
        (
            "smooth_sin_cos",
            lambda x: jnp.sin(2 * x) + 0.3 * jnp.cos(5 * x),
            -3.0,
            3.0,
        ),
        (
            "sin_plus_linear",
            lambda x: jnp.sin(3 * x) + 0.5 * x,
            -3.0,
            3.0,
        ),
        (
            "cubic_bump",
            lambda x: 0.3 * x**3 - x + 0.5 * jnp.sin(4 * x),
            -2.5,
            2.5,
        ),
    ]

    for name, f, a, b in specs:
        run_one(name, f, a, b, cfg)


if __name__ == "__main__":
    main()
