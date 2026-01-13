#!/usr/bin/env python
# coding: utf-8

import os
import sys

# CRITICAL: Force CPU and Enable Float64 BEFORE any jax imports
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Setup paths
cwd = Path(os.getcwd())
repo_root = cwd
while repo_root.parent != repo_root:
    if (repo_root / 'infodynamics_jax').exists():
        break
    repo_root = repo_root.parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

examples_dir = repo_root / 'examples'
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from infodynamics_jax.core import Phi
from infodynamics_jax.gp.kernels import rbf
from infodynamics_jax.gp.kernels.params import KernelParams
from infodynamics_jax.gp.likelihoods import get as get_likelihood
from infodynamics_jax.gp.predict import predict_typeii
from infodynamics_jax.gp.sparsify import fitc_log_evidence
from infodynamics_jax.inference.particle import AnnealedSMC, AnnealedSMCCFG
from infodynamics_jax.inference.optimisation import TypeII, TypeIICFG
from infodynamics_jax.inference.optimisation.vfe import make_vfe_objective
from infodynamics_jax.infodynamics import make_hyperprior
from utils import synthetic, compute_metrics, setup_plot_style, COLORS, plot_with_uncertainty
from utils.smc_array_only import annealed_smc_array

# Ensure JAX config matches environment
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

matplotlib.use('module://matplotlib_inline.backend_inline')
setup_plot_style()

class CFG:
    N_train = 120
    N_test = 60
    noise_std = 0.2
    domain = (-2.5, 2.5)
    M = 20
    n_particles = 64
    n_steps = 24
    ess_threshold = 0.6
    rejuvenation = 'hmc'
    rejuvenation_steps = 2
    step_size = 0.02
    n_leapfrog = 8
    bma_samples = 80
    typeii_steps = 300
    typeii_lr = 1e-2

cfg = CFG()
DATASETS = ['sine_mix', 'step']

def unpack_state(theta, shape_z):
    log_ell, log_sf2, log_sn2 = theta[0], theta[1], theta[2]
    Z = theta[3:].reshape(shape_z)
    return log_ell, log_sf2, log_sn2, Z

def get_energy_fn(X, y, hyperprior_fn, shape_z, jitter):
    def energy_theta(theta):
        log_ell, log_sf2, log_sn2, Z = unpack_state(theta, shape_z)
        params = KernelParams(lengthscale=jnp.exp(log_ell), variance=jnp.exp(log_sf2))
        noise_var = jnp.exp(log_sn2)
        
        # FITC negative log evidence
        E_fitc = -fitc_log_evidence(
            kernel_fn=rbf,
            params=params,
            X=X,
            y=y,
            Z=Z,
            noise_var=noise_var,
            jitter=jitter,
        )
        
        # Add hyperprior
        phi = Phi(kernel_params=params, Z=Z, likelihood_params={'noise_var': noise_var}, jitter=jitter)
        return E_fitc + hyperprior_fn(phi)
    return energy_theta

def run_benchmark(dataset_name, key):
    print(f"\n{'='*20} Benchmark: {dataset_name} {'='*20}")
    fn, title, _, _ = synthetic.get(dataset_name)
    key_data, key_smc, key_init = jax.random.split(key, 3)
    
    # Data
    X_all, Y_all, _ = synthetic.sample(dataset_name, N=cfg.N_train + cfg.N_test, noise=cfg.noise_std, domain=cfg.domain, key=key_data)
    X_all = X_all[:, None]
    perm = jax.random.permutation(key_data, X_all.shape[0])
    X_tr, Y_tr = X_all[perm[:cfg.N_train]], Y_all[perm[:cfg.N_train]]
    X_te, Y_te = X_all[perm[cfg.N_train:]], Y_all[perm[cfg.N_train:]]
    X_plot = jnp.linspace(cfg.domain[0], cfg.domain[1], 240)[:, None]
    Y_plot = fn(X_plot[:, 0])

    # Hyperprior (Matching implicit SMC prior)
    # log-normal with std 0.5 -> lambda = 1/(0.5^2) = 4.0
    hyperprior_fn = make_hyperprior(
        kernel_log_lambda=4.0,
        kernel_fields=["lengthscale", "variance"],
        likelihood_log_lambda=4.0,
        likelihood_keys=["noise_var"],
        likelihood_log_mu={'noise_var': jnp.log(cfg.noise_std**2)}
    )

    # MAP-II Baseline
    typeii_cfg = TypeIICFG(steps=cfg.typeii_steps, lr=cfg.typeii_lr, optimizer='adam', jit=True)
    typeii = TypeII(cfg=typeii_cfg)
    vfe_obj = make_vfe_objective(kernel_fn=rbf, residual='fitc')
    
    # We need to wrap VFE with hyperprior for MAP-II
    def map_ii_obj(phi, X, y):
        return vfe_obj(phi, X, y) + hyperprior_fn(phi)
    
    Z0 = jnp.linspace(cfg.domain[0], cfg.domain[1], cfg.M)[:, None]
    phi_init = Phi(
        kernel_params=KernelParams(lengthscale=jnp.array(1.0), variance=jnp.array(1.0)),
        Z=Z0,
        likelihood_params={'noise_var': jnp.array(cfg.noise_std**2)},
        jitter=1e-6,
    )
    
    t0 = time.time()
    res_map = typeii.run(energy=map_ii_obj, phi_init=phi_init, energy_args=(X_tr, Y_tr))
    phi_map = res_map.phi
    t_map = time.time() - t0
    print(f"MAP-II completed in {t_map:.2f}s")

    # SMC
    energy_fn = get_energy_fn(X_tr, Y_tr, hyperprior_fn, Z0.shape, phi_init.jitter)
    
    def init_particles(k, n):
        kl, kv, kn, kz = jax.random.split(k, 4)
        log_l = jnp.log(phi_init.kernel_params.lengthscale) + jax.random.normal(kl, (n,)) * 0.5
        log_v = jnp.log(phi_init.kernel_params.variance) + jax.random.normal(kv, (n,)) * 0.5
        log_n = jnp.log(phi_init.likelihood_params['noise_var']) + jax.random.normal(kn, (n,)) * 0.5
        Z_noisy = Z0[None] + 0.2 * jax.random.normal(kz, (n, *Z0.shape))
        return jnp.concatenate([log_l[:,None], log_v[:,None], log_n[:,None], Z_noisy.reshape(n, -1)], axis=1)

    particles_init = init_particles(key_init, cfg.n_particles)
    t0 = time.time()
    smc_res = annealed_smc_array(
        key=key_smc,
        init_particles=particles_init,
        energy_fn=energy_fn,
        n_steps=cfg.n_steps,
        ess_threshold=cfg.ess_threshold,
        step_size=cfg.step_size,
        n_leapfrog=cfg.n_leapfrog,
        rejuvenation_steps=cfg.rejuvenation_steps,
    )
    t_smc = time.time() - t0
    print(f"SMC completed in {t_smc:.2f}s")

    # Metrics & Plotting
    # ... helper for BMA ...
    def predict_bma(particles, logw, X_star):
        w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
        mus, vars_ = [], []
        for i in range(len(w)):
            ll, lv, ln, Z = unpack_state(particles[i], Z0.shape)
            phi_i = Phi(KernelParams(jnp.exp(ll), jnp.exp(lv)), Z, {'noise_var': jnp.exp(ln)}, phi_init.jitter)
            m, v = predict_typeii(phi_i, X_star, X_tr, Y_tr, rbf, residual='fitc')
            mus.append(m); vars_.append(v)
        mus, vars_ = jnp.stack(mus), jnp.stack(vars_)
        mean_bma = (w[:, None] * mus).sum(axis=0)
        var_bma = (w[:, None] * (vars_ + mus**2)).sum(axis=0) - mean_bma**2
        return mean_bma, jnp.sqrt(jnp.maximum(var_bma, 1e-12)), mus, w

    smc_mean, smc_std, smc_curves, smc_weights = predict_bma(smc_res['particles'], smc_res['logw'], X_plot)
    map_mean, map_var = predict_typeii(phi_map, X_plot, X_tr, Y_tr, rbf, residual='fitc')
    map_std = jnp.sqrt(map_var)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax_pred = axes[0, 0]
    ax_pred.scatter(np.array(X_tr)[:, 0], np.array(Y_tr), s=8, alpha=0.4, label="Train", color="C7")
    ax_pred.scatter(np.array(X_te)[:, 0], np.array(Y_te), s=8, alpha=0.4, label="Test", color="C3")
    ax_pred.plot(np.array(X_plot)[:, 0], np.array(Y_plot), "k--", lw=2, label="True f")
    
    # MAP-II
    ax_pred.plot(np.array(X_plot)[:, 0], np.array(map_mean), color="C2", lw=2, label="MAP-II mean")
    ax_pred.fill_between(np.array(X_plot)[:, 0].flatten(), np.array(map_mean - 2*map_std), np.array(map_mean + 2*map_std), alpha=0.15, color="C2")
    
    # SMC
    ax_pred.plot(np.array(X_plot)[:, 0], np.array(smc_mean), color="C0", lw=2, label="SMC mean")
    ax_pred.fill_between(np.array(X_plot)[:, 0].flatten(), np.array(smc_mean - 2*smc_std), np.array(smc_mean + 2*smc_std), alpha=0.15, color="C0")
    
    # Raw curves
    idx_curves = jnp.argsort(smc_weights)[-10:]
    for i in idx_curves:
        ax_pred.plot(np.array(X_plot)[:, 0], np.array(smc_curves[i]), lw=0.5, alpha=0.3, color="C0")
    
    # Inducing
    for z in phi_map.Z: ax_pred.axvline(float(z), color="C2", ls=":", alpha=0.3)
    smc_Z_mean = (smc_weights[:, None, None] * smc_res['particles'][:, 3:].reshape(-1, *Z0.shape)).sum(axis=0)
    for z in smc_Z_mean: ax_pred.axvline(float(z), color="C0", ls="--", alpha=0.3)
    
    ax_pred.legend(fontsize=8); ax_pred.set_title(f"{dataset_name}: Predictive")

    # Hyperparameters
    p = smc_res['particles']
    axes[1,0].hist(np.array(jnp.exp(p[:,0])), bins=30, density=True, alpha=0.7, color="C0", label="SMC")
    axes[1,0].axvline(float(phi_map.kernel_params.lengthscale), color="C2", ls="--", label="MAP-II")
    axes[1,0].set_title("Lengthscale")
    
    axes[1,1].hist(np.array(jnp.exp(p[:,1])), bins=30, density=True, alpha=0.7, color="C0")
    axes[1,1].axvline(float(phi_map.kernel_params.variance), color="C2", ls="--")
    axes[1,1].set_title("Variance")
    
    axes[1,2].hist(np.array(jnp.exp(p[:,2])), bins=30, density=True, alpha=0.7, color="C0")
    axes[1,2].axvline(float(phi_map.likelihood_params['noise_var']), color="C2", ls="--")
    axes[1,2].axvline(cfg.noise_std**2, color="C3", ls="-.", label="True")
    axes[1,2].set_title("Noise Var")
    
    # ESS & Beta
    ax_eb = axes[0, 1]
    ax_eb.plot(np.array(smc_res['betas']), 'o-', color="C1", label="beta")
    ax_ess = ax_eb.twinx()
    ax_ess.plot(np.array(smc_res['ess_trace']), 'x--', color="C4", label="ESS")
    ax_eb.set_title("SMC Diagnostics")

    # Residuals
    m_te_smc, s_te_smc, _, _ = predict_bma(smc_res['particles'], smc_res['logw'], X_te)
    m_te_map, v_te_map = predict_typeii(phi_map, X_te, X_tr, Y_tr, rbf, residual='fitc')
    res_smc = (Y_te - m_te_smc) / s_te_smc
    res_map = (Y_te - m_te_map) / jnp.sqrt(v_te_map)
    axes[0, 2].hist(np.array(res_smc).flatten(), bins=20, density=True, alpha=0.5, label="SMC", color="C0")
    axes[0, 2].hist(np.array(res_map).flatten(), bins=20, density=True, alpha=0.5, label="MAP-II", color="C2")
    axes[0, 2].set_title("Test Residuals")
    axes[0, 2].legend()
    
    plt.tight_layout(); plt.show()
    
    # Metrics
    smc_metrics = compute_metrics(Y_te, m_te_smc, s_te_smc)
    map_metrics = compute_metrics(Y_te, m_te_map, jnp.sqrt(v_te_map))
    
    return {
        'SMC': {**smc_metrics, 'time': t_smc},
        'MAP-II': {**map_metrics, 'time': t_map}
    }

# Main loop
results = {}
key = jax.random.key(42)
for ds in DATASETS:
    key, subkey = jax.random.split(key)
    results[ds] = run_benchmark(ds, subkey)

# Print Table
print("\n" + "="*80)
print(f"{'Dataset':<15} | {'Method':<10} | {'RMSE':<8} | {'NLPD':<8} | {'Time (s)':<8}")
print("-" * 80)
for ds, res in results.items():
    for method in ['SMC', 'MAP-II']:
        m = res[method]
        print(f"{ds:<15} | {method:<10} | {m['rmse']:<8.4f} | {m['nlpd']:<8.4f} | {m['time']:<8.2f}")
print("="*80)
