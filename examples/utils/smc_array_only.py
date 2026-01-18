# Array-only annealed SMC with HMC rejuvenation (JAX).
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax, random


def multinomial_resample_array(key, logw, n_particles: int) -> jnp.ndarray:
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    w = w / jnp.sum(w)
    return random.choice(key, n_particles, shape=(n_particles,), p=w)


def hmc_step_array(
    key,
    theta,
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    step_size: float,
    n_leapfrog: int,
    beta: float,
) -> jnp.ndarray:
    def U(q):
        return beta * energy_fn(q)

    grad_U = jax.grad(U)

    key_mom, key_u = random.split(key, 2)
    p0 = random.normal(key_mom, shape=theta.shape)
    q0 = theta

    def leapfrog(i, state):
        q, p = state
        p = p - step_size * grad_U(q)
        q = q + step_size * p
        return q, p

    p = p0 - 0.5 * step_size * grad_U(q0)
    q, p = lax.fori_loop(0, n_leapfrog, leapfrog, (q0, p))
    p = p - 0.5 * step_size * grad_U(q)
    p = -p

    H0 = U(q0) + 0.5 * jnp.sum(p0 ** 2)
    H1 = U(q) + 0.5 * jnp.sum(p ** 2)
    log_accept_ratio = -(H1 - H0)
    accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
    accept = jnp.log(random.uniform(key_u)) < log_accept_ratio
    return jnp.where(accept, q, q0), accept_prob


def annealed_smc_array(
    *,
    key,
    init_particles: jnp.ndarray,
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_steps: int,
    ess_threshold: float,
    step_size: float,
    n_leapfrog: int,
    rejuvenation_steps: int,
) -> dict:
    n_particles = init_particles.shape[0]
    logw = jnp.zeros(n_particles)
    logZ_est = 0.0
    betas = jnp.linspace(0.0, 1.0, n_steps + 1)

    def step_fn(carry, t):
        particles, logw, logZ_est, key = carry
        beta_prev = betas[t]
        beta_curr = betas[t + 1]
        delta_beta = beta_curr - beta_prev

        key, key_resample, key_rejuv = random.split(key, 3)

        energies = jax.vmap(energy_fn)(particles)
        delta_logw = -delta_beta * energies

        max_logw_prev = jnp.max(logw)
        w_prev = jnp.exp(logw - max_logw_prev)
        w_prev = w_prev / jnp.sum(w_prev)

        logw = logw + delta_logw

        max_logw = jnp.max(logw)
        w = jnp.exp(logw - max_logw)
        w = w / jnp.sum(w)
        ess = 1.0 / jnp.sum(w ** 2)
        U = jnp.sum(w * energies)
        var_E = jnp.sum(w * (energies - U) ** 2)

        weighted_delta = delta_logw + jnp.log(w_prev + 1e-10)
        max_weighted = jnp.max(weighted_delta)
        logZ_inc = max_weighted + jnp.log(jnp.sum(jnp.exp(weighted_delta - max_weighted)))
        logZ_est = logZ_est + logZ_inc

        # KL Divergence: D_KL(pi_{t-1} || pi_t)
        kl = delta_beta * jnp.sum(w_prev * energies) + logZ_inc

        def resample(particles, logw, key):
            indices = multinomial_resample_array(key, logw, n_particles)
            particles = particles[indices]
            logw = jnp.zeros_like(logw)
            return particles, logw

        do_resample = ess < ess_threshold * n_particles
        particles, logw = lax.cond(
            do_resample,
            lambda _: resample(particles, logw, key_resample),
            lambda _: (particles, logw),
            operand=None,
        )

        def rejuvenate_one(key_i, theta_i):
            def body_fn(_, state):
                key_c, q, p_acc = state
                key_c, subkey = random.split(key_c)
                q, prob = hmc_step_array(subkey, q, energy_fn, step_size, n_leapfrog, beta_curr)
                return key_c, q, p_acc + prob

            _, q, total_prob = lax.fori_loop(0, rejuvenation_steps, body_fn, (key_i, theta_i, 0.0))
            return q, total_prob / rejuvenation_steps

        keys_rejuv = random.split(key_rejuv, n_particles)
        particles, accept_probs = jax.vmap(rejuvenate_one)(keys_rejuv, particles)
        avg_accept = jnp.mean(accept_probs)

        C = (beta_curr ** 2) * var_E
        
        # Ensemble-averaged diagnostics
        theta_mean = jnp.sum(w[:, None] * particles, axis=0)
        theta_std = jnp.sqrt(jnp.sum(w[:, None] * (particles - theta_mean) ** 2, axis=0))
        
        return (particles, logw, logZ_est, key), (ess, logZ_est, U, C, kl, avg_accept, theta_mean, theta_std)

    (particles, logw, logZ_est, _), (ess_trace, logZ_trace, U_trace, C_trace, kl_trace, accept_trace, theta_mean_trace, theta_std_trace) = lax.scan(
        step_fn, (init_particles, logw, logZ_est, key), jnp.arange(n_steps)
    )

    return {
        "particles": particles,
        "logw": logw,
        "ess_trace": ess_trace,
        "logZ_est": logZ_est,
        "logZ_trace": logZ_trace,
        "U_trace": U_trace,
        "C_trace": C_trace,
        "kl_trace": kl_trace,
        "accept_trace": accept_trace,
        "theta_mean_trace": theta_mean_trace,
        "theta_std_trace": theta_std_trace,
        "betas": betas,
    }
from jax.scipy.special import logsumexp

def smc_predict_mixture(
    particles,
    logw,
    X_train,
    Y_train,
    X_test,
    Y_test,
    Z_shape,
    jitter,
    k_top=10,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Compute mixture predictive mean, variance, and NLPD for the top-weighted particles.
    """
    from infodynamics_jax.gp.predict import predict_typeii
    from infodynamics_jax.gp.kernels import rbf
    from infodynamics_jax.gp.kernels.params import KernelParams
    from infodynamics_jax.core import Phi
    
    # Normalize weights
    logw = logw - logsumexp(logw)
    w = jnp.exp(logw)
    
    # Sort and take top particles to reduce computation
    idx = jnp.argsort(logw)[::-1][:k_top]
    p_top = particles[idx]
    w_top = w[idx]
    w_top = w_top / jnp.sum(w_top)
    
    def predict_one(theta):
        log_ell, log_sf2, log_sn2 = theta[0], theta[1], theta[2]
        Z = theta[3:].reshape(Z_shape)
        params = KernelParams(lengthscale=jnp.exp(log_ell), variance=jnp.exp(log_sf2))
        phi = Phi(Z=Z, kernel_params=params, likelihood_params={'noise_var': jnp.exp(log_sn2)})
        return predict_typeii(phi, X_test, X_train, Y_train, rbf, residual='fitc')

    means, vars = jax.vmap(predict_one)(p_top)
    
    # Mixture mean and variance
    mix_mean = jnp.sum(w_top[:, None] * means, axis=0)
    # Correct mixture variance = weighted mean of variances + weighted variance of means
    mix_var = jnp.sum(w_top[:, None] * (vars + means**2), axis=0) - mix_mean**2
    mix_var = jnp.maximum(mix_var, 1e-12)
    
    # Mixture NLPD
    nlpd = compute_mixture_nlpd(Y_test, means, vars, w_top)
    
    return mix_mean, mix_var, nlpd

def compute_mixture_nlpd(y, means, vars, weights):
    """Compute NLPD for a mixture of Gaussians."""
    # log p(y|x) = log sum_i w_i * p(y | mu_i, var_i)
    log_probs = -0.5 * (jnp.log(2 * jnp.pi * vars) + (y[None, :] - means)**2 / vars)
    # log(sum exp(log_w + log_p))
    log_weights = jnp.log(weights + 1e-12)
    log_density = jax.scipy.special.logsumexp(log_weights[:, None] + log_probs, axis=0)
    return -jnp.mean(log_density)

def z_mpd_norm_from_particles(particles, logw, Z_shape, x_min, x_max):
    """
    Order parameter: Normalized mean pairwise distance of inducing points.
    """
    logw = logw - logsumexp(logw)
    w = jnp.exp(logw)
    # Extract Z: shape [P, M*D] -> [P, M, D]
    Z = particles[:, 3:].reshape((particles.shape[0],) + Z_shape)
    z = Z[..., 0] # Use first dimension for distance in 1D problems
    
    def mpd(z_single):
        d = jnp.abs(z_single[:, None] - z_single[None, :])
        M = z_single.shape[0]
        return jnp.sum(jnp.triu(d, k=1)) / jnp.maximum(M * (M - 1) / 2.0, 1.0)
        
    mpds = jax.vmap(mpd)(z)
    scale = jnp.maximum(x_max - x_min, 1e-8)
    return float(jnp.sum(w * mpds) / scale)
