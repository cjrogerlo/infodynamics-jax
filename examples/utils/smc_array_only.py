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
    accept = jnp.log(random.uniform(key_u)) < -(H1 - H0)
    return jnp.where(accept, q, q0)


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

        weighted_delta = delta_logw + jnp.log(w_prev + 1e-10)
        max_weighted = jnp.max(weighted_delta)
        logZ_inc = max_weighted + jnp.log(jnp.sum(jnp.exp(weighted_delta - max_weighted)))
        logZ_est = logZ_est + logZ_inc

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
                key_c, q = state
                key_c, subkey = random.split(key_c)
                q = hmc_step_array(subkey, q, energy_fn, step_size, n_leapfrog, beta_curr)
                return key_c, q

            _, q = lax.fori_loop(0, rejuvenation_steps, body_fn, (key_i, theta_i))
            return q

        keys_rejuv = random.split(key_rejuv, n_particles)
        particles = jax.vmap(rejuvenate_one)(keys_rejuv, particles)

        return (particles, logw, logZ_est, key), ess

    (particles, logw, logZ_est, _), ess_trace = lax.scan(
        step_fn, (init_particles, logw, logZ_est, key), jnp.arange(n_steps)
    )

    return {
        "particles": particles,
        "logw": logw,
        "ess_trace": ess_trace,
        "logZ_est": logZ_est,
        "betas": betas,
    }
