# infodynamics_jax/gp/ansatz/mc.py
"""
Monte Carlo estimator for non-conjugate likelihoods.

This module provides a Monte Carlo estimator for expectations of the form:
    E_{q(f_i|phi)}[ -log p(y_i | f_i, phi) ]

where q(f_i|phi) is a Gaussian marginal induced by q(u|phi).

Advantages:
    - No special functions or quadrature tables needed
    - Works universally for any likelihood
    - JIT/vmap-friendly

Disadvantages:
    - Variance in the estimate (requires more samples)
    - Requires PRNG key
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .state import VariationalState


@dataclass(frozen=True)
class MonteCarlo:
    """
    Monte Carlo estimator for expectations under Gaussian marginals.
    
    Provides expectations of the form:
        E_{N(mu, var)}[ -log p(y | f, phi) ]
    
    via sampling: f ~ N(mu, var), then averaging nll(y, f, phi).
    """
    n_samples: int = 16
    
    def expect_nll_1d(
        self,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        var: jnp.ndarray,
        phi,
        nll_1d_fn: Callable,
        key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """
        Compute 1D expectation via Monte Carlo:
            E_{f ~ N(mu, var)}[ -log p(y | f, phi) ]
        
        Parameters
        ----------
        y : scalar or (D,)
            Observation
        mu : scalar or (D,)
            Mean of q(f)
        var : scalar or (D,)
            Variance of q(f) (>= 0)
        phi : Phi
            Structural parameters
        nll_1d_fn : callable
            Signature: nll_1d_fn(y, f, phi.likelihood_params) -> scalar
        key : PRNG key
            For sampling f ~ N(mu, var)
        
        Returns
        -------
        scalar
            Expected negative log-likelihood
        """
        # Ensure mu and var are at least 1D
        mu = jnp.atleast_1d(mu)
        var = jnp.atleast_1d(var)
        y = jnp.atleast_1d(y)
        
        # Clip variance to avoid numerical issues
        var = jnp.clip(var, a_min=0.0)
        
        # Sample f ~ N(mu, var)
        # Shape: (n_samples, D)
        eps = jax.random.normal(key, shape=(self.n_samples, mu.shape[0]), dtype=mu.dtype)
        f_samps = mu[None, :] + jnp.sqrt(var[None, :]) * eps  # (n_samples, D)
        
        # Evaluate nll for each sample
        def nll_of_sample(f):
            # f: (D,), y: (D,)
            return jnp.sum(jax.vmap(lambda y_d, f_d: nll_1d_fn(y_d, f_d, phi.likelihood_params))(y, f))
        
        vals = jax.vmap(nll_of_sample)(f_samps)  # (n_samples,)
        return jnp.mean(vals)
    
    def expect_nll_factorised(
        self,
        phi,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        kernel_fn: Callable,
        state: VariationalState,
        nll_1d_fn: Callable,
        key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """
        Compute factorised expectation:
            sum_i E_{q(f_i|phi)}[ nll(y_i, f_i, phi) ]
        
        where q(f_i|phi) is induced by q(u|phi) via sparse GP conditional.
        
        This is the main entry point for MC estimation in InertialEnergy.
        
        Parameters
        ----------
        phi : Phi
            Structural parameters
        X : (N, Q)
            Input locations
        Y : (N,) or (N, D)
            Observations
        kernel_fn : callable
            Kernel function
        state : VariationalState
            Variational state q(u|phi)
        nll_1d_fn : callable
            Likelihood function: nll_1d_fn(y, f, phi.likelihood_params) -> scalar
        key : PRNG key
            For sampling f ~ q(f|phi)
        
        Returns
        -------
        scalar
            Sum of expected negative log-likelihoods
        """
        from .expected import qfi_from_qu_full, expected_nll_gaussian_1d, _as_2d
        
        Y = _as_2d(Y)
        
        if state.cov_type != "full" or state.L_u is None:
            raise ValueError("expected_nll_factorised_mc currently expects full-cov state with L_u.")
        
        # Compute marginals q(f_i|phi)
        mu_f, var_f = qfi_from_qu_full(phi, X, kernel_fn, state.m_u, state.L_u)  # (N, D)
        var_f = jnp.clip(var_f, a_min=0.0)
        
        # Fast path for Gaussian likelihood
        if nll_1d_fn.__name__ == "neg_loglik_1d" and hasattr(phi, "likelihood_params"):
            noise_var = phi.likelihood_params.get("noise_var", None)
            if noise_var is not None:
                return jnp.sum(
                    expected_nll_gaussian_1d(Y, mu_f, var_f, noise_var)
                )
        
        # MC estimation for non-conjugate likelihoods
        N, D = Y.shape
        
        # Sample f ~ q(f|phi) for all (N, D)
        # Shape: (n_samples, N, D)
        eps = jax.random.normal(key, shape=(self.n_samples, N, D), dtype=mu_f.dtype)
        f_samps = mu_f[None, :, :] + jnp.sqrt(var_f[None, :, :]) * eps  # (n_samples, N, D)
        
        # Evaluate nll for each sample
        def nll_of_sample(f):
            # f: (N, D), Y: (N, D)
            # Sum over all (N, D) pairs
            return jnp.sum(
                jax.vmap(
                    lambda yi, fi: jnp.sum(
                        jax.vmap(lambda yijd, fijd: nll_1d_fn(yijd, fijd, phi.likelihood_params))(yi, fi)
                    )
                )(Y, f)
            )
        
        vals = jax.vmap(nll_of_sample)(f_samps)  # (n_samples,)
        return jnp.mean(vals)
