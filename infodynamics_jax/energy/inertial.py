# infodynamics_jax/energy/inertial.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Literal, Callable

import jax
import jax.numpy as jnp
import optax

from .base import EnergyTerm
from ..gp.ansatz.expected import (
    expected_nll_factorised_gh,
    expected_nll_factorised_mc,
)
from ..gp.ansatz.state import VariationalState
from ..gp.ansatz.gh import GaussHermite
from ..gp.sparsify import SparsifiedKernel
from ..gp.utils import safe_cholesky


@dataclass
class InertialCFG:
    """Configuration for inertial (data-dependent) energy."""

    estimator: Optional[Literal["analytic", "gh", "mc"]] = None  # backend selector

    # --- MC backend ---
    n_mc_samples: int = 16

    # --- Inner (ansatz) profiling ---
    # 0 = frozen η; >0 = approximate profiling via inner optimisation
    inner_steps: int = 0
    inner_lr: float = 1e-2

    gh_n: int = 20


def _gaussian_collapsed_energy(phi, X, Y, *, kernel_fn: Callable, residual: str = "fitc") -> jnp.ndarray:
    """
    Collapsed energy computation for Gaussian likelihood.
    
    This computes the inertial energy for Gaussian likelihood using the exact
    collapsed (u already marginalised) posterior under the sparsified kernel.
    
    For Gaussian likelihood, the exact posterior under sparsified kernel S_ff is:
        μ(φ) = S_ff(φ) (S_ff(φ) + σ²I)^{-1} y
        Σ(φ) = S_ff(φ) - S_ff(φ) (S_ff(φ) + σ²I)^{-1} S_ff(φ)
    
    Then: E_{q*(f | phi)}[-log p(y | f, phi)] = sum_i E_{q*(f_i | phi)}[-log N(y_i; f_i, σ²)]
    
    Args:
        phi: Structural parameters (must have .Z, .kernel_params, .likelihood_params)
        X: Input locations (N, Q)
        Y: Observations (N,) or (N, D)
        kernel_fn: Kernel function kernel_fn(X1, X2, kernel_params) -> (N1, N2)
        residual: Residual type ("fitc", "sor", "dtc")
    
    Returns:
        Scalar energy value.
    """
    Y = Y[:, None] if Y.ndim == 1 else Y  # (N, D)
    N, D = Y.shape
    
    # Extract parameters
    Z = phi.Z  # (M, Q)
    noise_var = phi.likelihood_params["noise_var"]  # scalar or (D,)
    noise_var = jnp.asarray(noise_var)
    if noise_var.ndim == 0:
        noise_var = noise_var[None]  # (1,)
    noise_var = jnp.broadcast_to(noise_var, (D,))  # (D,)
    
    # Get sparsified kernel S_ff = Q + R (u already marginalised)
    sk = SparsifiedKernel(kernel_fn=kernel_fn, residual=residual)
    S_ff = sk.S_ff(phi.kernel_params, X, Z, jitter=phi.jitter)  # (N, N)
    
    # For each output dimension, compute exact posterior and energy
    def energy_for_output_d(y_d, noise_var_d):
        """
        Compute energy for single output dimension using exact posterior under S_ff.
        
        Exact posterior under sparsified kernel:
            μ(φ) = S_ff(φ) (S_ff(φ) + σ²I)^{-1} y
            Σ(φ) = S_ff(φ) - S_ff(φ) (S_ff(φ) + σ²I)^{-1} S_ff(φ)
        
        Then compute: E_{q*(f_i)}[-log N(y_i; f_i, sigma^2)]
        """
        y_d = y_d[:, None]  # (N, 1)
        
        # S_ff + σ²I
        S_noise = S_ff + noise_var_d * jnp.eye(N, dtype=S_ff.dtype)
        
        # Cholesky of S_noise with safe jitter adjustment
        # Use a small jitter relative to noise_var to ensure numerical stability
        jitter = jnp.maximum(phi.jitter, noise_var_d * 1e-6)
        L_S = safe_cholesky(S_noise, jitter=jitter, max_jitter=1e-2)
        
        # μ = S_ff (S_ff + σ²I)^{-1} y
        #   = S_ff @ solve(S_noise, y)
        mu = S_ff @ jax.scipy.linalg.cho_solve((L_S, True), y_d)  # (N, 1)
        mu = mu[:, 0]  # (N,)
        
        # Σ = S_ff - S_ff (S_ff + σ²I)^{-1} S_ff
        # Diagonal: diag(Σ) = diag(S_ff) - diag(S_ff @ solve(S_noise, S_ff))
        S_ff_solve_S_noise = jax.scipy.linalg.cho_solve((L_S, True), S_ff)  # (N, N)
        var_f = jnp.diag(S_ff) - jnp.sum(S_ff * S_ff_solve_S_noise, axis=1)  # (N,)
        var_f = jnp.clip(var_f, a_min=0.0)
        
        # Compute E_{q*(f_i)}[-log N(y_i; f_i, sigma^2)]
        # = 0.5 * (log(2*pi*sigma^2) + ((y_i - mu_i)^2 + var_i) / sigma^2)
        energy = 0.5 * (
            jnp.log(2.0 * jnp.pi * noise_var_d)
            + ((y_d[:, 0] - mu) ** 2 + var_f) / noise_var_d
        )
        return jnp.sum(energy)
    
    # Sum over output dimensions
    total_energy = jnp.sum(
        jax.vmap(energy_for_output_d, in_axes=(1, 0))(Y, noise_var)
    )
    
    return total_energy


class InertialEnergy(EnergyTerm):
    """
    Data-dependent inertial energy.
    
    This term represents the *data-dependent* contribution only:
        E_inertial(phi; X, Y) = E_{q(f | phi)}[ -log p(y | f, phi) ]
    
    Any structural regularisers (e.g. KL terms, hyperpriors) should be
    composed elsewhere (e.g. via energy.compose or inference/optimisation).
    
    This class defines:
        E_inertial(phi; X, Y)
    
    For Gaussian likelihood, automatically uses collapsed (analytic) computation
    where u is already marginalised. For non-Gaussian likelihoods, uses ansatz-based
    estimators (GH or MC) which may involve an inner variational optimisation over
    η = (m_u, L_u), but exposes ONLY the resulting scalar energy.
    
    NOTE: This is a MODEL ENERGY, not an optimisation objective.
    For type-II / VGA inference, use inference/optimisation/vfe.py instead.
    """

    def __init__(
        self,
        kernel_fn,
        likelihood,
        cfg: InertialCFG,
        analytic_energy_fn: Optional[Callable] = None,
        residual: str = "fitc",
    ):
        self.kernel_fn = kernel_fn
        self.likelihood = likelihood
        self.cfg = cfg
        self.residual = residual
        
        # For Gaussian likelihood, automatically use collapsed (analytic) path
        if analytic_energy_fn is None:
            if getattr(likelihood, "supports_analytic_marginal", False):
                # Gaussian likelihood: use collapsed computation directly
                analytic_energy_fn = partial(
                    _gaussian_collapsed_energy,
                    kernel_fn=kernel_fn,
                    residual=residual,
                )
        
        self.analytic_energy_fn = analytic_energy_fn
        # Gauss-Hermite quadrature instance for GH estimator
        self.gh = GaussHermite(n=int(cfg.gh_n))

    def __call__(self, phi, X, Y, key: Optional[jax.random.KeyArray] = None) -> jnp.ndarray:
        """
        Compute inertial energy for given (phi, X, Y).
        
        This is the model energy E_inertial(phi; X, Y), not an optimisation objective.
        """
        # --- Safe automatic backend selection ---
        estimator = self.cfg.estimator
        
        # For Gaussian likelihood, always use collapsed (analytic) path
        if getattr(self.likelihood, "supports_analytic_marginal", False):
            if self.analytic_energy_fn is None:
                raise ValueError(
                    "Gaussian likelihood detected but analytic_energy_fn is None. "
                    "This should not happen - please report this as a bug."
                )
            estimator = "analytic"
        elif estimator is None:
            # Auto-selection for non-Gaussian likelihoods
            raise ValueError(
                "InertialCFG.estimator is None. "
                "Set estimator to 'gh' or 'mc' for non-Gaussian likelihoods."
            )
        # --- Analytic backend: exact Gaussian evidence under sparsified kernel ---
        # This path MUST NOT depend on any ansatz/inner state.
        if estimator == "analytic":
            if self.analytic_energy_fn is None:
                raise ValueError(
                    "estimator='analytic' requires `analytic_energy_fn` implementing exact Gaussian evidence "
                    "under the chosen (possibly sparsified) prior."
                )
            # Try calling with kernel_fn as kwarg (for collapsed Gaussian evidence)
            # If it fails, fall back to (phi, X, Y) signature
            try:
                return self.analytic_energy_fn(phi, X, Y, kernel_fn=self.kernel_fn)
            except TypeError:
                # Function doesn't accept kernel_fn kwarg, use standard signature
                return self.analytic_energy_fn(phi, X, Y)

        # --- Ansatz-based backends (universal, for non-Gaussian likelihoods) ---
        state = self._solve_inner(phi, X, Y)

        if estimator == "gh":
            return expected_nll_factorised_gh(
                phi=phi,
                X=X,
                Y=Y,
                kernel_fn=self.kernel_fn,
                state=state,
                nll_1d_fn=self.likelihood.neg_loglik_1d,
                gh=self.gh,
            )

        if estimator == "mc":
            if key is None:
                raise ValueError("MC estimator requires PRNGKey.")
            return expected_nll_factorised_mc(
                phi=phi,
                X=X,
                Y=Y,
                kernel_fn=self.kernel_fn,
                state=state,
                nll_1d_fn=self.likelihood.neg_loglik_1d,
                key=key,
                n_samples=self.cfg.n_mc_samples,
            )

        raise ValueError(f"Unknown estimator: {estimator}")

    def _solve_inner(self, phi, X, Y) -> VariationalState:
        """Compute an approximate inner optimum η̂(φ).

        NOTE:
        - This defines a *profiled* inertial energy:
              E_inertial(φ) ≈ E(φ, η̂(φ)).
        - Finite `inner_steps` correspond to an approximation
          (not an exact energy).
        - Inner optimisation is deterministic and part of the
          energy definition, not the inference algorithm.
        """

        # Initialize VariationalState using ansatz method
        state = VariationalState.initialise(phi, X, Y)

        if self.cfg.inner_steps <= 0:
            return state

        def loss_fn(s: VariationalState) -> jnp.ndarray:
            # Inner profiling objective: always use GH (deterministic, differentiable).
            # This is *profiling* (approximate argmin), not sampling.
            return expected_nll_factorised_gh(
                phi=phi,
                X=X,
                Y=Y,
                kernel_fn=self.kernel_fn,
                state=s,
                nll_1d_fn=self.likelihood.neg_loglik_1d,
                gh=self.gh,
            )

        # JIT-friendly inner optimisation using optax + lax.scan.
        # NOTE: this is *profiling* (approximate argmin over η), not a sampler.
        tx = optax.sgd(self.cfg.inner_lr)
        opt_state = tx.init(state)

        def step(carry, _):
            s, os = carry
            val, grad = jax.value_and_grad(loss_fn)(s)
            updates, os = tx.update(grad, os, params=s)
            s = optax.apply_updates(s, updates)
            return (s, os), val

        (state, _), _vals = jax.lax.scan(step, (state, opt_state), xs=None, length=self.cfg.inner_steps)
        return state
        