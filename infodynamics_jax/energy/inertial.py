# infodynamics_jax/energy/inertial.py
from __future__ import annotations

from dataclasses import dataclass
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


class InertialEnergy(EnergyTerm):
    """
    Data-dependent inertial energy.
    
    This term represents the *data-dependent* contribution only:
        E_inertial(phi; X, Y) = E_{q(f | phi)}[ -log p(y | f, phi) ]
    
    Any structural regularisers (e.g. KL terms, hyperpriors) should be
    composed elsewhere (e.g. via energy.compose or inference/optimisation).
    
    This class defines:
        E_inertial(phi; X, Y)
    
    It MAY involve an inner variational optimisation over η = (m_u, L_u),
    but exposes ONLY the resulting scalar energy.
    
    NOTE: This is a MODEL ENERGY, not an optimisation objective.
    For type-II / VGA inference, use inference/optimisation/vfe.py instead.
    """

    def __init__(
        self,
        kernel_fn,
        likelihood,
        cfg: InertialCFG,
        analytic_energy_fn: Optional[Callable] = None,
    ):
        self.kernel_fn = kernel_fn
        self.likelihood = likelihood
        self.cfg = cfg
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
        if estimator is None:
            # Auto-selection is allowed ONLY if it strictly improves correctness
            # (i.e. analytic marginalisation is available).
            if (
                getattr(self.likelihood, "supports_analytic_marginal", False)
                and self.analytic_energy_fn is not None
            ):
                estimator = "analytic"
            else:
                raise ValueError(
                    "InertialCFG.estimator is None but analytic path is unavailable. "
                    "Set estimator to 'gh' or 'mc', or provide `analytic_energy_fn` and a likelihood with "
                    "supports_analytic_marginal=True."
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

        state = VariationalState.initialise(phi, X)

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
        