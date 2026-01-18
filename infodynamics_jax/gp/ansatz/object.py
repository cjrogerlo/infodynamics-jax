# infodynamics_jax/gp/ansatz/object.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp

from ..kernels import get as get_kernel
from ..likelihoods import get as get_likelihood

from .expected import (
    VariationalState,
    expected_nll_factorised_gh,
    expected_nll_factorised_mc,
)
from .expected import expected_nll_gaussian_1d
from .gh import GaussHermite


Estimator = Literal["conjugate", "gh", "mc"]


@dataclass
class InertialEnergy:
    """
    Callable inertial energy object.

    This object represents the primitive quantity of the framework:

        E(phi)
        = E_{q(f|phi)}[-log p(y | f, phi)]

    All approximation choices (conjugate / GH / MC, full/diag covariance)
    are implementation details internal to this object.

    The inference layer (optimisation, MCMC, SMC, dynamics)
    must ONLY interact with this object via __call__.
    """

    kernel: str
    likelihood: str

    estimator: Estimator = "gh"

    # GH-specific
    gh_order: int = 20
    gh_dtype: jnp.dtype = jnp.float64

    # MC-specific
    mc_samples: int = 16

    def __post_init__(self):
        # Resolve kernel / likelihood once (pure functions)
        self.kernel_fn = get_kernel(self.kernel)
        self.likelihood_obj = get_likelihood(self.likelihood)

        # sanity checks
        if self.estimator not in ("conjugate", "gh", "mc"):
            raise ValueError(f"Unknown estimator: {self.estimator}")

        if self.estimator == "conjugate":
            if not hasattr(self.likelihood_obj, "is_gaussian") or not self.likelihood_obj.is_gaussian:
                raise ValueError(
                    "Conjugate estimator requested, but likelihood is not Gaussian."
                )

        if self.estimator == "gh": self.gh = GaussHermite(n=self.gh_order)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        phi,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        state: Optional[VariationalState] = None,
        *,
        key: Optional[jax.random.KeyArray] = None,
    ) -> jnp.ndarray:
        """
        Evaluate inertial energy E(phi).

        Parameters
        ----------
        phi:
            Structural parameters (kernel hyperparameters, inducing inputs,
            likelihood hyperparameters, etc.)

        X: (N,Q)
            Inputs.

        Y: (N,) or (N,D)
            Observations.

        state:
            VariationalState for q(u|phi).
            Required for non-conjugate estimators (gh / mc).

        key:
            PRNG key for MC estimator.

        Returns
        -------
        energy: scalar
            Inertial energy E(phi).
        """
        if self.estimator == "conjugate":
            return self._conjugate_energy(phi, X, Y)

        if state is None:
            raise ValueError("VariationalState must be provided for non-conjugate energy.")

        if self.estimator == "gh":
            return self._gh_energy(phi, X, Y, state)

        if self.estimator == "mc":
            if key is None:
                raise ValueError("PRNG key required for MC energy estimator.")
            return self._mc_energy(phi, X, Y, state, key)

        raise RuntimeError("Unreachable.")

    # ------------------------------------------------------------------
    # internal implementations
    # ------------------------------------------------------------------

    def _conjugate_energy(self, phi, X, Y) -> jnp.ndarray:
        """
        Gaussian likelihood, exact expectation.

        Uses:
            E_{N(mu,var)}[-log p(y|f,phi)]
        with mu=0, var=K_ff(phi).
        """
        # full GP prior covariance
        Kff = self.kernel_fn(X, X, phi.kernel_params)
        Kff = 0.5 * (Kff + Kff.T)

        var = jnp.diag(Kff)
        mu = jnp.zeros_like(var)

        # broadcast Y if needed
        Yb = Y if Y.ndim == 2 else Y[:, None]

        def one_dim(y, m, v):
            # Gaussian case: use analytic solution (no ansatz needed)
            noise_var = phi.likelihood_params.get("noise_var", jnp.array(0.1))
            return expected_nll_gaussian_1d(y, m, v, noise_var)

        vals = jax.vmap(
            lambda yrow, m, v: jnp.sum(jax.vmap(one_dim)(yrow, m, v)),
            in_axes=(0, 0, 0),
        )(Yb, mu[:, None], var[:, None])

        return jnp.sum(vals)

    def _gh_energy(self, phi, X, Y, state: VariationalState) -> jnp.ndarray:
        """
        Rao–Blackwellised inertial energy using Gauss–Hermite quadrature.
        """
        nll_1d = self.likelihood_obj.neg_loglik_1d

        return expected_nll_factorised_gh(
            phi=phi,
            X=X,
            Y=Y,
            kernel_fn=self.kernel_fn,
            state=state,
            nll_1d_fn=nll_1d,
            gh=self.gh,
        )

    def _mc_energy(
        self,
        phi,
        X,
        Y,
        state: VariationalState,
        key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """
        Rao–Blackwellised inertial energy using Monte Carlo estimator.
        """
        nll_1d = self.likelihood_obj.neg_loglik_1d

        return expected_nll_factorised_mc(
            phi=phi,
            X=X,
            Y=Y,
            kernel_fn=self.kernel_fn,
            state=state,
            nll_1d_fn=nll_1d,
            key=key,
            n_samples=self.mc_samples,
        )
