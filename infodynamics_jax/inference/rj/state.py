# infodynamics_jax/inference/rj/state.py
"""
RJ-MCMC State for Sparse Bayesian GP with Conjugate and Non-Conjugate Likelihoods.

This module defines RJState, which extends the library's Phi and VariationalState
concepts for transdimensional MCMC sampling.

For conjugate likelihoods (Gaussian):
- Uses VFE directly
- Optional cached matrices (Lm, A) for efficient rank-1 updates

For non-conjugate likelihoods (Bernoulli, Poisson, etc.):
- Uses InertialEnergy + VariationalState
- Maintains variational posterior q(u) = N(m_u, S_u) over inducing variables
- Dynamic M: Number of active inducing points
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Any, Optional

from infodynamics_jax.core import Phi
from infodynamics_jax.gp.ansatz.state import VariationalState


@jax.tree_util.register_pytree_node_class
@dataclass
class RJState:
    """
    RJ-MCMC state for sparse Bayesian GP with conjugate and non-conjugate likelihoods.

    This integrates:
    1. Library's Phi (structural parameters)
    2. Library's VariationalState (q(u) posterior) - for non-conjugate case
    3. RJ-MCMC specific components (dynamic M, cached matrices)

    Attributes:
        phi: Library's Phi (kernel_params, Z, likelihood_params, jitter)
        variational_state: Library's VariationalState (m_u, L_u or s_u_diag)
                          Required for non-conjugate, None for conjugate
        M: Current number of active inducing points (scalar int32)
        Z_buf: Buffer of inducing point indices (M_max,) int32
               Only first M entries are active

        # Energy/objective value:
        energy: Current energy/negative ELBO (scalar)
                For conjugate: VFE value
                For non-conjugate: InertialEnergy value

        # Optional cached matrices for conjugate case:
        # (These are None for non-conjugate likelihoods)
        Lm: Optional Cholesky of K_ZZ + jitter (M_max, M_max)
        A: Optional K_ZZ^{-1} K_ZX (M_max, N)
    """

    phi: Phi                        # Library's Phi
    variational_state: Optional[VariationalState] = None  # For non-conjugate only
    M: jnp.ndarray = None           # () int32
    Z_buf: jnp.ndarray = None       # (M_max,) int32
    energy: jnp.ndarray = None      # () - current energy value (ELBO / VFE)
    theta: Optional[jnp.ndarray] = None   # (D+2,) flat hyperparameters

    # Cached GP factors (for rank-1 updates)
    Kuu: Optional[jnp.ndarray] = None     # (M_max, M_max)
    Lm: Optional[jnp.ndarray] = None      # (M_max, M_max) - Cholesky of Kuu
    Kuf: Optional[jnp.ndarray] = None     # (M_max, N)
    A: Optional[jnp.ndarray] = None       # (M_max, N) - solve(Lm, Kuf)
    A2: Optional[jnp.ndarray] = None      # (M_max, N) - solve(Lm.T, A) == Kuu^{-1} @ Kuf

    # Optional cached matrices (for conjugate case only)
    LB: Optional[jnp.ndarray] = None      # (M_max, M_max) - Cholesky of B = I + (1/sn2) * A @ A.T
    logdetB: Optional[jnp.ndarray] = None # () - log determinant of B
    v: Optional[jnp.ndarray] = None        # (M_max,) - solution vector for VFE
    vnorm2: Optional[jnp.ndarray] = None  # () - ||v||^2
    sumA2: Optional[jnp.ndarray] = None   # () - sum of squared A elements
    elbo: Optional[jnp.ndarray] = None    # () - VFE/ELBO value (alias for energy)

    def tree_flatten(self):
        """Flatten RJState into children and auxiliary data for PyTree."""
        children = (
            self.phi,
            self.variational_state,
            self.M,
            self.Z_buf,
            self.energy,
            self.theta,
            self.Kuu,
            self.Lm,
            self.Kuf,
            self.A,
            self.A2,
            self.LB,
            self.logdetB,
            self.v,
            self.vnorm2,
            self.sumA2,
            self.elbo,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten children and auxiliary data into RJState."""
        return cls(*children)

    @property
    def active_Z(self) -> jnp.ndarray:
        """Return active inducing point indices (first M entries of Z_buf)."""
        return self.Z_buf[:self.M]

    @property
    def active_Z_locations(self) -> jnp.ndarray:
        """Return actual inducing point locations (X[Z_buf[:M]])."""
        return self.phi.Z  # phi.Z should contain the active inducing points

    @property
    def is_conjugate(self) -> bool:
        """Check if this state is for conjugate (Gaussian) likelihood."""
        return self.variational_state is None
    

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for logging/debugging."""
        summary = {
            "M": int(self.M),
            "energy": float(self.energy),
            "lengthscale": float(self.phi.kernel_params.lengthscale),
            "variance": float(self.phi.kernel_params.variance),
            "is_conjugate": self.is_conjugate,
        }

        # Add likelihood params if present
        if "noise_var" in self.phi.likelihood_params:
            summary["noise_var"] = float(self.phi.likelihood_params["noise_var"])

        # Add variational state info for non-conjugate case
        if self.variational_state is not None:
            summary["m_u_norm"] = float(jnp.linalg.norm(self.variational_state.m_u))
            if self.variational_state.L_u is not None:
                summary["L_u_diag_mean"] = float(jnp.mean(jnp.diag(self.variational_state.L_u)))

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert RJState to dictionary for serialization."""
        return {
            "phi": {
                "kernel_params": {
                    "lengthscale": self.phi.kernel_params.lengthscale,
                    "variance": self.phi.kernel_params.variance,
                },
                "Z": self.phi.Z,
                "likelihood_params": self.phi.likelihood_params,
                "jitter": self.phi.jitter,
            },
            "variational_state": {
                "m_u": self.variational_state.m_u,
                "L_u": self.variational_state.L_u,
                "s_u_diag": self.variational_state.s_u_diag,
                "cov_type": self.variational_state.cov_type,
            } if self.variational_state is not None else None,
            "M": self.M,
            "Z_buf": self.Z_buf,
            "energy": self.energy,
            "theta": self.theta,
            "Kuu": self.Kuu,
            "Lm": self.Lm,
            "Kuf": self.Kuf,
            "A": self.A,
            "A2": self.A2,
            "LB": self.LB,
            "logdetB": self.logdetB,
            "v": self.v,
            "vnorm2": self.vnorm2,
            "sumA2": self.sumA2,
            "elbo": self.elbo,
        }
