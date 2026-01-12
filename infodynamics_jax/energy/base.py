# infodynamics_jax/energy/base.py
from __future__ import annotations

from typing import Protocol, runtime_checkable
import jax.numpy as jnp


@runtime_checkable
class EnergyTerm(Protocol):
    """
    Protocol for energy terms.

    Design principles
    -----------------
    - An EnergyTerm represents a scalar-valued energy functional.
    - It MUST be callable and return a scalar `jnp.ndarray` with shape ().
    - It MUST be side-effect free (pure function modulo RNG).
    - It MAY accept a PRNGKey if stochastic estimation is involved.

    Canonical conventions
    ---------------------
    We deliberately do NOT enforce a single call signature.
    Instead, the following conventions are assumed across the library:

    * PriorEnergy:
        E(X) -> scalar

    * InertialEnergy:
        E(phi, X, Y, key=None) -> scalar

    * TargetEnergy (composition):
        E(phi, X, Y, key=None, include_prior=True) -> scalar

    Inference algorithms MUST treat EnergyTerm as a black box
    and MUST NOT inspect or rely on any internal structure.
    """

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        """
        Compute energy.

        Returns
        -------
        jnp.ndarray
            Scalar energy (shape ()).

        Notes
        -----
        - Returning non-scalar values is a contract violation.
        - Any randomness must be controlled via an explicit PRNGKey.
        """
        ...
