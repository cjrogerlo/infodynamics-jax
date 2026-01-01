# infodynamics_jax/energy/object.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Callable

import jax.numpy as jnp

from .gaussian import inertial_energy_gaussian_closed_form
from .expected import expected_nll_factorised_gh, expected_nll_factorised_mc


Approx = Literal["closed_form", "gh", "mc"]


@dataclass(frozen=True)
class InertialEnergy:
    """
    Unified inertial energy callable.

    Conjugate Gaussian:
      E(phi) = E_{p(f|phi)}[-log p(y|f,phi)]  (closed form)

    Non-conjugate (factorised likelihood):
      Approximate with q(f_i|phi) marginals induced by q(u|phi),
      then sum 1D expectations via GH or MC.
    """
    likelihood: object
    approx: Approx = "gh"
    n_gh: int = 32
    n_mc: int = 32

    def __call__(self, *, key=None, y, K_ff=None, mu=None, var=None, phi=None):
        """
        Choose one of:
        - closed_form: needs (y, K_ff, phi)
        - gh        : needs (y, mu, var, phi)
        - mc        : needs (key, y, mu, var, phi)
        """
        if self.approx == "closed_form":
            if K_ff is None or phi is None:
                raise ValueError("closed_form requires K_ff and phi.")
            return inertial_energy_gaussian_closed_form(y, K_ff, phi)

        if self.approx == "gh":
            if mu is None or var is None or phi is None:
                raise ValueError("gh requires mu, var, phi.")
            return expected_nll_factorised_gh(y, mu, var, phi, self.likelihood, n_gh=self.n_gh)

        if self.approx == "mc":
            if key is None or mu is None or var is None or phi is None:
                raise ValueError("mc requires key, mu, var, phi.")
            return expected_nll_factorised_mc(key, y, mu, var, phi, self.likelihood, n_mc=self.n_mc)

        raise ValueError(f"Unknown approx: {self.approx}")