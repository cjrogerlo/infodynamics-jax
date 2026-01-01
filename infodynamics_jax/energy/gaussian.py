# infodynamics_jax/energy/gaussian.py
from __future__ import annotations

import jax.numpy as jnp


def gaussian_expected_nll_1d(y, mu, var, phi) -> jnp.ndarray:
    """
    For y | f,phi ~ N(f, noise_var(phi)):

        E_{N(mu,var)}[ -log p(y|f,phi) ]
      = 0.5 * [ log(2π σ^2) + ((y-mu)^2 + var)/σ^2 ].

    IMPORTANT: σ^2 is part of phi (hyperparameter), do not omit phi.
    """
    sigma2 = phi.noise_var
    sigma2 = jnp.asarray(sigma2)
    return 0.5 * (jnp.log(2.0 * jnp.pi * sigma2) + ((y - mu) ** 2 + var) / sigma2)
