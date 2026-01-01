"""
Gauss–Hermite collapsed inertial energy.

This module implements a *partially collapsed* inertial energy for
factorised, non-conjugate likelihoods, using one-dimensional
Gauss–Hermite quadrature.

Mathematical role
-----------------
For factorised likelihoods of the form

    p(y | f, phi) = ∏_i p(y_i | f_i, phi),

we define the inertial energy as

    E(phi) = E_{p(f | phi)}[ -log p(y | f, phi) ].

When the likelihood is non-conjugate, this expectation is approximated
via a variational Gaussian approximation

    q(f) = N(m, S),

and *collapsed analytically per marginal dimension*:

    E(phi) ≈ ∑_i E_{q(f_i)}[ -log p(y_i | f_i, phi) ].

Each one-dimensional expectation is evaluated using Gauss–Hermite
quadrature.

Design principles
-----------------
- This module does *not* construct or optimise q(f).
- This module does *not* depend on kernel structure or inducing points.
- Only marginal means and variances (m_i, S_ii) are required.
- The result is a scalar inertial energy suitable for outer optimisation
  or sampling over phi.

This corresponds to the classical "1D collapsed" treatment used in
variational GP classification and count models (e.g. Nickisch &
Rasmussen, Titsias, Hensman et al.).
"""

import jax.numpy as jnp
from jax.scipy.special import roots_hermite


def expected_inertial_energy_gh(
    y,
    log_prob_1d,
    likelihood_params,
    m,
    S_diag,
    num_points=20,
):
    """
    One-dimensional Gauss–Hermite collapsed inertial energy.

    Parameters
    ----------
    y : (N,)
        Observations.
    log_prob_1d : callable
        log_prob_1d(y_i, f_i, likelihood_params) -> scalar.
    likelihood_params : dict
        Likelihood hyperparameters (subset of phi).
    m : (N,)
        Marginal means of q(f).
    S_diag : (N,)
        Marginal variances of q(f).
    num_points : int
        Number of Gauss–Hermite quadrature points.

    Returns
    -------
    energy : scalar
        Approximated inertial energy.
    """
    xs, ws = roots_hermite(num_points)

    def one_dim(y_i, m_i, v_i):
        f = jnp.sqrt(2.0 * v_i) * xs + m_i
        lp = log_prob_1d(y_i, f, likelihood_params)
        return -jnp.sum(ws * lp) / jnp.sqrt(jnp.pi)

    return jnp.sum(
        jnp.vectorize(one_dim)(y, m, S_diag)
    )