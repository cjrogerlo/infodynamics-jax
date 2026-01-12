# infodynamics_jax/energy/gh.py
from __future__ import annotations
import jax.numpy as jnp
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Precomputed Gauss–Hermite (physicists') nodes and weights (GH-20)
# -----------------------------------------------------------------------------

_GH20_X = jnp.array([
    -5.387480890011232,
    -4.603682449550744,
    -3.944764040115625,
    -3.347854567383216,
    -2.788806058428130,
    -2.254974002089275,
    -1.738537712116586,
    -1.234076215395323,
    -0.737473728545394,
    -0.245340708300901,
     0.245340708300901,
     0.737473728545394,
     1.234076215395323,
     1.738537712116586,
     2.254974002089275,
     2.788806058428130,
     3.347854567383216,
     3.944764040115625,
     4.603682449550744,
     5.387480890011232,
])

_GH20_W = jnp.array([
    2.229393645534151e-13,
    4.399340992273180e-10,
    1.086069370769281e-07,
    7.802556478532063e-06,
    2.283386360163539e-04,
    3.243773342237861e-03,
    2.481052088746361e-02,
    1.090172060200233e-01,
    2.866755053628341e-01,
    4.622436696006100e-01,
    4.622436696006100e-01,
    2.866755053628341e-01,
    1.090172060200233e-01,
    2.481052088746361e-02,
    3.243773342237861e-03,
    2.283386360163539e-04,
    7.802556478532063e-06,
    1.086069370769281e-07,
    4.399340992273180e-10,
    2.229393645534151e-13,
])


@dataclass(frozen=True)
class GaussHermite:
    """
    Deterministic Gauss–Hermite quadrature helper.

    Provides expectations of the form:
        E_{N(mu, var)}[ -log p(y | f, phi) ]
    """
    n: int = 20

    def nodes_weights(self):
        if self.n != 20:
            raise NotImplementedError(
                "Only GH-20 is supported. Add more tables if needed."
            )
        return _GH20_X, _GH20_W

    def expect_nll_1d(self, y, mu, var, phi, nll_1d_fn):
        """
        Compute 1D expectation:
            E_{f ~ N(mu, var)}[ -log p(y | f, phi) ]

        Parameters
        ----------
        y : scalar
        mu : scalar
        var : scalar (>= 0)
        phi : Phi
        nll_1d_fn : callable
            Signature: nll_1d_fn(y, f, phi.likelihood_params) -> scalar
        """
        x, w = self.nodes_weights()

        # Transform GH nodes to N(mu, var)
        # Protect sqrt operation: ensure var is not too small to avoid numerical issues
        EPS_VAR = 1e-12
        var_safe = jnp.maximum(var, EPS_VAR)
        f = mu + jnp.sqrt(var_safe) * jnp.sqrt(2.0) * x

        # Evaluate negative log-likelihood
        vals = nll_1d_fn(y, f, phi.likelihood_params)

        # GH expectation formula
        return jnp.sum(w * vals) / jnp.sqrt(jnp.pi)

