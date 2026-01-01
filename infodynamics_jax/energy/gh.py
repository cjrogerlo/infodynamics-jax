# infodynamics_jax/energy/gh.py
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GaussHermite:
    """
    Deterministic Gauss–Hermite quadrature for 1D expectations under N(mu, var).

    We use:
        E_{N(mu,var)}[ g(f) ] = 1/sqrt(pi) * sum_j w_j * g(mu + sqrt(2*var) * x_j)

    Notes:
    - Assumes var >= 0 (we clip in callers).
    - Pure JAX, jit/vmap-friendly.
    """
    n: int = 20
    dtype: jnp.dtype = jnp.float64

    def nodes_weights(self):
        # Use NumPy-backed Hermite nodes/weights through jax.numpy? Not available.
        # We hardcode via jnp.polynomial.hermite? Also not available.
        #
        # Practical approach: use jax.scipy.special.roots_hermite if present.
        # If your JAX doesn't have it, replace this with a precomputed table.
        try:
            from jax.scipy.special import roots_hermite  # type: ignore
            x, w = roots_hermite(self.n)
        except Exception as e:
            raise RuntimeError(
                "Gauss–Hermite nodes/weights not available in this JAX build. "
                "Either (i) add a precomputed GH table, or (ii) use MC estimator."
            ) from e

        x = x.astype(self.dtype)
        w = w.astype(self.dtype)
        return x, w

    def expect_nll_1d(self, y, mu, var, phi, nll_1d_fn):
        """
        Compute E_{N(mu,var)}[ nll_1d_fn(y,f,phi) ].

        Inputs are scalars (or broadcastable scalars).
        """
        x, w = self.nodes_weights()
        var = jnp.clip(var, a_min=0.0)
        f = mu + jnp.sqrt(2.0 * var) * x  # (n,)
        vals = jax.vmap(lambda ff: nll_1d_fn(y, ff, phi))(f)  # (n,)
        return (w @ vals) / jnp.sqrt(jnp.pi)
