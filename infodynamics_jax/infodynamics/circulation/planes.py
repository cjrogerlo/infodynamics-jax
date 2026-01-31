"""
Plane builders (bases) for antisymmetric circulation.

Design note:
For large d (e.g. images flattened), particle count P is usually much smaller.
We therefore prefer the Gram trick to avoid an SVD on (P x d).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .operator import Planes

Array = jnp.ndarray


def stage_schedule(lam) -> jnp.ndarray:
    """Mid-anneal peak for stage parameter lam; zero at endpoints."""
    lam = jnp.asarray(lam)
    return lam * (1.0 - lam)


# Backward-compatible alias
def beta_schedule(beta) -> jnp.ndarray:
    return stage_schedule(beta)


@dataclass(frozen=True)
class PCAPlanes:
    """
    Stage-frozen PCA planes built from a particle cloud.

    This is a convenient wrapper around Planes(q1, q2) plus the mean used
    for centering (useful for diagnostics).
    """

    planes: Planes
    mean: Array  # (d,)

    @classmethod
    def from_particles(cls, particles: Array, k: int) -> "PCAPlanes":
        """
        Build leading PCA planes from a stack of particles.

        Args:
            particles: (P, N, Q) or (P, d) array.
            k: number of planes (uses top 2k right singular vectors).
        """
        if particles.ndim == 3:
            p, n, q = particles.shape
            d = n * q
            x = particles.reshape(p, d)
        elif particles.ndim == 2:
            p, d = particles.shape
            x = particles
        else:
            raise ValueError(f"particles must have ndim 2 or 3, got {particles.ndim}")

        mu = jnp.mean(x, axis=0)
        xc = x - mu

        # Gram trick: top right singular vectors of XC (P x d) recovered from
        # an SVD of G = XC XC^T (P x P).
        #
        # On METAL, JAX SVD/Eigh lowering is incomplete. Fall back to NumPy
        # on host for robustness (this path is not JIT-friendly).
        gram = xc @ xc.T  # (P, P)
        if jax.default_backend() == "metal":
            gram_np = np.asarray(gram)
            u_np, lam_np, _vt = np.linalg.svd(gram_np, full_matrices=False)
            u = jnp.asarray(u_np, dtype=xc.dtype)
            lam = jnp.asarray(lam_np, dtype=xc.dtype)
        else:
            u, lam, _vt = jnp.linalg.svd(gram, full_matrices=False)  # lam sorted desc

        r = min(2 * int(k), int(u.shape[1]))
        u = u[:, :r]
        lam = lam[:r]

        s = jnp.sqrt(jnp.maximum(lam, 0.0))  # singular values of XC
        vr = xc.T @ u
        denom = jnp.where(s[None, :] > 0, s[None, :], 1.0)
        dirs = vr / denom  # (d, r), approx orthonormal

        # Split into paired plane directions.
        q1 = dirs[:, 0::2]
        q2 = dirs[:, 1::2]
        # If r is odd (rare), pad q2 with zeros.
        if q2.shape[1] < q1.shape[1]:
            pad = jnp.zeros((dirs.shape[0], 1), dtype=dirs.dtype)
            q2 = jnp.concatenate([q2, pad], axis=1)

        planes = Planes(q1=q1, q2=q2)
        return cls(planes=planes, mean=mu)

    def apply(self, v: Array, omega: Array) -> Array:
        """Compute C v using the stored planes (no schedule scaling)."""
        from .operator import apply_Cv  # local import to keep module graph simple

        return apply_Cv(self.planes, omega, v)


def canonical_pairs(q_dim: int, k: int | None = None) -> Tuple[Tuple[int, int], ...]:
    """
    Canonical coordinate index pairs (i, j) for 2D planes in R^{q_dim}.

    This is a safe baseline when you want a fixed, interpretable basis.
    """
    pairs = [(i, j) for i in range(int(q_dim)) for j in range(i + 1, int(q_dim))]
    if k is None:
        return tuple(pairs)
    return tuple(pairs[: int(k)])
