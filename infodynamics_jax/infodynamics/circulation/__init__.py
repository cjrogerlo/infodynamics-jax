"""
Circulation (solenoidal) operators for nonreversible dynamics.

This package defines how an antisymmetric operator C is *formed* and applied
implicitly (no dense matrices). It does not compute model gradients; it only
acts on vectors provided by inference code.
"""

import jax.numpy as jnp

from .operator import Planes, apply_Cv
from .planes import PCAPlanes, beta_schedule, stage_schedule, canonical_pairs
from .gating import ConstantOmega, LinearTanhGate
from .stats import PrecomputedStats
from .stage import StageFrozenCirculation
from .utils import flatten_pytree_to_vec, unflatten_vec_to_pytree, build_pca_planes_from_pytree_particles
from .per_image import canonical_q_pairs, apply_Cv_per_image, make_cg_operator_y_only


def apply_skew_from_planes(v, basis, omega):
    """
    Back-compat helper for the older basis layout.

    Args:
        v: (d,)
        basis: (d, 2K) stacked [q1, q2, q3, q4, ...]
        omega: (K,)
    """
    basis = jnp.asarray(basis)
    d, two_k = basis.shape
    k_planes = min(int(jnp.asarray(omega).shape[0]), int(two_k // 2))
    if k_planes == 0:
        return jnp.zeros_like(v)
    q1 = basis[:, 0 : 2 * k_planes : 2]
    q2 = basis[:, 1 : 2 * k_planes : 2]
    planes = Planes(q1=q1, q2=q2)
    return apply_Cv(planes, omega, v)


def flatten_latent(x: jnp.ndarray):
    """Back-compat helper: flatten array -> (vec, shape)."""
    return x.reshape(-1), x.shape


def unflatten_latent(x_vec: jnp.ndarray, shape):
    """Back-compat helper: reshape vec -> shape."""
    return x_vec.reshape(shape)


def make_cg_operator(planes: PCAPlanes, omega, schedule_fn=None):
    """
    Back-compat helper returning C_beta v for array-shaped v.

    Signature: (_x, v, lam) -> C_lam v (when schedule_fn is provided).
    If schedule_fn is None, returns raw Cv (no schedule scaling).
    """

    def apply(_x, v, lam=1.0):
        v_vec = v.reshape(-1)
        cv = planes.apply(v_vec, omega)
        if schedule_fn is not None:
            cv = jnp.asarray(schedule_fn(lam)) * cv
        return cv.reshape(v.shape)

    return apply


# Backward-compatible alias (deprecated name)
make_curl_transform = make_cg_operator


__all__ = [
    "Planes",
    "apply_Cv",
    "PCAPlanes",
    "apply_skew_from_planes",
    "beta_schedule",
    "stage_schedule",
    "canonical_pairs",
    "flatten_latent",
    "unflatten_latent",
    "ConstantOmega",
    "LinearTanhGate",
    "PrecomputedStats",
    "StageFrozenCirculation",
    "canonical_q_pairs",
    "apply_Cv_per_image",
    "make_cg_operator_y_only",
    "make_cg_operator",
    "make_curl_transform",
    "flatten_pytree_to_vec",
    "unflatten_vec_to_pytree",
    "build_pca_planes_from_pytree_particles",
]
