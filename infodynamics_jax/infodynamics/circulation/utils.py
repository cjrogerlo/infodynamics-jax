"""
Utility helpers for circulation (pure reshaping / PCA plane building).

These functions do not compute gradients; they only reshape pytrees and
build plane bases from particle clouds.
"""
from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from .planes import PCAPlanes


def flatten_pytree_to_vec(pytree: Any) -> Tuple[jnp.ndarray, Tuple[Any, list[tuple], list[int]]]:
    """
    Flatten a pytree of arrays into a single 1D vector plus reconstruction spec.
    """
    leaves, treedef = tree_flatten(pytree)
    shapes = [x.shape for x in leaves]
    sizes = [int(jnp.size(x)) for x in leaves]
    if len(sizes) == 0:
        vec = jnp.zeros((0,), dtype=jnp.float32)
    else:
        vec = jnp.concatenate([x.reshape(-1) for x in leaves], axis=0)
    return vec, (treedef, shapes, sizes)


def unflatten_vec_to_pytree(vec: jnp.ndarray, spec: Tuple[Any, list[tuple], list[int]]) -> Any:
    """Inverse of flatten_pytree_to_vec."""
    treedef, shapes, sizes = spec
    if len(sizes) == 0:
        return tree_unflatten(treedef, [])
    splits = jnp.cumsum(jnp.array(sizes[:-1]))
    chunks = jnp.split(vec, splits, axis=0)
    leaves = [c.reshape(sh) for c, sh in zip(chunks, shapes)]
    return tree_unflatten(treedef, leaves)


def build_pca_planes_from_pytree_particles(particles: Any, k: int):
    """
    Flatten stacked particles [P, ...] to (P, d) and build PCA planes.

    Returns:
        planes: PCAPlanes object (stage-frozen)
        spec: flatten/unflatten spec for mapping vectors back to pytree
    """
    q0 = tree_map(lambda x: x[0], particles)
    _, spec = flatten_pytree_to_vec(q0)

    def flatten_one(q):
        v, _ = flatten_pytree_to_vec(q)
        return v

    xmat = jax.vmap(flatten_one)(particles)  # (P, d)
    planes = PCAPlanes.from_particles(xmat, k=k)
    return planes, spec

