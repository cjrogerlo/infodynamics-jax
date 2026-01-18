# infodynamics_jax/gp/kernels/params.py
from __future__ import annotations
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass(frozen=True)
class KernelParams:
    """
    Unified kernel hyperparameters as a pytree.

    Unused fields are allowed; kernels just ignore what they don't need.
    """

    # Core
    lengthscale: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))
    variance: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))

    # Matern / RQ / Periodic
    nu: jnp.ndarray = field(default_factory=lambda: jnp.array(1.5))
    alpha: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))
    period: jnp.ndarray = field(default_factory=lambda: jnp.array(1.0))

    # Linear / Polynomial
    offset: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))
    degree: jnp.ndarray = field(default_factory=lambda: jnp.array(2.0))

    # ---- pytree protocol ----
    def tree_flatten(self):
        children = (
            self.lengthscale,
            self.variance,
            self.nu,
            self.alpha,
            self.period,
            self.offset,
            self.degree,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            lengthscale=children[0],
            variance=children[1],
            nu=children[2],
            alpha=children[3],
            period=children[4],
            offset=children[5],
            degree=children[6],
        )


def RBFParams(
    *,
    log_amp: jnp.ndarray | None = None,
    log_len: jnp.ndarray | None = None,
    variance: jnp.ndarray | None = None,
    lengthscale: jnp.ndarray | None = None,
) -> KernelParams:
    """
    Backward-compatible factory for legacy RBFParams.

    Old API used log_amp/log_len; map them to variance/lengthscale.
    """
    if variance is not None and log_amp is not None:
        raise TypeError("Provide only one of log_amp or variance.")
    if lengthscale is not None and log_len is not None:
        raise TypeError("Provide only one of log_len or lengthscale.")

    if variance is None:
        variance = jnp.exp(log_amp) if log_amp is not None else jnp.array(1.0)
    if lengthscale is None:
        lengthscale = jnp.exp(log_len) if log_len is not None else jnp.array(1.0)

    return KernelParams(lengthscale=lengthscale, variance=variance)
