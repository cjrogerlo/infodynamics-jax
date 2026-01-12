# infodynamics_jax/energy/compose.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from .base import EnergyTerm


# -----------------------------
# Generic energy algebra
# -----------------------------

@dataclass(frozen=True)
class SumEnergy(EnergyTerm):
    """Sum of energy terms. Each term must accept the same call signature."""
    terms: Sequence[EnergyTerm]

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        out = jnp.array(0.0)
        for t in self.terms:
            out = out + t(*args, **kwargs)
        return out


@dataclass(frozen=True)
class WeightedEnergy(EnergyTerm):
    """Scalar-weighted energy term: w * E(·)."""
    term: EnergyTerm
    weight: float

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        return jnp.asarray(self.weight) * self.term(*args, **kwargs)


@dataclass(frozen=True)
class ConditionalEnergy(EnergyTerm):
    """Enable/disable a term via a predicate on the call arguments."""
    term: EnergyTerm
    predicate: Callable[..., bool]

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        flag = self.predicate(*args, **kwargs)
        return jax.lax.cond(
            flag,
            lambda _: self.term(*args, **kwargs),
            lambda _: jnp.array(0.0),
            operand=None,
        )


# -----------------------------
# Project-level target energy
# -----------------------------

@dataclass(frozen=True)
class TargetEnergy(EnergyTerm):
    """Top-level target energy used by inference.

    This is the *only* object inference should call.

    Signature:
        E(phi, X, Y, key=None, include_prior=True)

    - `phi`: structural hyperparameters (kernel params, inducing locations, likelihood params, ...)
    - `X`: inputs / latent states (supervised: observed; latent models: latent)
    - `Y`: observations / labels
    - `key`: PRNGKey, required only if an underlying term needs randomness (e.g. MC expected NLL)
    - `include_prior`: if False, skip the latent-state prior (supervised default)

    Notes:
    - We intentionally do *not* include hyperpriors here.
    - Hyperpriors on φ (kernel params, Z, likelihood params) are NOT energy terms.
      They can be added via `extra` parameter as regular functions, or handled in inference layer.
    - Annealing (beta) is handled by inference, either by (i) tempering log-likelihood in weights,
      or (ii) wrapping `inertial` with `WeightedEnergy` if you want energy-level tempering.
    """

    inertial: EnergyTerm
    prior: Optional[EnergyTerm] = None
    extra: Optional[Sequence[EnergyTerm]] = None

    def __call__(
        self,
        phi,
        X,
        Y,
        key: Optional[jax.random.KeyArray] = None,
        include_prior: bool = True,
    ) -> jnp.ndarray:
        E = self.inertial(phi, X, Y, key=key)

        if self.prior is not None and include_prior:
            # PriorEnergy is defined as a function of X only.
            E = E + self.prior(X)

        if self.extra:
            for t in self.extra:
                # Extra terms may depend on (phi, X, Y) in general.
                # Note: Hyperpriors on φ can be added here as functions (not EnergyTerm).
                E = E + t(phi, X, Y)

        return E