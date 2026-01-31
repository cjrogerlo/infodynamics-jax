# infodynamics_jax/core/typing.py
from __future__ import annotations
from typing import Protocol, Any
from .upphi import Upphi

try:
    from jax import Array as JaxArray
except Exception:
    JaxArray = Any  # Fallback type for older JAX
Array = JaxArray


class Energy(Protocol):
    """
    Scalar energy functional on Upphi.

    Inference algorithms must treat this as a black box.
    """

    def __call__(self, phi: Upphi, data: Any) -> float:
        ...


class EnergyWithAux(Energy, Protocol):
    """
    Energy with auxiliary internal variables (e.g. eta).

    Aux variables are NOT part of Upphi and NOT part of inference state.
    """

    def aux(self, phi: Upphi, data: Any) -> dict:
        ...