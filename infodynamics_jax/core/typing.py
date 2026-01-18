# infodynamics_jax/core/typing.py
from __future__ import annotations
from typing import Protocol, Any
from .phi import Phi


class Energy(Protocol):
    """
    Scalar energy functional on Phi.

    Inference algorithms must treat this as a black box.
    """

    def __call__(self, phi: Phi, data: Any) -> float:
        ...


class EnergyWithAux(Energy, Protocol):
    """
    Energy with auxiliary internal variables (e.g. eta).

    Aux variables are NOT part of Phi and NOT part of inference state.
    """

    def aux(self, phi: Phi, data: Any) -> dict:
        ...