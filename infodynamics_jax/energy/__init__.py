# infodynamics_jax/energy/__init__.py
from __future__ import annotations

from .base import EnergyTerm
from .compose import SumEnergy, WeightedEnergy, ConditionalEnergy, TargetEnergy
from .prior import PriorEnergy
from .inertial import InertialEnergy, InertialCFG, InertialEnergyCFG
from .vfe import VFEEnergy, titsias_vfe_energy

__all__ = [
    "EnergyTerm",
    "SumEnergy",
    "WeightedEnergy",
    "ConditionalEnergy",
    "TargetEnergy",
    "PriorEnergy",
    "InertialEnergy",
    "InertialCFG",
    "InertialEnergyCFG",
    "VFEEnergy",
    "titsias_vfe_energy",
]
