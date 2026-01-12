# infodynamics_jax/gp/ansatz/__init__.py
"""
Ansatz layer for non-conjugate likelihoods.

Gaussian likelihoods don't need ansatz - they use analytic solutions.
This module provides ansatz-based estimators (GH, MC) for non-Gaussian cases.
"""
from .state import VariationalState
from .expected import (
    qfi_from_qu_full,
    expected_nll_factorised_gh,
    expected_nll_factorised_mc,
)
from .gh import GaussHermite
from .mc import MonteCarlo

__all__ = [
    "VariationalState",
    "qfi_from_qu_full",
    "expected_nll_factorised_gh",
    "expected_nll_factorised_mc",
    "GaussHermite",
    "MonteCarlo",
]
