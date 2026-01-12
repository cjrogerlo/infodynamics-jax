# infodynamics_jax/gp/ansatz/__init__.py
"""
Ansatz layer for non-conjugate likelihoods.

Gaussian likelihoods don't need ansatz - they use analytic solutions.
This module provides ansatz-based estimators (GH, MC) for non-Gaussian cases.

IMPORTANT: For type-II inference (VFE / ML-II), inducing variables are fully
marginalised via sparsified kernels. No posterior state over u is constructed.
VariationalState is only used for non-conjugate likelihoods (GH/MC estimators).
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
