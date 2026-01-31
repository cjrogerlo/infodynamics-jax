# infodynamics_jax/infodynamics/likelihoods/__init__.py
"""
Gradient-domain Gaussian likelihoods with separable covariance.

This module implements Route A (strict Gaussian + linear operator) for
incorporating structural priors (Sobel gradients, LoG, multi-scale) into
the likelihood while maintaining analytic marginalization.

Key components:
- GradLikelihoodCFG: Configuration dataclass
- build_output_operator: Construct A and S = A A^T from config
- separable_nll_from_collapsed: Compute NLL given collapsed C_N
- linear_ops: Sobel, LoG, multiscale, and stacking utilities

Mathematical model:
    Y_tilde = Y @ A^T
    vec(Y_tilde) ~ N(0, C_N ⊗ S)
where C_N is the GP covariance (Nyström/VFE) and S = A A^T.
"""
from .gaussian_separable import (
    MatrixNormalCache,
    GradLikelihoodCFG,
    collapsed_build_matrixnormal,
    collapsed_build_matrixnormal_from_kernel,
    matrix_normal_cache,
    matrix_normal_nll,
    matrix_normal_nll_from_cache,
    build_output_operator,
    separable_nll_from_collapsed,
    separable_nll_pixel_only,
)
from . import linear_ops

__all__ = [
    # Cache and NLL
    "MatrixNormalCache",
    "matrix_normal_cache",
    "matrix_normal_nll",
    "matrix_normal_nll_from_cache",
    # Collapsed builds
    "collapsed_build_matrixnormal",
    "collapsed_build_matrixnormal_from_kernel",
    # High-level API
    "GradLikelihoodCFG",
    "build_output_operator",
    "separable_nll_from_collapsed",
    "separable_nll_pixel_only",
    # Linear operators submodule
    "linear_ops",
]
