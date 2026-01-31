# infodynamics_jax/infodynamics/likelihoods/gaussian_separable.py
r"""
Matrix-normal Gaussian likelihood with separable covariance:
    vec(YA^T) ~ N(0, C_N \otimes S)
where C_N is the GP covariance across samples (N), and S = A A^T across outputs.

This module provides:
- Nystrom/VFE-style collapsed build for C_N
- Matrix-normal log-likelihood using Cholesky factors
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from infodynamics_jax.gp.utils import safe_cholesky


@dataclass
class MatrixNormalCache:
    L_cn: jnp.ndarray  # Cholesky of C_N (N x N)
    L_s: jnp.ndarray   # Cholesky of S (M x M)
    logdet_cn: jnp.ndarray
    logdet_s: jnp.ndarray


def _logdet_from_chol(L: jnp.ndarray) -> jnp.ndarray:
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def collapsed_build_matrixnormal(
    Kfu: jnp.ndarray,
    Kuu: jnp.ndarray,
    noise_var: float,
    jitter: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build Nyström/VFE-style C_N = Q_ff + sigma^2 I.

    Args:
        Kfu: (N, M) kernel between X and Z
        Kuu: (M, M) kernel between Z and Z
        noise_var: scalar sigma^2
    Returns:
        L_cn: Cholesky of C_N
        V: (M, N) where Q_ff = V^T V
        L_uu: Cholesky of Kuu
    """
    L_uu = safe_cholesky(Kuu, jitter=jitter, max_jitter=1e-2)
    V = jsp.linalg.solve_triangular(L_uu, Kfu.T, lower=True)  # (M, N)
    Qff = V.T @ V
    Cn = Qff + jnp.array(noise_var, dtype=Kfu.dtype) * jnp.eye(Kfu.shape[0], dtype=Kfu.dtype)
    L_cn = safe_cholesky(Cn, jitter=jitter, max_jitter=1e-2)
    return L_cn, V, L_uu


def collapsed_build_matrixnormal_from_kernel(
    kernel_fn: Callable,
    X: jnp.ndarray,
    Z: jnp.ndarray,
    kernel_params,
    noise_var: float,
    jitter: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convenience wrapper to build C_N from a kernel function."""
    Kuu = kernel_fn(Z, Z, kernel_params)
    Kfu = kernel_fn(X, Z, kernel_params)
    return collapsed_build_matrixnormal(Kfu, Kuu, noise_var=noise_var, jitter=jitter)


def matrix_normal_cache(L_cn: jnp.ndarray, L_s: jnp.ndarray) -> MatrixNormalCache:
    return MatrixNormalCache(
        L_cn=L_cn,
        L_s=L_s,
        logdet_cn=_logdet_from_chol(L_cn),
        logdet_s=_logdet_from_chol(L_s),
    )


def matrix_normal_nll_from_cache(
    Y_tilde: jnp.ndarray,
    cache: MatrixNormalCache,
    scale_ratio: float = 1.0,
    include_const: bool = False,
) -> jnp.ndarray:
    """
    Negative log-likelihood for matrix-normal with separable covariance.

    Args:
        Y_tilde: (N, M) transformed outputs (Y A^T)
        cache: MatrixNormalCache with L_cn, L_s
        scale_ratio: optional scaling (e.g., dimension minibatch)
        include_const: include N*M*log(2π) term if True
    """
    N, M = Y_tilde.shape
    L_cn = cache.L_cn
    L_s = cache.L_s

    # Whiten in output space (S): (M, N)
    Yt = Y_tilde.T  # (M, N)
    Yt_w = jsp.linalg.solve_triangular(L_s, Yt, lower=True)  # (M, N)

    # Solve C_N for each column
    Yw = Yt_w.T  # (N, M)
    alpha = jsp.linalg.solve_triangular(L_cn, Yw, lower=True)
    alpha = jsp.linalg.solve_triangular(L_cn.T, alpha, lower=False)

    quad = jnp.sum(Yw * alpha)
    logdet = M * cache.logdet_cn + N * cache.logdet_s
    const = (N * M) * jnp.log(jnp.array(2.0 * jnp.pi, dtype=Y_tilde.dtype)) if include_const else 0.0

    nll = 0.5 * (logdet + quad + const)
    return jnp.array(scale_ratio, dtype=Y_tilde.dtype) * nll


def matrix_normal_nll(
    Y_tilde: jnp.ndarray,
    L_cn: jnp.ndarray,
    L_s: jnp.ndarray,
    scale_ratio: float = 1.0,
    include_const: bool = False,
) -> jnp.ndarray:
    cache = matrix_normal_cache(L_cn, L_s)
    return matrix_normal_nll_from_cache(Y_tilde, cache, scale_ratio=scale_ratio, include_const=include_const)


# =============================================================================
# High-level API: Configuration-based construction
# =============================================================================


@dataclass
class GradLikelihoodCFG:
    """
    Configuration for gradient-domain Gaussian likelihood.

    This enables Route A (strict Gaussian + linear operator) from the design doc:
    - Y_tilde = Y @ A^T (linear transform of outputs)
    - vec(Y_tilde) ~ N(0, C_N ⊗ S), where S = A A^T
    - C_N is the GP covariance (Nyström/VFE low-rank)

    Attributes:
        enabled: Whether to enable gradient-domain likelihood
        use_luma: Apply operators on luminance channel only (shape-focused)
        include_identity: Include identity operator (pixel-level)
        identity_weight: Weight for identity operator

        sobel_enabled: Enable Sobel gradient operator
        sobel_weight: Weight for Sobel operator

        laplacian_enabled: Enable Laplacian operator
        laplacian_weight: Weight for Laplacian operator
        laplacian_eight_connected: Use 8-connected Laplacian (default False)

        log_enabled: Enable LoG (Laplacian of Gaussian) operator
        log_weight: Weight for LoG operator
        log_size: LoG kernel size (must be odd)
        log_sigma: LoG kernel sigma
        log_sigmas: Optional tuple of LoG sigmas (overrides log_sigma if provided)
        log_weights: Optional tuple of LoG weights (parallel to log_sigmas)

        multiscale_enabled: Enable multi-scale gradients
        multiscale_scales: Tuple of downsample factors (e.g., (1, 2, 4))
        multiscale_weights: Tuple of weights for each scale
    """
    enabled: bool = True
    use_luma: bool = True

    include_identity: bool = True
    identity_weight: float = 1.0

    sobel_enabled: bool = True
    sobel_weight: float = 1.0

    laplacian_enabled: bool = False
    laplacian_weight: float = 0.5
    laplacian_eight_connected: bool = False

    log_enabled: bool = True
    log_weight: float = 0.5
    log_size: int = 5
    log_sigma: float = 1.0
    log_sigmas: Tuple[float, ...] = ()
    log_weights: Tuple[float, ...] = ()

    multiscale_enabled: bool = True
    multiscale_scales: Tuple[int, ...] = (1, 2, 4)
    multiscale_weights: Tuple[float, ...] = (1.0, 0.5, 0.25)

    def __post_init__(self):
        if len(self.multiscale_scales) != len(self.multiscale_weights):
            raise ValueError("multiscale_scales and multiscale_weights must have same length")
        if self.log_size % 2 == 0:
            raise ValueError("log_size must be odd")
        if self.log_sigmas and self.log_weights:
            if len(self.log_sigmas) != len(self.log_weights):
                raise ValueError("log_sigmas and log_weights must have same length")


def build_output_operator(
    cfg: GradLikelihoodCFG,
    H: int,
    W: int,
    C: int,
    edge_weight_map: Optional[jnp.ndarray] = None,
    edge_weight_on: Tuple[str, ...] = ("sobel", "log", "laplacian"),
    whiten_mode: str = "full",  # "full" | "blockdiag" | "none"
):
    """
    Build the output operator A and its associated S = A A^T from configuration.

    Args:
        cfg: GradLikelihoodCFG configuration
        H, W, C: Image dimensions (height, width, channels)
        edge_weight_map: Optional fixed per-pixel weight map (H,W or H,W,1)
        edge_weight_on: Operator names to apply edge weights to
        whiten_mode: Output whitening mode: "full", "blockdiag", or "none"

    Returns:
        Tuple of (apply_A, L_s, logdet_S):
        - apply_A: function (N, D) -> (N, M) to transform flattened images
        - L_s: Cholesky of S = A A^T (or None if whiten_mode="none")
        - logdet_S: log determinant of S (0 if whiten_mode="none")
    """
    from infodynamics_jax.ops.image_linear import (
        identity_op, sobel_op, laplacian_op, log_op, multiscale_op, stack_ops, weighted_op, compute_S, compute_S_blockdiag
    )

    if whiten_mode not in ("full", "blockdiag", "none"):
        raise ValueError("whiten_mode must be one of: 'full', 'blockdiag', 'none'")

    def _maybe_weight(op, tag):
        if edge_weight_map is None or tag not in edge_weight_on:
            return op
        return weighted_op(op, edge_weight_map, name="edge_weight")

    ops = []
    weights = []

    if not cfg.enabled:
        ops = [identity_op(H, W, C)]
        weights = [1.0]
    else:
        # Identity (pixel-level)
        if cfg.include_identity:
            ops.append(identity_op(H, W, C))
            weights.append(cfg.identity_weight)

        # Sobel gradients
        if cfg.sobel_enabled:
            if cfg.multiscale_enabled:
                ms_sobel = multiscale_op(
                    sobel_op, H, W, C,
                    scales=cfg.multiscale_scales,
                    weights=cfg.multiscale_weights,
                    weight_map=edge_weight_map if "sobel" in edge_weight_on else None,
                    use_luma=cfg.use_luma,
                )
                ops.append(ms_sobel)
                weights.append(cfg.sobel_weight)
            else:
                ops.append(_maybe_weight(sobel_op(H, W, C, use_luma=cfg.use_luma), "sobel"))
                weights.append(cfg.sobel_weight)

    # Laplacian
    if cfg.laplacian_enabled:
        if cfg.multiscale_enabled:
            ms_lap = multiscale_op(
                laplacian_op, H, W, C,
                scales=cfg.multiscale_scales,
                weights=cfg.multiscale_weights,
                weight_map=edge_weight_map if "laplacian" in edge_weight_on else None,
                use_luma=cfg.use_luma,
                eight_connected=cfg.laplacian_eight_connected,
            )
            ops.append(ms_lap)
            weights.append(cfg.laplacian_weight)
        else:
            ops.append(_maybe_weight(laplacian_op(H, W, C, use_luma=cfg.use_luma, eight_connected=cfg.laplacian_eight_connected), "laplacian"))
            weights.append(cfg.laplacian_weight)

        # LoG (Laplacian of Gaussian)
        if cfg.log_enabled:
            if cfg.multiscale_enabled:
                ms_log = multiscale_op(
                    log_op, H, W, C,
                    scales=cfg.multiscale_scales,
                    weights=cfg.multiscale_weights,
                    weight_map=edge_weight_map if "log" in edge_weight_on else None,
                    use_luma=cfg.use_luma,
                    size=cfg.log_size,
                    sigma=cfg.log_sigma,
                )
                ops.append(ms_log)
                weights.append(cfg.log_weight)
            else:
                ops.append(_maybe_weight(log_op(H, W, C, size=cfg.log_size, sigma=cfg.log_sigma, use_luma=cfg.use_luma), "log"))
                weights.append(cfg.log_weight)

    if len(ops) == 0:
        ops = [identity_op(H, W, C)]
        weights = [1.0]

    if len(ops) == 1:
        op_apply = ops[0]
        w = weights[0]
        if w != 1.0:
            _orig_fn = op_apply.apply_flat_fn
            op_apply = type(op_apply)(
                name=op_apply.name,
                in_shape=op_apply.in_shape,
                out_dim=op_apply.out_dim,
                apply_flat_fn=lambda y, fn=_orig_fn, wt=w: jnp.array(wt, dtype=y.dtype) * fn(y),
            )
    else:
        op_apply = stack_ops(ops, weights=weights, name="grad_lik")

    if whiten_mode == "none":
        L_s = None
        logdet_S = jnp.array(0.0, dtype=jnp.float32)
    elif whiten_mode == "blockdiag":
        S, L_s = compute_S_blockdiag(ops, weights=weights)
        logdet_S = _logdet_from_chol(L_s)
    else:
        S, L_s = compute_S(op_apply)
        logdet_S = _logdet_from_chol(L_s)

    return op_apply.apply_flat, L_s, logdet_S

    def _maybe_weight(op, tag):
        if edge_weight_map is None or tag not in edge_weight_on:
            return op
        return weighted_op(op, edge_weight_map, name="edge_weight")

    ops = []
    weights = []

    # Identity (pixel-level)
    if cfg.include_identity:
        ops.append(identity_op(H, W, C))
        weights.append(cfg.identity_weight)

    # Sobel gradients
    if cfg.sobel_enabled:
        if cfg.multiscale_enabled:
            ms_sobel = multiscale_op(
                sobel_op, H, W, C,
                scales=cfg.multiscale_scales,
                weights=cfg.multiscale_weights,
                weight_map=edge_weight_map if "sobel" in edge_weight_on else None,
                use_luma=cfg.use_luma,
            )
            ops.append(ms_sobel)
            weights.append(cfg.sobel_weight)
        else:
            ops.append(_maybe_weight(sobel_op(H, W, C, use_luma=cfg.use_luma), "sobel"))
            weights.append(cfg.sobel_weight)

    # LoG (Laplacian of Gaussian)
    if cfg.log_enabled:
        if cfg.log_sigmas:
            if cfg.log_weights:
                log_pairs = list(zip(cfg.log_sigmas, cfg.log_weights))
            else:
                log_pairs = [(s, cfg.log_weight) for s in cfg.log_sigmas]
        else:
            log_pairs = [(cfg.log_sigma, cfg.log_weight)]

        for sigma, w_log in log_pairs:
            if cfg.multiscale_enabled:
                ms_log = multiscale_op(
                    log_op, H, W, C,
                    scales=cfg.multiscale_scales,
                    weights=cfg.multiscale_weights,
                    weight_map=edge_weight_map if "log" in edge_weight_on else None,
                    use_luma=cfg.use_luma,
                    size=cfg.log_size,
                    sigma=float(sigma),
                )
                ops.append(ms_log)
                weights.append(w_log)
            else:
                ops.append(_maybe_weight(log_op(H, W, C, size=cfg.log_size, sigma=float(sigma), use_luma=cfg.use_luma), "log"))
                weights.append(w_log)

    if len(ops) == 0:
        op = identity_op(H, W, C)
    elif len(ops) == 1:
        op = ops[0]
        # Apply weight scaling
        _orig_fn = op.apply_flat_fn
        w = weights[0]
        op = type(op)(
            name=op.name,
            in_shape=op.in_shape,
            out_dim=op.out_dim,
            apply_flat_fn=lambda y, fn=_orig_fn, wt=w: jnp.array(wt, dtype=y.dtype) * fn(y),
        )
    else:
        op = stack_ops(ops, weights=weights, name="grad_lik")

    S, L_s = compute_S(op)
    logdet_S = _logdet_from_chol(L_s)
    return op.apply_flat, L_s, logdet_S


def separable_nll_from_collapsed(
    L_cn: jnp.ndarray,
    Y_sub: jnp.ndarray,
    apply_A: Callable,
    L_s: Optional[jnp.ndarray],
    logdet_S: Optional[jnp.ndarray],
    scale_ratio: float = 1.0,
    include_const: bool = False,
) -> jnp.ndarray:
    """
    Compute separable matrix-normal NLL using a collapsed C_N Cholesky.

    This interfaces with existing collapsed_build patterns where L_cn is
    the Cholesky of C_N = Q_ff + σ²I.

    Args:
        L_cn: (N, N) Cholesky of C_N from collapsed_build
        Y_sub: (N, D) output observations (possibly dimension-minibatched)
        apply_A: function (N, D) -> (N, M) to transform outputs
        L_s: (M, M) Cholesky of S = A A^T (or None to disable output whitening)
        logdet_S: scalar log|S| (0 if L_s is None)
        scale_ratio: dimension minibatch scale factor
        include_const: include constant term

    Returns:
        Scalar negative log-likelihood
    """
    Y_tilde = apply_A(Y_sub)  # (N, M)
    N, M = Y_tilde.shape

    # Output whitening (optional)
    if L_s is None:
        Yw = Y_tilde
        logdet_S = jnp.array(0.0, dtype=Y_sub.dtype)
    else:
        Yt = Y_tilde.T  # (M, N)
        Yt_w = jsp.linalg.solve_triangular(L_s, Yt, lower=True)  # (M, N)
        Yw = Yt_w.T  # (N, M)

    # Solve C_N^{-1} for each (whitened) column
    alpha = jsp.linalg.solve_triangular(L_cn, Yw, lower=True)
    alpha = jsp.linalg.solve_triangular(L_cn.T, alpha, lower=False)

    # Quadratic term: tr(S^{-1} Y^T C_N^{-1} Y) = sum(Yw * alpha)
    quad = jnp.sum(Yw * alpha)

    # Log determinant: M * log|C_N| + N * log|S|
    logdet_cn = _logdet_from_chol(L_cn)
    logdet = M * logdet_cn + N * logdet_S

    const = (N * M) * jnp.log(jnp.array(2.0 * jnp.pi, dtype=Y_sub.dtype)) if include_const else 0.0

    nll = 0.5 * (logdet + quad + const)
    return jnp.array(scale_ratio, dtype=Y_sub.dtype) * nll


def separable_nll_pixel_only(
    L_cn: jnp.ndarray,
    Y_sub: jnp.ndarray,
    sn2: jnp.ndarray,
    scale_ratio: float = 1.0,
) -> jnp.ndarray:
    """
    Standard collapsed Gaussian NLL (pixel-level only, no output transform).

    This is equivalent to the existing inertial_expected_nll without gradient terms.

    Args:
        L_cn: (N, N) Cholesky of C_N
        Y_sub: (N, D) output observations
        sn2: scalar noise variance
        scale_ratio: dimension minibatch scale factor

    Returns:
        Scalar negative log-likelihood (quadratic term only, no logdet)
    """
    alpha = jsp.linalg.solve_triangular(L_cn, Y_sub, lower=True)
    alpha = jsp.linalg.solve_triangular(L_cn.T, alpha, lower=False)
    resid_sq = jnp.sum(Y_sub * alpha)
    return 0.5 * resid_sq * jnp.array(scale_ratio, dtype=Y_sub.dtype)
