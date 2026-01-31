# -*- coding: utf-8 -*-
"""
Inferential Infodynamics — Structural Prior Energies (v0)
==========================================================

This module defines structural prior energies: data-independent energy
functions that encode structural assumptions about latent variables X.

Core principle:
    Prior energy = structural constraints on latent variables X that
    are independent of observed data Y.

    These encode physical/geometric assumptions (spatial coherence,
    dimension semantics, parts assembly) that guide the inference.

Module responsibilities:
    ✅ Structural prior energy functions E_prior(X)
    ✅ Configuration objects (PriorCFG, PriorMeta)
    ✅ Prior module system (PriorModule, PriorEnergy)
    ✅ Grid topology construction

    ❌ Does NOT depend on data Y
    ❌ Does NOT depend on kernels / GP algebra
    ❌ Does NOT implement dynamics
    ❌ Does NOT depend on beta / annealing
    → Data-dependent energy belongs to inertial_energy.py
    → Dynamics belong to infodynamics.py
    → State management belongs to inference.py

Architecture:
    Three-Layer Design:
    1. Configuration Layer: PriorCFG (weights), PriorMeta (topology)
    2. Atomic Functions Layer: Pure mathematical prior energies
    3. Module Layer: PriorModule subclasses (semantic wrappers)

Structural Directions:
    - Across-N: Spatial/shape coherence (Laplacian, Thin-plate, TV)
    - Across-Q: Dimension semantics (Hierarchical ARD, Column sparsity)
    - Across-(N,Q): Parts assembly (Row sparsity)

Version: v0
Status:  experimental but structurally frozen

Author: Chi-Jen Roger Lo
License: Apache License 2.0
Copyright (c) 2024 Chi-Jen Roger Lo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax import lax

EPS = 1e-9


# ============================================================
# Configuration objects
# ============================================================

@dataclass
class PriorCFG:
    """
    Weights for structural priors.
    All lambdas default to 0 (disabled).
    """
    # Base Gaussian reference on X
    x_l2_lambda: float = 1.0

    # Structural prior on kernel hyperparameters (project-specific)
    ell_prior_lambda: float = 0.0

    # Shape / geometry priors
    laplacian_lambda: float = 0.0
    thinplate_lambda: float = 0.0
    tv_lambda: float = 0.0
    tv_epsilon: float = 1e-3

    # Anti-collapse / volume
    repulsion_lambda: float = 0.0
    repulsion_p: float = 2.0
    repulsion_epsilon: float = 1e-3
    repulsion_max_pairs: int = 4096

    # Global scale / anisotropy control
    whiten_lambda: float = 0.0

    # Subspace decoupling (e.g. shape vs colour)
    decouple_lambda: float = 0.0
    block_split: Optional[int] = None

    # Across-Q hierarchical / sparsity priors
    # hier_ard_lambda: Typical range 0.1 - 2.0.
    # WARNING: If x_l2_lambda is much larger (e.g. 1.0) than hier_ard_lambda (e.g. 0.005),
    # the isotropic shrinkage will drown out the ARD ordering effect.
    hier_ard_lambda: float = 1.0
    hier_ard_mode: str = "exp"  # "exp" or "poly"
    hier_ard_gamma: float = 0.9  # for exp mode
    hier_ard_power: float = 1.5  # for poly mode
    row_sparsity_lambda: float = 0.0
    row_sparsity_epsilon: float = 1e-3
    col_sparsity_lambda: float = 0.0
    col_sparsity_epsilon: float = 1e-3


@dataclass
class PriorMeta:
    """
    Structural metadata (NOT data-dependent).
    These must be fixed by model design, not inferred from Y.
    """
    edges: Optional[jnp.ndarray] = None      # (E, 2) graph edges
    pair_idx: Optional[jnp.ndarray] = None   # (P, 2) repulsion pairs
    block_split: Optional[int] = None         # optional override

    @staticmethod
    def grid(H: int, W: int, neighbourhood: str = "4n"):
        """
        Create PriorMeta with grid topology.

        This is the semantic entry point for grid-based topology.
        The topology itself is a prior assumption, not a tool.

        Parameters
        ----------
        H : int
            Grid height (number of rows)
        W : int
            Grid width (number of columns)
        neighbourhood : str
            "4n" for 4-neighbour (cardinal directions)
            "8n" for 8-neighbour (includes diagonals)

        Returns
        -------
        PriorMeta
            With edges set to the grid topology

        Notes
        -----
        Why is this valid prior metadata?
        - Edges depend only on index arrangement, not data Y
        - Equivalent to choosing a reference topology on latent space
        - Like defining a mesh in FEM/PDE before solving
        """
        if neighbourhood == "4n":
            edges = _grid_edges_4n(H, W)
        elif neighbourhood == "8n":
            edges = _grid_edges_8n(H, W)
        else:
            raise ValueError(f"Unknown neighbourhood: {neighbourhood}. Use '4n' or '8n'")
        return PriorMeta(edges=edges)


# ============================================================
# Helper functions
# ============================================================

def _charbonnier(r2, eps):
    return jnp.sqrt(r2 + eps * eps)


def _safe_int(x):
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def _grid_edges_4n(H: int, W: int):
    """
    Internal helper: 4-neighbour grid topology.

    Returns undirected edges connecting each node to its
    right and bottom neighbors (avoiding duplicates).

    This is a prior assumption about latent space topology,
    not a data-dependent tool.
    """
    def idx(r, c):
        """Row-major index: i = row * W + col"""
        return r * W + c

    # Vectorized approach: generate all possible edges, then filter
    # Right edges: (r, c) -> (r, c+1) for all valid pairs
    r_right = jnp.arange(H)[:, None]  # (H, 1)
    c_right = jnp.arange(W - 1)[None, :]  # (1, W-1)
    i_right_from = (r_right * W + c_right).flatten()  # (H*(W-1),)
    i_right_to = (r_right * W + (c_right + 1)).flatten()  # (H*(W-1),)
    right_edges = jnp.stack([i_right_from, i_right_to], axis=1)  # (H*(W-1), 2)
    
    # Bottom edges: (r, c) -> (r+1, c) for all valid pairs
    r_bottom = jnp.arange(H - 1)[:, None]  # (H-1, 1)
    c_bottom = jnp.arange(W)[None, :]  # (1, W)
    i_bottom_from = (r_bottom * W + c_bottom).flatten()  # ((H-1)*W,)
    i_bottom_to = ((r_bottom + 1) * W + c_bottom).flatten()  # ((H-1)*W,)
    bottom_edges = jnp.stack([i_bottom_from, i_bottom_to], axis=1)  # ((H-1)*W, 2)
    
    # Concatenate all edges
    if H * (W - 1) > 0 and (H - 1) * W > 0:
        edges = jnp.concatenate([right_edges, bottom_edges], axis=0)
    elif H * (W - 1) > 0:
        edges = right_edges
    elif (H - 1) * W > 0:
        edges = bottom_edges
    else:
        edges = jnp.array([], dtype=jnp.int32).reshape(0, 2)
    
    return edges


def _grid_edges_8n(H: int, W: int):
    """
    Internal helper: 8-neighbour grid topology.

    Includes diagonal connections for more isotropic shape priors.
    Use when 4-neighbour creates axis-aligned artifacts.
    """
    def idx(r, c):
        """Row-major index: i = row * W + col"""
        return r * W + c

    # Start with 4-neighbor edges
    edges_4n = _grid_edges_4n(H, W)
    
    # Add diagonal SE edges: (r, c) -> (r+1, c+1) for valid pairs
    r_se = jnp.arange(H - 1)[:, None]  # (H-1, 1)
    c_se = jnp.arange(W - 1)[None, :]  # (1, W-1)
    i_se_from = (r_se * W + c_se).flatten()  # ((H-1)*(W-1),)
    i_se_to = ((r_se + 1) * W + (c_se + 1)).flatten()  # ((H-1)*(W-1),)
    se_edges = jnp.stack([i_se_from, i_se_to], axis=1)  # ((H-1)*(W-1), 2)
    
    # Add diagonal SW edges: (r, c) -> (r+1, c-1) for valid pairs
    r_sw = jnp.arange(H - 1)[:, None]  # (H-1, 1)
    c_sw = jnp.arange(1, W)[None, :]  # (1, W-1)
    i_sw_from = (r_sw * W + c_sw).flatten()  # ((H-1)*(W-1),)
    i_sw_to = ((r_sw + 1) * W + (c_sw - 1)).flatten()  # ((H-1)*(W-1),)
    sw_edges = jnp.stack([i_sw_from, i_sw_to], axis=1)  # ((H-1)*(W-1), 2)
    
    # Concatenate all edges
    edge_list = [edges_4n]
    if (H - 1) * (W - 1) > 0:
        edge_list.extend([se_edges, sw_edges])
    
    if len(edge_list) > 0:
        edges = jnp.concatenate(edge_list, axis=0)
    else:
        edges = jnp.array([], dtype=jnp.int32).reshape(0, 2)
    
    return edges


# ============================================================
# Base priors
# ============================================================

def x_l2_prior(X, lam: float):
    """
    Gaussian reference measure on latent variables.

    Mathematical form:
        E = (lam / 2) * ||X||^2 = (lam / 2) * Σ_i Σ_q X_{iq}^2

    Effect on X (N×Q):
        - Penalizes large values in ALL entries of X
        - Acts as a baseline regularization, preventing unbounded growth
        - Does NOT differentiate between shape/color/structure

    Semantic effect:
        - Controls overall scale of latent variables
        - Prevents drift/explosion during optimization
        - Neutral with respect to shape vs color (affects both equally)

    When to use:
        - Almost always enabled (default lambda=1.0)
        - Essential for numerical stability
        - Does NOT create shape/color separation by itself
    """
    if lam <= 0.0:
        return jnp.array(0.0, dtype=X.dtype)
    return 0.5 * lam * jnp.sum(X * X)


def ell_l2_shrinkage(phi_raw, kcfg, lam: float):
    """
    L2 shrinkage prior on log-lengthscales.

    NOTE:
    - Pure structural prior
    - Never multiplied by beta
    - Project-specific assumption: phi_raw has dimension 17
    """
    if lam <= 0.0:
        return jnp.array(0.0, dtype=phi_raw.dtype)

    if int(phi_raw.shape[0]) != 17:
        return jnp.array(0.0, dtype=phi_raw.dtype)

    s1, s2 = kcfg.split_indices
    d_c = int(s1)
    d_m = int(s2 - s1)

    idx = 0
    log_ell_c = phi_raw[idx:idx + d_c]
    idx += d_c + 1  # skip sf2_c
    log_ell_m = phi_raw[idx:idx + d_m]

    mu_c = jnp.log(jnp.array(kcfg.ell_coarse_init, dtype=phi_raw.dtype))
    mu_m = jnp.log(jnp.array(kcfg.ell_mid_init, dtype=phi_raw.dtype))

    return lax.stop_gradient(
        lam * (
            jnp.sum((log_ell_c - mu_c) ** 2) +
            jnp.sum((log_ell_m - mu_m) ** 2)
        )
    )


# ============================================================
# Geometry / shape priors
# ============================================================

def graph_laplacian_prior(X, edges, lam: float):
    """
    First-order smoothness prior on a graph.

    Mathematical form:
        E = (lam / 2) * Σ_{(i,j)∈edges} ||X_i - X_j||^2
        where ||X_i - X_j||^2 = Σ_q (X_{iq} - X_{jq})^2

    Effect on X (N×Q):
        - For each edge (i,j), penalizes differences between rows X_i and X_j
        - Encourages neighboring positions (in grid space) to have similar latent codes
        - Operates ACROSS positions (N dimension), affects ALL dimensions (Q) equally

    Semantic effect:
        - **SHAPE/STRUCTURE PRIOR**: Encourages spatial continuity
        - Makes latent field smooth over spatial neighbors
        - Critical for "assemblable shapes" - parts that connect smoothly
        - Does NOT differentiate shape vs color (smooths both)

    When to use:
        - When you want spatially coherent latent representations
        - Essential for shape-stable generation (prevents fragmented parts)
        - Works best with grid topology (edges from PriorMeta.grid)
        - Typical lambda: 1e-2 to 1e-1
    """
    if lam <= 0.0 or edges is None:
        return jnp.array(0.0, dtype=X.dtype)

    Xi = X[edges[:, 0]]
    Xj = X[edges[:, 1]]
    diff = Xi - Xj
    return 0.5 * lam * jnp.sum(jnp.sum(diff * diff, axis=-1))


def thinplate_prior(X, edges, lam: float):
    """
    Biharmonic (thin-plate) prior.

    Mathematical form:
        E = (lam / 2) * Σ_i ||(LX)_i||^2
        where (LX)_i = Σ_{j∈neighbors(i)} (X_i - X_j) is the discrete Laplacian

    Effect on X (N×Q):
        - Computes Laplacian Lx at each position i
        - Penalizes curvature (second derivative) rather than slope (first derivative)
        - Stricter than graph_laplacian: allows linear gradients but penalizes bending
        - Operates ACROSS positions (N), affects ALL dimensions (Q) equally

    Semantic effect:
        - **SHAPE/STRUCTURE PRIOR**: Encourages "flexible but smooth" shapes
        - Allows gradual changes but penalizes sharp bends/curves
        - More restrictive than Laplacian → use with caution
        - Can make sampling harder if too strong

    When to use:
        - When Laplacian is too weak but you still want smoothness
        - For "soft, flexible" shape priors (e.g., deformable objects)
        - Start with small lambda (1e-3 to 1e-2)
        - Risk: Can be too stiff → harder sampling
    """
    if lam <= 0.0 or edges is None:
        return jnp.array(0.0, dtype=X.dtype)

    Lx = jnp.zeros_like(X)
    i = edges[:, 0]
    j = edges[:, 1]
    d = X[i] - X[j]

    Lx = Lx.at[i].add(d)
    Lx = Lx.at[j].add(-d)

    return 0.5 * lam * jnp.sum(jnp.sum(Lx * Lx, axis=-1))


def tv_charbonnier_prior(X, edges, lam: float, eps: float):
    """
    Total-variation-like prior with Charbonnier penalty.

    Mathematical form:
        E = λ * Σ_{(i,j)∈edges} sqrt(||X_i - X_j||^2 + ε^2)

    Effect on X (N×Q):
        - Non-quadratic penalty on differences between neighbors
        - Allows piecewise smooth fields: smooth within regions, sharp at boundaries
        - More tolerant of discontinuities than Laplacian/thinplate
        - Operates ACROSS positions (N), affects ALL dimensions (Q) equally

    Semantic effect:
        - **SHAPE/STRUCTURE PRIOR**: Allows "part-based" assembly
        - Permits sharp boundaries between parts while keeping parts smooth internally
        - Good for "component-based" generation (e.g., articulated objects)
        - Does NOT differentiate shape vs color (allows boundaries in both)

    When to use:
        - When you want piecewise smooth latent fields
        - For "parts/segments" that can have sharp boundaries
        - Start from lambda=0, add gradually if needed
        - Risk: Non-quadratic → harder optimization
    """
    if lam <= 0.0 or edges is None:
        return jnp.array(0.0, dtype=X.dtype)

    Xi = X[edges[:, 0]]
    Xj = X[edges[:, 1]]
    r2 = jnp.sum((Xi - Xj) ** 2, axis=-1)
    return lam * jnp.sum(_charbonnier(r2, eps))


# ============================================================
# Anti-collapse / scale control
# ============================================================

def repulsion_prior(X, pairs, lam: float, p: float, eps: float, max_pairs: int):
    """
    Repulsive prior to prevent latent collapse.

    Mathematical form:
        E = λ * Σ_{(i,j)∈pairs} 1 / (||X_i - X_j||^p + ε)

    Effect on X (N×Q):
        - For each pair (i,j), penalizes points being too close
        - Encourages latent points to spread out in latent space
        - Operates ACROSS positions (N dimension), affects ALL dimensions (Q) equally
        - If pairs=None, uses deterministic subsampling (no randomness)

    Semantic effect:
        - **VOLUME/SPACING PRIOR**: Prevents collapse to single point or small cluster
        - Ensures latent space has "volume" → diversity in generation
        - Does NOT target shape vs color specifically (affects overall diversity)
        - Critical for avoiding mode collapse

    When to use:
        - When latent points collapse to small region
        - For maintaining diversity in generated samples
        - Typical lambda: 1e-3 to 1e-2 (small)
        - Works well with hierarchical ARD (complementary effects)
    """
    if lam <= 0.0:
        return jnp.array(0.0, dtype=X.dtype)

    N = X.shape[0]
    if pairs is None:
        P = min(N, max_pairs)
        i = jnp.arange(P)
        j = (i * 997 + 101) % N
        j = jnp.where(i == j, (j + 1) % N, j)
        pairs = jnp.stack([i, j], axis=1)

    Xi = X[pairs[:, 0]]
    Xj = X[pairs[:, 1]]
    r2 = jnp.sum((Xi - Xj) ** 2, axis=-1)
    r = _charbonnier(r2, eps)

    return lam * jnp.sum(1.0 / (r ** p + eps))


def whiten_prior(X, lam: float):
    """
    Whitening prior: encourages latent covariance to be close to identity.

    Mathematical form:
        E = λ * ||Σ - I||_F^2
        where Σ = (1/N) * X_c^T X_c is the sample covariance
        and X_c = X - mean(X, axis=0) is the centered X

    Effect on X (N×Q):
        - Computes covariance matrix Σ (Q×Q) from X
        - Penalizes deviations from identity (no correlation, unit variance per dim)
        - Operates ACROSS both N and Q: considers all positions and dimensions together
        - Prevents some dimensions from dominating or dying out

    Semantic effect:
        - **SCALE/ANISOTROPY CONTROL**: Keeps all dimensions at similar scale
        - Prevents "dead dimensions" (always near zero) or "explosive dimensions"
        - Does NOT create shape/color separation (keeps all dims balanced)
        - Can conflict with decouple_prior if you want shape/color separation

    When to use:
        - When some dimensions collapse to zero or explode
        - For balanced latent representations
        - Warning: May conflict with shape/color factorization goals
        - Typical lambda: 1e-3 to 1e-2
    """
    if lam <= 0.0 or X.shape[0] <= 1:
        return jnp.array(0.0, dtype=X.dtype)

    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    Sigma = (Xc.T @ Xc) / X.shape[0]
    I = jnp.eye(X.shape[1], dtype=X.dtype)

    return lam * jnp.sum((Sigma - I) ** 2)


# ============================================================
# Subspace structure priors
# ============================================================

def decouple_prior(X, split: Optional[int], lam: float):
    """
    Penalises cross-covariance between two latent blocks.

    Useful for shape / colour / appearance separation.
    """
    if lam <= 0.0 or split is None:
        return jnp.array(0.0, dtype=X.dtype)

    D = X.shape[1]
    split = _safe_int(split)
    if split <= 0 or split >= D:
        return jnp.array(0.0, dtype=X.dtype)

    G = X[:, :split]
    C = X[:, split:]
    N = X.shape[0]

    if N <= 1:
        return jnp.array(0.0, dtype=X.dtype)

    Gc = G - jnp.mean(G, axis=0, keepdims=True)
    Cc = C - jnp.mean(C, axis=0, keepdims=True)

    cross_cov = (Gc.T @ Cc) / N
    return lam * jnp.sum(cross_cov ** 2)


# ============================================================
# Across-Q hierarchical / sparsity priors
# ============================================================

def hierarchical_ard_prior(X, lam: float, mode: str = "exp", gamma: float = 0.9, power: float = 1.5):
    """
    Hierarchical ARD prior: ordered shrinkage across latent dimensions.

    Creates a semantic coordinate system where:
    - Early dimensions (q=1,2,...) = cheap → tend to carry global/main structure
    - Later dimensions (q=Q-1,Q) = expensive → only used when necessary (details/residuals)

    WARNING: If an isotropic L2 prior (e.g. x_l2_lambda=1.0) is much stronger than 
    this ARD prior (e.g. hier_ard_lambda=0.005), the uniform shrinkage will 
    overwhelm the ordering effect, resulting in near-uniform variances.

    E_hier(X) = (λ/2) * Σ_q w_q ||X_{·q}||^2

    Weighting schemes:
    - For exp mode:     w_q = (gamma_eff)^q     with gamma_eff > 1 (monotonically increasing)
    - For poly mode:    w_q = (q+1)^power

    Parameters
    ----------
    X : jnp.ndarray, shape (N, Q)
        Latent variables
    lam : float
        Overall strength
    mode : str
        "exp" for exponential weights, "poly" for polynomial
    gamma : float
        (For exp mode) Growth rate (gamma_eff > 1 enforced for monotonicity)
    power : float
        (For poly mode) Power for polynomial weights

    Returns
    -------
    energy : scalar
        Hierarchical ARD energy
    """
    if lam <= 0.0:
        return jnp.array(0.0, dtype=X.dtype)

    Q = X.shape[1]
    q_indices = jnp.arange(Q, dtype=X.dtype)

    if mode == "exp":
        # Enforce increasing weights with q.
        # If gamma <= 1 is provided, invert it to ensure growth.
        gamma_eff = jnp.array(gamma, dtype=X.dtype)
        gamma_eff = jnp.where(gamma_eff <= 1.0, 1.0 / (gamma_eff + EPS), gamma_eff)
        weights = gamma_eff ** q_indices
    elif mode == "poly":
        # w_q = (q+1)^p (q+1 to avoid 0^p)
        weights = (q_indices + 1.0) ** jnp.array(power, dtype=X.dtype)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'exp' or 'poly'")

    # Compute ||X_{·q}||^2 for each dimension q
    col_norms_sq = jnp.sum(X * X, axis=0)  # (Q,)

    # Weighted sum
    energy = 0.5 * lam * jnp.sum(weights * col_norms_sq)
    return energy


def row_group_lasso_prior(X, lam: float, eps: float):
    """
    Row-wise group lasso prior: encourages each position to use few dimensions.

    E_row(X) = λ * Σ_i sqrt(Σ_q X_{iq}^2 + ε^2)

    Effect:
    - For each position i (pixel/patch/index), encourages sparsity across dimensions
    - This is the cleanest "parts prior": different positions can use different
      subsets of dimensions → assembly capability emerges from here

    Parameters
    ----------
    X : jnp.ndarray, shape (N, Q)
        Latent variables
    lam : float
        Strength
    eps : float
        Smoothing parameter for sqrt

    Returns
    -------
    energy : scalar
        Row sparsity energy
    """
    if lam <= 0.0:
        return jnp.array(0.0, dtype=X.dtype)

    # For each row i: sqrt(sum_q X_{iq}^2 + eps^2)
    row_norms_sq = jnp.sum(X * X, axis=1)  # (N,)
    row_norms = _charbonnier(row_norms_sq, eps)  # sqrt(r^2 + eps^2)

    return lam * jnp.sum(row_norms)


def col_group_lasso_prior(X, lam: float, eps: float):
    """
    Column-wise group lasso prior: encourages using few dimensions globally.

    E_col(X) = λ * Σ_q sqrt(Σ_i X_{iq}^2 + ε^2)

    Effect:
    - Encourages using only a few latent dimensions (global factor selection)
    - Combined with hierarchical ARD:
      * hierarchical = soft ordering / soft shrinkage
      * col sparsity = hard-ish selection

    Parameters
    ----------
    X : jnp.ndarray, shape (N, Q)
        Latent variables
    lam : float
        Strength
    eps : float
        Smoothing parameter for sqrt

    Returns
    -------
    energy : scalar
        Column sparsity energy
    """
    if lam <= 0.0:
        return jnp.array(0.0, dtype=X.dtype)

    # For each column q: sqrt(sum_i X_{iq}^2 + eps^2)
    col_norms_sq = jnp.sum(X * X, axis=0)  # (Q,)
    col_norms = _charbonnier(col_norms_sq, eps)  # sqrt(r^2 + eps^2)

    return lam * jnp.sum(col_norms)


# ============================================================
# Module-based, class-oriented priors
# ============================================================

# ---------------------
# Abstract base class
# ---------------------

class PriorModule:
    """
    Abstract base class for a structural prior.

    A PriorModule represents a *named inductive bias* on latent space,
    not merely a numerical regulariser.
    """

    name: str

    def energy(self, X, phi_raw, meta):
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


# ---------------------
# Base / stability priors
# ---------------------

class XGaussianPrior(PriorModule):
    """
    Isotropic Gaussian reference measure on latent variables X.

    E = (λ/2) ||X||²
    """

    def __init__(self, lam: float):
        self.lam = lam
        self.name = "x_l2"

    def energy(self, X, phi_raw, meta):
        return x_l2_prior(X, self.lam)

    def describe(self):
        return "Gaussian reference measure on latent variables."


class RepulsionPrior(PriorModule):
    """
    Repulsive prior preventing latent collapse and encouraging volume.
    """

    def __init__(self, lam: float, p: float = 2.0, eps: float = 1e-3, max_pairs: int = 4096):
        self.lam = lam
        self.p = p
        self.eps = eps
        self.max_pairs = max_pairs
        self.name = "repulsion"

    def energy(self, X, phi_raw, meta):
        pairs = meta.pair_idx if meta is not None else None
        return repulsion_prior(X, pairs, self.lam, self.p, self.eps, self.max_pairs)

    def describe(self):
        return "Repulsive prior preventing latent point collapse."


class WhitenPrior(PriorModule):
    """
    Whitening prior encouraging latent covariance to be close to identity.

    Prevents anisotropic collapse and keeps all dimensions at similar scale.
    """

    def __init__(self, lam: float):
        self.lam = lam
        self.name = "whiten"

    def energy(self, X, phi_raw, meta):
        return whiten_prior(X, self.lam)

    def describe(self):
        return "Whitening prior (encourages identity covariance)."


# ---------------------
# Geometry / field priors
# ---------------------

class LaplacianPrior(PriorModule):
    """
    Graph Laplacian prior enforcing local smoothness of latent fields.
    """

    def __init__(self, lam: float):
        self.lam = lam
        self.name = "laplacian"

    def energy(self, X, phi_raw, meta):
        if meta is None:
            return jnp.array(0.0, dtype=X.dtype)
        return graph_laplacian_prior(X, meta.edges, self.lam)

    def describe(self):
        return "Graph Laplacian smoothness prior."


class ThinPlatePrior(PriorModule):
    """
    Biharmonic (thin-plate) prior penalising curvature.
    """

    def __init__(self, lam: float):
        self.lam = lam
        self.name = "thinplate"

    def energy(self, X, phi_raw, meta):
        if meta is None:
            return jnp.array(0.0, dtype=X.dtype)
        return thinplate_prior(X, meta.edges, self.lam)

    def describe(self):
        return "Thin-plate (biharmonic) smoothness prior."


class TVPrior(PriorModule):
    """
    Total-variation / Charbonnier prior allowing piecewise smooth fields.
    """

    def __init__(self, lam: float, eps: float = 1e-3):
        self.lam = lam
        self.eps = eps
        self.name = "tv"

    def energy(self, X, phi_raw, meta):
        if meta is None:
            return jnp.array(0.0, dtype=X.dtype)
        return tv_charbonnier_prior(X, meta.edges, self.lam, self.eps)

    def describe(self):
        return "Total-variation (Charbonnier) prior on latent field."


# ---------------------
# Structural / factorisation priors
# ---------------------

class DecouplePrior(PriorModule):
    """
    Penalises cross-covariance between latent subspaces (e.g. shape vs colour).
    """

    def __init__(self, split: int, lam: float):
        self.split = split
        self.lam = lam
        self.name = "decouple"

    def energy(self, X, phi_raw, meta):
        split = self.split
        if meta is not None and meta.block_split is not None:
            split = meta.block_split
        return decouple_prior(X, split, self.lam)

    def describe(self):
        return "Subspace decoupling prior (cross-covariance penalty)."


class HierarchicalARDPrior(PriorModule):
    """
    Hierarchical ARD prior: ordered shrinkage across latent dimensions.

    Later dimensions are **more costly** than earlier dimensions.
    Implements an ordered semantic hierarchy across Q.
    Creates semantic coordinate system where early dims are cheap (global/main)
    and later dims are expensive (details/residuals).
    """

    def __init__(self, lam: float, mode: str = "exp", gamma: float = 0.9, power: float = 1.5):
        self.lam = lam
        self.mode = mode
        self.gamma = gamma
        self.power = power
        self.name = "hier_ard"

    def energy(self, X, phi_raw, meta):
        return hierarchical_ard_prior(X, self.lam, self.mode, self.gamma, self.power)

    def describe(self):
        return f"Hierarchical ARD prior (mode={self.mode}, ordered shrinkage)."


class RowSparsityPrior(PriorModule):
    """
    Row-wise group lasso: encourages each position to use few dimensions.

    This is the cleanest "parts prior": different positions can use different
    subsets of dimensions → assembly capability emerges.
    """

    def __init__(self, lam: float, eps: float = 1e-3):
        self.lam = lam
        self.eps = eps
        self.name = "row_sparsity"

    def energy(self, X, phi_raw, meta):
        return row_group_lasso_prior(X, self.lam, self.eps)

    def describe(self):
        return "Row-wise sparsity prior (parts/position-level)."


class ColSparsityPrior(PriorModule):
    """
    Column-wise group lasso: encourages using few dimensions globally.

    Combined with hierarchical ARD:
    - hierarchical = soft ordering / soft shrinkage
    - col sparsity = hard-ish selection
    """

    def __init__(self, lam: float, eps: float = 1e-3):
        self.lam = lam
        self.eps = eps
        self.name = "col_sparsity"

    def energy(self, X, phi_raw, meta):
        return col_group_lasso_prior(X, self.lam, self.eps)

    def describe(self):
        return "Column-wise sparsity prior (global factor selection)."


# ---------------------
# Prior Energy Orchestrator
# ---------------------

class PriorEnergy:
    """
    Collection of structural priors defining the reference measure.

    This class is the ONLY object that inference code should call.
    """

    def __init__(self, modules):
        self.modules = modules

    def energy(self, X, phi_raw, meta=None):
        E = jnp.array(0.0, dtype=X.dtype)
        for m in self.modules:
            E = E + m.energy(X, phi_raw, meta)
        return E

    def summary(self):
        return [m.describe() for m in self.modules]


# ---------------------
# Backward-compatible prior_energy function
# ---------------------

def prior_energy(X, phi_raw, kcfg, pcfg: PriorCFG, meta: Optional[PriorMeta] = None):
    """
    Backward-compatible wrapper using PriorCFG.

    Prefer using PriorEnergy with explicit modules.
    """
    modules = [
        XGaussianPrior(pcfg.x_l2_lambda),
        RepulsionPrior(
            pcfg.repulsion_lambda,
            pcfg.repulsion_p,
            pcfg.repulsion_epsilon,
            pcfg.repulsion_max_pairs,
        ),
        WhitenPrior(pcfg.whiten_lambda),
        LaplacianPrior(pcfg.laplacian_lambda),
        ThinPlatePrior(pcfg.thinplate_lambda),
        TVPrior(pcfg.tv_lambda, pcfg.tv_epsilon),
        DecouplePrior(pcfg.block_split, pcfg.decouple_lambda),
        HierarchicalARDPrior(
            pcfg.hier_ard_lambda,
            pcfg.hier_ard_mode,
            pcfg.hier_ard_gamma,
            pcfg.hier_ard_power,
        ),
        RowSparsityPrior(pcfg.row_sparsity_lambda, pcfg.row_sparsity_epsilon),
        ColSparsityPrior(pcfg.col_sparsity_lambda, pcfg.col_sparsity_epsilon),
    ]

    E = jnp.array(0.0, dtype=X.dtype)
    E += ell_l2_shrinkage(phi_raw, kcfg, pcfg.ell_prior_lambda)

    prior = PriorEnergy(modules)
    return E + prior.energy(X, phi_raw, meta)


__all__ = [
    "PriorCFG",
    "PriorMeta",
    "prior_energy",
    "x_l2_prior",
    "ell_l2_shrinkage",
    "graph_laplacian_prior",
    "thinplate_prior",
    "tv_charbonnier_prior",
    "repulsion_prior",
    "whiten_prior",
    "decouple_prior",
    "hierarchical_ard_prior",
    "row_group_lasso_prior",
    "col_group_lasso_prior",
    "PriorModule",
    "PriorEnergy",
    "XGaussianPrior",
    "RepulsionPrior",
    "WhitenPrior",
    "LaplacianPrior",
    "ThinPlatePrior",
    "TVPrior",
    "DecouplePrior",
    "HierarchicalARDPrior",
    "RowSparsityPrior",
    "ColSparsityPrior",
]


# ============================================================
# ARCHITECTURE DOCUMENTATION
# ============================================================
#
# This module implements a complete "latent physics specification"
# for structural priors on latent variables X (N×Q).
#
# ============================================================
# TREE STRUCTURE (Conceptual × Code Alignment)
# ============================================================
#
# prior_energy.py
# │
# ├── [Design Contract / Philosophy]
# │   ├── HARD CONSTRAINTS (no Y, no kernel, no likelihood, no beta, no dynamics)
# │   └── QUICK START (minimal shape-assembling prior recipe)
# │
# ├── Configuration Layer
# │   ├── PriorCFG: All prior weights (lambda values)
# │   │   ├── Base scale: x_l2_lambda
# │   │   ├── Geometry/shape (Across-N): laplacian, thinplate, tv
# │   │   ├── Anti-collapse: repulsion, whiten
# │   │   ├── Subspace structure: decouple, block_split
# │   │   └── Across-Q semantics (CORE): hier_ard, row_sparsity, col_sparsity
# │   └── PriorMeta: Structural metadata (edges, pairs, block_split)
# │       └── grid(H, W, neighbourhood): Factory for grid topology
# │
# ├── Internal Helpers (private)
# │   ├── _charbonnier, _safe_int
# │   └── _grid_edges_4n, _grid_edges_8n
# │
# ├── Atomic Prior Energies (pure mathematics)
# │   ├── Base: x_l2_prior, ell_l2_shrinkage
# │   ├── Geometry (Across-N): graph_laplacian, thinplate, tv_charbonnier
# │   ├── Anti-collapse: repulsion, whiten
# │   ├── Subspace (Across-Q): decouple
# │   └── Semantics (Across-Q/(N,Q)): hierarchical_ard, row_group_lasso, col_group_lasso
# │
# ├── Prior Modules (semantic wrappers = inductive biases)
# │   ├── PriorModule (abstract base)
# │   ├── Base/stability: XGaussianPrior, RepulsionPrior, WhitenPrior
# │   ├── Geometry/field: LaplacianPrior, ThinPlatePrior, TVPrior
# │   └── Factorisation/semantics: DecouplePrior, HierarchicalARDPrior,
# │                                 RowSparsityPrior, ColSparsityPrior
# │
# ├── PriorEnergy (orchestrator)
# │   ├── energy(X, phi_raw, meta) -> scalar
# │   └── summary() -> List[str]
# │
# ├── Legacy API: prior_energy(X, phi_raw, kcfg, pcfg, meta)
# └── __all__ (explicit public surface, 25 exports)
#
# Interpretation:
#   Top: "Define world and assumptions"
#   Middle: "Define mathematical energies"
#   Bottom: "Turn math into semantic modules for inference systems"
#
# ============================================================
# ARCHITECTURE OVERVIEW TABLES
# ============================================================
#
# 1. Configuration / Metadata
# ┌─────────────────┬──────────┬────────────────────────────┬──────────┐
# │ Component       │ Type     │ Responsibility              │ Touches  │
# │                 │          │                            │ Data?    │
# ├─────────────────┼──────────┼────────────────────────────┼──────────┤
# │ PriorCFG        │ dataclass│ All prior weights (knobs)  │ ❌       │
# │ PriorMeta       │ dataclass│ Latent space topology/     │ ❌       │
# │                 │          │ structural assumptions      │          │
# │ PriorMeta.grid()│ factory  │ Define grid topology       │ ❌       │
# │                 │          │ (4n / 8n)                   │          │
# └─────────────────┴──────────┴────────────────────────────┴──────────┘
#
# 2. Atomic Prior Energies (Mathematical Layer)
# ┌──────────────────────────┬──────────────┬──────────────────┬──────────┐
# │ Function                 │ Operates on  │ Mathematical     │ Semantic │
# │                          │              │ Role             │          │
# ├──────────────────────────┼──────────────┼──────────────────┼──────────┤
# │ x_l2_prior               │ X            │ Baseline Gaussian │ ❌       │
# │ graph_laplacian_prior    │ Across-N     │ First-order      │ ⚠️(shape)│
# │                          │              │ smoothness       │          │
# │ thinplate_prior          │ Across-N     │ Second-order     │ ⚠️       │
# │                          │              │ smoothness       │          │
# │ tv_charbonnier_prior     │ Across-N     │ Piecewise smooth │ ⚠️       │
# │ repulsion_prior          │ Across-N     │ Volume/spacing   │ ❌       │
# │ whiten_prior             │ N×Q          │ Covariance       │ ❌       │
# │                          │              │ control          │          │
# │ decouple_prior           │ Across-Q     │ Block            │ ⚠️       │
# │                          │              │ independence     │          │
# │ hierarchical_ard_prior   │ Across-Q     │ Dimension        │ ✅       │
# │                          │              │ hierarchy        │          │
# │ row_group_lasso_prior    │ Across-(N,Q) │ Parts prior      │ ✅       │
# │ col_group_lasso_prior    │ Across-Q     │ Factor selection │ ✅       │
# └──────────────────────────┴──────────────┴──────────────────┴──────────┘
#
# 3. PriorModule (Semantic Layer / Inductive Bias)
# ┌──────────────────────┬──────────────────────┬──────────────────┬────────┐
# │ Class                 │ Wraps Function       │ Semantic Role    │ Core?  │
# ├──────────────────────┼──────────────────────┼──────────────────┼────────┤
# │ XGaussianPrior        │ x_l2_prior           │ Reference measure│ Base   │
# │ RepulsionPrior        │ repulsion_prior      │ Anti-collapse    │ Important│
# │ WhitenPrior           │ whiten_prior         │ Equal scale      │ Optional│
# │ LaplacianPrior        │ graph_laplacian_prior│ Shape continuity │ Important│
# │ ThinPlatePrior        │ thinplate_prior      │ Curvature penalty│ Optional│
# │ TVPrior               │ tv_charbonnier_prior │ Part boundaries  │ Optional│
# │ DecouplePrior         │ decouple_prior       │ Shape/color      │ Important│
# │ HierarchicalARDPrior  │ hierarchical_ard_    │ Semantic         │ **CORE**│
# │                      │   prior              │ hierarchy        │        │
# │ RowSparsityPrior      │ row_group_lasso_prior│ Assembly ability │ **CORE**│
# │ ColSparsityPrior      │ col_group_lasso_prior│ Dimension        │ Important│
# │                      │                      │ selection        │        │
# └──────────────────────┴──────────────────────┴──────────────────┴────────┘
#
# ============================================================
# API USAGE
# ============================================================
#
# ✅ Recommended API (Research & Future)
# ────────────────────────────────────────────────────────────
# from prior_energy import (
#     PriorEnergy, PriorMeta,
#     XGaussianPrior, HierarchicalARDPrior,
#     RowSparsityPrior, LaplacianPrior
# )
#
# priors = PriorEnergy([
#     XGaussianPrior(1.0),
#     HierarchicalARDPrior(0.1, mode="exp", gamma=0.9),
#     RowSparsityPrior(0.01, eps=1e-3),
#     LaplacianPrior(0.05),
# ])
#
# meta = PriorMeta.grid(H=32, W=32, neighbourhood="4n")
# E_prior = priors.energy(X, phi_raw, meta=meta)
# print(priors.summary())
#
# ⚠️ Legacy API (Compatibility)
# ────────────────────────────────────────────────────────────
# from prior_energy import PriorCFG, PriorMeta, prior_energy
#
# pcfg = PriorCFG(
#     hier_ard_lambda=0.1,
#     row_sparsity_lambda=0.01,
#     laplacian_lambda=0.05,
# )
# meta = PriorMeta.grid(H=32, W=32, neighbourhood="4n")
# E_prior = prior_energy(X, phi_raw, kcfg, pcfg, meta=meta)
#
# ============================================================
# MINIMAL BUT EFFECTIVE CONFIGURATION
# ============================================================
#
# For "shape-assembling" generation, start with these 3 priors:
#
#   pcfg = PriorCFG(
#       hier_ard_lambda=0.1,      # Semantic hierarchy (safe, stabilizes)
#       hier_ard_mode="exp",
#       hier_ard_gamma=0.9,       # Auto-inverted to >1 if needed
#       row_sparsity_lambda=0.01, # Parts assembly (small!)
#       laplacian_lambda=0.05,    # Spatial coherence (moderate)
#   )
#   meta = PriorMeta.grid(H=32, W=32, neighbourhood="4n")
#
# This enables:
#   ✅ Semantic dimension ordering (hierarchical ARD)
#   ✅ Parts-based composition (row sparsity)
#   ✅ Spatial coherence (Laplacian)
#
# ============================================================
# DESIGN PRINCIPLES
# ============================================================
#
# 1. Three-Layer Separation:
#    Configuration → Atomic Functions → Modules
#    (Knobs)        → (Pure Math)     → (Semantic Wrappers)
#
# 2. Dual API Design:
#    - Research API: PriorEnergy + explicit modules (clear semantics)
#    - Experimental API: prior_energy() + PriorCFG (simple & fast)
#
# 3. Hard Constraints (HARD CONSTRAINTS):
#    ❌ NO data Y
#    ❌ NO kernels / GP algebra
#    ❌ NO likelihood terms
#    ❌ NO beta / annealing
#    ❌ NO dynamics (Langevin / SDE / samplers)
#
# 4. Extensibility:
#    To add new prior:
#    1. Implement atomic function (mathematical layer)
#    2. Create PriorModule subclass (semantic layer)
#    3. (Optional) Add to wrapper (compatibility layer)
#
# ============================================================
# SUMMARY
# ============================================================
#
# This module is NOT just a "regularization collection",
# but a complete "latent physics specification" for the generation system.
#
# Completed theoretical foundation:
#   ✅ Three structural directions complete:
#      - Across-N: Spatial/shape coherence
#      - Across-Q: Dimension semantic structure
#      - Across-(N,Q): Parts assembly capability
#   ✅ Clean architecture: No data leakage, clear class/function separation
#   ✅ Theoretically correct: All priors depend only on X, phi_raw, meta
#
# This file is "research-grade" and ready for:
#   ✅ Paper writing (architecture can be used as appendix)
#   ✅ Experimental design (minimal config is clear)
#   ✅ Library core module (API design is complete)
#