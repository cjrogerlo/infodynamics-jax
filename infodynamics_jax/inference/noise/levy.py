# infodynamics_jax/inference/noise/levy.py
"""
α-stable Lévy noise for stochastic inference dynamics.

This module implements the Chambers-Mallows-Stuck (CMS) algorithm for
sampling from symmetric α-stable distributions.

Mathematical Background
-----------------------
α-stable distributions are a family of heavy-tailed distributions:
  - α = 2: Gaussian (lightest tails)
  - α = 1: Cauchy (heavy tails)
  - α < 2: Lévy (intermediate, heavy tails)

For symmetric α-stable (skewness β = 0, location μ = 0):
  X ~ S_α(σ, 0, 0)

The CMS algorithm generates samples via:
  X = σ * [sin(αU) / cos(U)^{1/α}] * [cos(U - αU) / W]^{(1-α)/α}

where U ~ Uniform(-π/2, π/2) and W ~ Exponential(1).

Special Case: α = 2
-------------------
When α = 2, the α-stable distribution is Gaussian with variance 2σ².
We handle this case separately for numerical stability.

Reference:
    Chambers, J. M., Mallows, C. L., & Stuck, B. W. (1976).
    A method for simulating stable random variables.
    Journal of the american statistical association, 71(354), 340-344.
"""
import jax.numpy as jnp
import jax.random as jrand
from jax import lax

EPS = 1e-12


def alpha_stable_noise(
    key,
    shape,
    alpha: float = 1.6,
    scale: float = 1.0,
    dtype=jnp.float32,
):
    """
    Sample from symmetric α-stable Lévy distribution.

    Uses the Chambers-Mallows-Stuck algorithm for efficient sampling.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    shape : tuple
        Output shape.
    alpha : float
        Stability parameter, 0 < α ≤ 2.
        - α = 2: Gaussian
        - α = 1: Cauchy
        - α < 2: Heavy-tailed Lévy
    scale : float
        Scale parameter σ > 0.
    dtype : jnp.dtype
        Output dtype.

    Returns
    -------
    samples : jnp.ndarray
        Array of shape `shape` with α-stable samples.

    Notes
    -----
    - For α = 2, returns Gaussian samples scaled by sqrt(2) * scale.
    - For α ≠ 2, uses CMS algorithm.
    """
    a = jnp.array(alpha, dtype=dtype)
    s = jnp.array(scale, dtype=dtype)

    key_u, key_w = jrand.split(key, 2)
    U = jrand.uniform(key_u, shape, minval=-0.5 * jnp.pi, maxval=0.5 * jnp.pi, dtype=dtype)
    W = jrand.exponential(key_w, shape, dtype=dtype) + EPS

    # Handle α = 2 separately (Gaussian case)
    def _gaussian_case(_):
        # α-stable with α=2 is N(0, 2σ²), so X ~ σ√2 * N(0,1)
        sigma = s * jnp.sqrt(jnp.array(2.0, dtype=dtype))
        return sigma * jrand.normal(key_u, shape, dtype=dtype)

    # Handle α = 1 separately (Cauchy case)
    def _cauchy_case(_):
        return s * jnp.tan(U)

    # General α-stable case
    def _levy_case(_):
        cosU = jnp.clip(jnp.cos(U), EPS, 1.0)

        # CMS formula
        num = jnp.sin(a * U)
        den = cosU ** (1.0 / a)
        term1 = num / (den + EPS)

        angle = (1.0 - a) * U
        cosA = jnp.clip(jnp.cos(angle), EPS, 1.0)
        term2 = (cosA / W) ** ((1.0 - a) / a)

        return s * term1 * term2

    # Dispatch based on alpha value
    is_gaussian = jnp.isclose(a, 2.0)
    is_cauchy = jnp.isclose(a, 1.0)

    # Use nested lax.cond for dispatch
    def _not_gaussian(_):
        return lax.cond(is_cauchy, _cauchy_case, _levy_case, None)

    result = lax.cond(is_gaussian, _gaussian_case, _not_gaussian, None)

    return jnp.nan_to_num(result)


def levy_noise(
    key,
    shape,
    alpha: float = 1.6,
    scale: float = 1.0,
    dt: float = 1.0,
    dtype=jnp.float32,
):
    """
    Lévy noise for Langevin dynamics at timestep dt.

    For Lévy-driven Langevin dynamics:
        dX_t = -∇E(X_t) dt + σ dL_t^α

    The noise increment scales as dt^{1/α}:
        ΔL ~ dt^{1/α} * S_α(σ)

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    shape : tuple
        Output shape.
    alpha : float
        Stability parameter (0 < α ≤ 2).
    scale : float
        Base scale parameter σ.
    dt : float
        Time step size.
    dtype : jnp.dtype
        Output dtype.

    Returns
    -------
    noise : jnp.ndarray
        Lévy noise increment scaled for timestep dt.
    """
    a = jnp.array(alpha, dtype=dtype)
    dt_jax = jnp.array(dt, dtype=dtype)

    # Scale factor for Lévy process: dt^{1/α}
    dt_scale = dt_jax ** (1.0 / a)

    return dt_scale * alpha_stable_noise(key, shape, alpha, scale, dtype)


# Backward-compatible alias
levy = alpha_stable_noise
