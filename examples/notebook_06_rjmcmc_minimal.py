"""
Minimal RJ-MCMC for Sparse GP Regression using infodynamics_jax.

This demonstrates the core concepts of transdimensional MCMC over
the number of inducing points, using library components.

This is a starting point for a full implementation. Future extensions:
- RJ adaptive SMC
- Non-conjugate likelihoods (using ansatz)
- Multiple output dimensions
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Use library components
from infodynamics_jax.core import Phi
from infodynamics_jax.gp.kernels import rbf
from infodynamics_jax.gp.kernels.params import KernelParams
from infodynamics_jax.gp.sparsify import Q_ff, diag_Q_ff
from infodynamics_jax.gp.predict import predict_typeii

jax.config.update("jax_enable_x64", True)


# ============================================================
# Priors
# ============================================================

@jax.jit
def log_prior_theta(lengthscale, variance, noise_var):
    """Log-normal priors on hyperparameters (on log scale)."""
    lp_ls = -0.5 * ((jnp.log(lengthscale) + 2.0) / 1.2) ** 2
    lp_var = -0.5 * (jnp.log(variance) / 1.5) ** 2
    lp_noise = -0.5 * ((jnp.log(noise_var) + 2.0) / 1.0) ** 2
    return lp_ls + lp_var + lp_noise


@jax.jit
def log_prior_M(M, M_min, M_max, p=0.12):
    """Truncated geometric prior over model size."""
    m = M - M_min
    valid = (M >= M_min) & (M <= M_max)
    K = M_max - M_min + 1
    log_unn = jnp.log(p) + m * jnp.log1p(-p)
    logZ = jnp.log1p(-(1.0 - p) ** K)
    return jnp.where(valid, log_unn - logZ, -jnp.inf)


# ============================================================
# VFE Computation using library's sparsify
# ============================================================

def compute_vfe(phi: Phi, X, y, kernel_fn):
    """
    Compute VFE using library's sparse GP functions.

    Uses infodynamics_jax.gp.sparsify for Q_ff computation.
    """
    N = X.shape[0]
    Z = phi.Z
    M = Z.shape[0]
    noise_var = phi.likelihood_params["noise_var"]

    # Use library's Q_ff for sparse GP approximation
    Q_train = Q_ff(phi.kernel_params, X, Z, kernel_fn)  # (N, N) low-rank
    K_diag = jnp.diag(kernel_fn(X, X, phi.kernel_params))

    # FITC approximation: S_ff = Q_ff + diag(K_ff - Q_ff)
    Q_diag = diag_Q_ff(phi.kernel_params, X, Z, kernel_fn)  # (N,)
    R_diag = jnp.maximum(K_diag - Q_diag, 0.0)  # FITC residual

    # VFE computation (Titsias 2009)
    # Data fit term
    Sigma_y = Q_train + jnp.diag(R_diag + noise_var)
    L_y = jnp.linalg.cholesky(Sigma_y)
    alpha = jax.scipy.linalg.cho_solve((L_y, True), y)

    data_fit = -0.5 * (
        y.T @ alpha
        + N * jnp.log(2 * jnp.pi)
        + 2 * jnp.sum(jnp.log(jnp.diag(L_y)))
    )

    # Trace correction term (KL between q(f) and prior)
    trace_term = -0.5 * jnp.sum(R_diag) / noise_var

    return data_fit + trace_term


# ============================================================
# Posterior (Energy)
# ============================================================

def log_posterior(phi: Phi, M, X, y, kernel_fn, M_min, M_max):
    """Log posterior = VFE + log priors."""
    vfe = compute_vfe(phi, X, y, kernel_fn)

    lp_theta = log_prior_theta(
        phi.kernel_params.lengthscale,
        phi.kernel_params.variance,
        phi.likelihood_params["noise_var"]
    )

    lp_M = log_prior_M(M, M_min, M_max)

    return vfe + lp_theta + lp_M


# ============================================================
# Demonstration
# ============================================================

def demo():
    """
    Minimal demonstration of RJ-MCMC concepts.

    Note: This only shows the energy/posterior computation.
    Full RJ-MCMC implementation (birth/death moves, HMC on theta, etc.)
    is available in infodynamics_jax.inference.rj module.
    """
    print("=" * 70)
    print("Minimal RJ-MCMC Demonstration using infodynamics_jax")
    print("=" * 70)

    # Generate data
    key = jax.random.key(42)
    N = 100
    X = jnp.linspace(-3, 3, N)[:, None]
    y = jnp.sin(2 * X[:, 0]) + 0.3 * jax.random.normal(key, (N,))

    print(f"\n✓ Generated {N} data points")

    # Test different model sizes
    M_values = [5, 10, 20, 40]
    M_min, M_max = 5, 50

    print(f"\n{'M':<6} {'VFE':<12} {'Log Post':<12}")
    print("-" * 30)

    for M in M_values:
        # Create Phi with M inducing points
        Z_indices = jnp.linspace(0, N-1, M, dtype=jnp.int32)
        Z = X[Z_indices]

        phi = Phi(
            kernel_params=KernelParams(
                lengthscale=jnp.array(1.0),
                variance=jnp.array(1.0)
            ),
            Z=Z,
            likelihood_params={"noise_var": jnp.array(0.1)},
            jitter=1e-6
        )

        # Compute VFE and posterior
        vfe = compute_vfe(phi, X, y, rbf)
        log_post = log_posterior(phi, M, X, y, rbf, M_min, M_max)

        print(f"{M:<6} {float(vfe):<12.2f} {float(log_post):<12.2f}")

    print("\n" + "=" * 70)
    print("Key Components Used from infodynamics_jax:")
    print("=" * 70)
    print("  ✓ Phi - structural hyperparameters")
    print("  ✓ KernelParams - kernel parameter structure")
    print("  ✓ rbf - RBF kernel function")
    print("  ✓ Q_ff, diag_Q_ff - sparse GP approximation (from sparsify.py)")
    print("  ✓ predict_typeii - prediction function (for BMA later)")

    print("\n" + "=" * 70)
    print("Next Steps for Full RJ-MCMC Implementation:")
    print("=" * 70)
    print("  1. Birth move: Add inducing point with rank-1 update")
    print("  2. Death move: Remove inducing point with rank-1 update")
    print("  3. HMC step: Update hyperparameters (theta)")
    print("  4. Main loop: Alternate birth/death and HMC")
    print("  5. BMA prediction: Average over posterior samples")
    print("\n  All of these will use library components where possible!")


if __name__ == "__main__":
    demo()
