import jax.numpy as jnp

def inertial_energy(y, S_ff, likelihood):
    # Gaussian likelihood only (v0.1)
    sigma2 = likelihood.noise_variance
    return 0.5 * jnp.sum(y**2 / sigma2) + 0.5 * jnp.trace(S_ff) / sigma2
