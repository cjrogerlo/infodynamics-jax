import jax.numpy as jnp

def gaussian_inertial_energy(y, S_ff, noise_var):
    """
    Prior-predictive inertial energy:
      E = E_{p(f|φ)}[-log p(y|f)]
    """
    N = y.shape[0]
    y2 = jnp.sum(y * y)
    trS = jnp.trace(S_ff)
    return 0.5 * (y2 + trS) / noise_var + 0.5 * N * jnp.log(2.0 * jnp.pi * noise_var)