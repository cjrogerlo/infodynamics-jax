import jax.numpy as jnp

def log_prob(y, f, params=None):
    """
    Poisson likelihood with log link:
      rate = exp(f)
    """
    return y * f - jnp.exp(f)