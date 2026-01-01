import jax.numpy as jnp

def log_prob(y, f, params=None):
    """
    Bernoulli likelihood with logistic link:
      p(y=1|f) = sigmoid(f)

    params: unused (kept for uniform API)
    """
    return y * f - jnp.log1p(jnp.exp(f))