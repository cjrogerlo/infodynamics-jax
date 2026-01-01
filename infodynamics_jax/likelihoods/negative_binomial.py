import jax.numpy as jnp

def log_prob(y, f, params):
    """
    Negative Binomial with log-mean parameterisation.

    params:
      r: dispersion (>0)
    """
    r = params["r"]
    mu = jnp.exp(f)

    return (
        jnp.lgamma(y + r)
        - jnp.lgamma(r)
        - jnp.lgamma(y + 1)
        + r * jnp.log(r)
        + y * jnp.log(mu)
        - (y + r) * jnp.log(mu + r)
    )