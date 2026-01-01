import jax.numpy as jnp

def log_prob(y, f, params):
    """
    Gaussian likelihood:
      y | f ~ N(f, sigma^2)

    params:
      noise_var: scalar
    """
    sigma2 = params["noise_var"]
    return (
        -0.5 * jnp.log(2.0 * jnp.pi * sigma2)
        -0.5 * (y - f) ** 2 / sigma2
    )