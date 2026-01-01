import jax.numpy as jnp

def log_prob(y, f, params):
    """
    Ordinal probit/logistic likelihood.

    params:
      thresholds: (K-1,)
    """
    thresholds = params["thresholds"]

    # Placeholder: actual implementation can be probit or logistic
    # This is intentionally left abstract.
    raise NotImplementedError("Ordinal likelihood not yet implemented.")