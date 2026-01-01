from typing import Protocol, Dict
import jax.numpy as jnp

class Likelihood(Protocol):
    """
    Likelihood interface.
    """

    def log_prob(
        self,
        y: jnp.ndarray,
        f: jnp.ndarray,
        params: Dict,
    ) -> jnp.ndarray:
        """
        Return log p(y | f, params).

        Shapes:
          y: (..., N)
          f: (..., N)
        Returns:
          log_prob: (..., N)
        """
        ...