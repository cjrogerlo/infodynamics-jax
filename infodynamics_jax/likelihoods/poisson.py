import jax.numpy as jnp
import jax.nn as jnn

class PoissonLikelihood:
    """
    Poisson likelihood with log link:
        p(y | f) = Poisson(exp(f))
    """

    @staticmethod
    def neg_loglik_1d(y, f, phi_like=None):
        """
        y >= 0 integer
        """
        rate = jnp.exp(f)
        return rate - y * f + jnn.lgamma(y + 1.0)