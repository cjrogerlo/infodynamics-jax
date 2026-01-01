import jax.numpy as jnp
import jax.nn as jnn

class NegativeBinomialLikelihood:
    """
    Negative Binomial likelihood with log-mean parameterisation.

    mean = exp(f)
    dispersion = r
    """

    @staticmethod
    def neg_loglik_1d(y, f, phi_like):
        """
        phi_like:
            {"dispersion": r}
        """
        r = phi_like["dispersion"]
        mu = jnp.exp(f)

        return (
            jnn.lgamma(y + r)
            - jnn.lgamma(r)
            - jnn.lgamma(y + 1.0)
            + r * jnp.log(r / (r + mu))
            + y * jnp.log(mu / (r + mu))
        ) * (-1.0)