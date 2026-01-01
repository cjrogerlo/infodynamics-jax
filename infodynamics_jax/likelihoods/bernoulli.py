import jax.numpy as jnp
import jax.nn as jnn

class BernoulliLikelihood:
    """
    Bernoulli likelihood with logistic link:
        p(y | f) = Bernoulli(sigmoid(f))
    """

    @staticmethod
    def neg_loglik_1d(y, f, phi_like=None):
        """
        y in {0,1}
        """
        # Stable binary cross entropy
        return jnn.softplus(f) - y * f

bernoulli = BernoulliLikelihood()
