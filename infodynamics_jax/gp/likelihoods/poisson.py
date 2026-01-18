# infodynamics_jax/gp/likelihoods/poisson.py
import jax.numpy as jnp
import jax.scipy.special as jsp

class PoissonLikelihood:
    @staticmethod
    def neg_loglik_1d(y, f, phi_like=None):
        rate = jnp.exp(f)
        return rate - y * f + jsp.gammaln(y + 1.0)

poisson = PoissonLikelihood()

