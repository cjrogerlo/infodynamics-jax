# infodynamics_jax/gp/likelihoods/gaussian.py
import jax.numpy as jnp

class GaussianLikelihood:
    """
    Gaussian likelihood:
        p(y | f, phi) = N(y; f, sigma^2)
    """

    @staticmethod
    def neg_loglik_1d(y, f, phi_like):
        """
        Negative log-likelihood for one scalar observation.

        phi_like:
            {"noise_var": sigma^2}
        """
        sigma2 = phi_like["noise_var"]
        return 0.5 * (
            jnp.log(2.0 * jnp.pi * sigma2)
            + (y - f) ** 2 / sigma2
        )

gaussian = GaussianLikelihood()
# Tag the function for analytic fast-path detection in expected energy computation
GaussianLikelihood.neg_loglik_1d._likelihood = "gaussian"

# Mark that this likelihood supports analytic marginalisation
GaussianLikelihood.supports_analytic_marginal = True
