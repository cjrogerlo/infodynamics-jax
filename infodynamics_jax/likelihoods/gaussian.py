import jax.numpy as jnp

class GaussianLikelihood:
    def __init__(self, noise_variance):
        self.noise_variance = noise_variance

    def log_prob(self, y, f):
        return -0.5 * ((y-f)**2 / self.noise_variance + jnp.log(2*jnp.pi*self.noise_variance))
