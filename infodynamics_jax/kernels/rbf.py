import jax.numpy as jnp

def rbf_kernel(X, X2, lengthscale, variance):
    X = X / lengthscale
    X2 = X2 / lengthscale
    sqdist = jnp.sum(X**2,1)[:,None] + jnp.sum(X2**2,1)[None,:] - 2*jnp.dot(X, X2.T)
    return variance * jnp.exp(-0.5 * sqdist)
