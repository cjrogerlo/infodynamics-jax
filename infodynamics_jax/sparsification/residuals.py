import jax.numpy as jnp

def fitc(K_xx, Q_xx):
    """
    Diagonal residual
    """
    return jnp.diag(jnp.diag(K_xx - Q_xx))