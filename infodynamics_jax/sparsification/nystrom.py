import jax.numpy as jnp

def Q_ff(K_xz, K_zz, jitter=1e-6):
    """
    K_xz: (N,M)
    K_zz: (M,M)
    return: (N,N)
    """
    Kzz = K_zz + jitter * jnp.eye(K_zz.shape[0])
    L = jnp.linalg.cholesky(Kzz)
    A = jnp.linalg.solve(L, K_xz.T)      # (M,N)
    return A.T @ A