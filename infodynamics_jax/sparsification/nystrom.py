import jax.numpy as jnp

def nystrom(K_xz, K_zz, jitter=1e-6):
    Kzz = K_zz + jitter * jnp.eye(K_zz.shape[0])
    L = jnp.linalg.cholesky(Kzz)
    A = jnp.linalg.solve(L, K_xz.T)
    return A.T @ A
