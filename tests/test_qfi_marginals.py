import jax.numpy as jnp

from infodynamics_jax.core.phi import Phi
from infodynamics_jax.kernels.params import KernelParams
from infodynamics_jax.energy.expected import qfi_from_qu_full

def test_qfi_shapes():
    N, M, D = 10, 7, 2

    X = jnp.linspace(0, 1, N)[:, None]
    Z = jnp.linspace(0, 1, M)[:, None]

    phi = Phi(
        kernel_params=KernelParams(),
        Z=Z,
        likelihood_params={},
    )

    m_u = jnp.zeros((M, D))
    L_u = jnp.eye(M)

    from infodynamics_jax.kernels import get
    kernel_fn = get("rbf")

    mu, var = qfi_from_qu_full(phi, X, kernel_fn, m_u, L_u)

    assert mu.shape == (N, D)
    assert var.shape == (N, D)
