import jax
import jax.numpy as jnp

from infodynamics_jax.core.phi import Phi
from infodynamics_jax.kernels.params import KernelParams
from infodynamics_jax.energy import InertialEnergy
from infodynamics_jax.energy.expected import VariationalState

def test_energy_is_finite():
    key = jax.random.PRNGKey(0)

    X = jnp.linspace(0, 1, 12)[:, None]
    y = jnp.sin(6 * X)

    phi = Phi(
        kernel_params=KernelParams(lengthscale=jnp.array(0.3)),
        Z=jnp.linspace(0, 1, 6)[:, None],
        likelihood_params={"noise_var": jnp.array(0.05)},
    )

    state = VariationalState(
        m_u=jnp.zeros((6, 1)),
        L_u=jnp.eye(6),
        cov_type="full",
    )

    energy = InertialEnergy(
        kernel="rbf",
        likelihood="gaussian",
        estimator="mc",
        mc_samples=4,
    )

    E = energy(phi, X, y, state=state, key=key)
    assert jnp.isfinite(E)
