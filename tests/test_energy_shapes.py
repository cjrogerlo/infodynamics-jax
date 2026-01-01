import jax
import jax.numpy as jnp

from infodynamics_jax.core.phi import Phi
from infodynamics_jax.kernels.params import KernelParams
from infodynamics_jax.energy import InertialEnergy
from infodynamics_jax.energy.expected import VariationalState

def _make_problem(D):
    N, M = 8, 6
    X = jnp.linspace(0, 1, N)[:, None]
    Y = jnp.ones((N, D))

    phi = Phi(
        kernel_params=KernelParams(),
        Z=jnp.linspace(0, 1, M)[:, None],
        likelihood_params={"noise_var": jnp.array(0.1)},
    )

    state = VariationalState(
        m_u=jnp.zeros((M, D)),
        L_u=jnp.eye(M),
        cov_type="full",
    )
    return phi, X, Y, state

def test_energy_mc_and_gh_shapes():
    for D in [1, 2]:
        phi, X, Y, state = _make_problem(D)

        # MC
        energy_mc = InertialEnergy(
            kernel="rbf",
            likelihood="gaussian",
            estimator="mc",
            mc_samples=4,
        )
        key = jax.random.PRNGKey(0)
        E_mc = energy_mc(phi, X, Y, state=state, key=key)
        assert E_mc.shape == ()

        # GH (may fallback internally)
        energy_gh = InertialEnergy(
            kernel="rbf",
            likelihood="gaussian",
            estimator="gh",
        )
        E_gh = energy_gh(phi, X, Y, state=state)
        assert E_gh.shape == ()
