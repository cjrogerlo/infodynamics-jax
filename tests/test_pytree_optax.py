import jax
import jax.numpy as jnp
import optax

from infodynamics_jax.core.phi import Phi
from infodynamics_jax.kernels.params import KernelParams
from infodynamics_jax.energy.expected import VariationalState

def test_pytree_and_optax():
    phi = Phi(
        kernel_params=KernelParams(),
        Z=jnp.zeros((5, 1)),
        likelihood_params={"noise_var": jnp.array(0.1)},
    )

    state = VariationalState(
        m_u=jnp.zeros((5, 1)),
        L_u=jnp.eye(5),
        cov_type="full",
    )

    # pytree map should work
    _ = jax.tree_util.tree_map(lambda x: x, phi)
    _ = jax.tree_util.tree_map(lambda x: x, state)

    # optax should accept params
    params = {"m_u": state.m_u, "L_u": state.L_u}
    opt = optax.adam(1e-2)
    opt_state = opt.init(params)

    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    _ = opt.update(grads, opt_state)
