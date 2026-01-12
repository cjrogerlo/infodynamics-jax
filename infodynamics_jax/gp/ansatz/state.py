# infodynamics_jax/energy/state.py
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
@dataclass
class VariationalState:
    """
    Variational Gaussian posterior over inducing variables:
        q(u) = N(m_u, S_u)
    """
    m_u: jnp.ndarray
    L_u: Optional[jnp.ndarray] = None
    s_u_diag: Optional[jnp.ndarray] = None
    cov_type: str = "full"   # static metadata

    def tree_flatten(self):
        # ONLY numerical leaves
        children = (self.m_u, self.L_u, self.s_u_diag)
        aux = self.cov_type
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        m_u, L_u, s_u_diag = children
        return cls(
            m_u=m_u,
            L_u=L_u,
            s_u_diag=s_u_diag,
            cov_type=aux,
        )

