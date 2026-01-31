# infodynamics_jax/gp/ansatz/state.py
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

    @classmethod
    def initialise(cls, phi, X, Y=None):
        """
        Initialize VariationalState from phi and X.
        
        Parameters
        ----------
        phi : Upphi
            Structural parameters containing inducing points Z
        X : jnp.ndarray
            Input data (N, Q)
        Y : jnp.ndarray, optional
            Output data (N,) or (N, D). If provided, used to determine output dimension D.
            If None, defaults to D=1.
        
        Returns
        -------
        VariationalState
            Initialized state with zero mean and identity covariance
        """
        M = phi.Z.shape[0]  # number of inducing points
        
        # Determine output dimension D
        if Y is not None:
            Y_flat = jnp.atleast_1d(Y)
            D = Y_flat.shape[-1] if Y_flat.ndim > 1 else 1
        else:
            D = 1
        
        return cls(
            m_u=jnp.zeros((M, D)),
            L_u=jnp.eye(M),
            cov_type="full",
        )

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

