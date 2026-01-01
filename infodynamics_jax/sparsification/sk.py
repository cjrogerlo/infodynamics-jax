from dataclasses import dataclass
import jax.numpy as jnp

from .nystrom import Q_ff
from .residuals import fitc

@dataclass
class SparsifiedKernel:
    kernel_fn: callable
    residual: str = "fitc"

    def S_ff(self, params, X, Z):
        K_xx = self.kernel_fn(params, X, X)
        K_xz = self.kernel_fn(params, X, Z)
        K_zz = self.kernel_fn(params, Z, Z)

        Q = Q_ff(K_xz, K_zz)

        if self.residual == "fitc":
            R = fitc(K_xx, Q)
        else:
            R = jnp.zeros_like(Q)

        return Q + R