# infodynamics_jax/gp/likelihoods/ordinal.py
import jax.numpy as jnp
import jax.nn as jnn

class OrdinalLikelihood:
    """
    Ordinal likelihood with cumulative logit model.

    p(y=k | f) = sigmoid(b_k - f) - sigmoid(b_{k-1} - f)
    """

    @staticmethod
    def neg_loglik_1d(y, f, phi_like):
        """
        y in {0, ..., K-1}

        phi_like:
            {"thresholds": b}, shape (K-1,)
        """
        b = phi_like["thresholds"]

        # Extend thresholds with -inf, +inf
        b_ext = jnp.concatenate([
            jnp.array([-jnp.inf]),
            b,
            jnp.array([jnp.inf])
        ])

        upper = jnn.sigmoid(b_ext[y + 1] - f)
        lower = jnn.sigmoid(b_ext[y] - f)

        prob = upper - lower
        return -jnp.log(prob + 1e-12)

ordinal = OrdinalLikelihood()
