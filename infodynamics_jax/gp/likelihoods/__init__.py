# infodynamics_jax/gp/likelihoods/__init__.py

from .base import register, get

from .gaussian import gaussian
from .bernoulli import bernoulli
from .poisson import poisson
from .negative_binomial import negative_binomial
from .ordinal import ordinal

# --------------------------------------------------
# Registry
# --------------------------------------------------
register("gaussian", gaussian)
register("bernoulli", bernoulli)
register("poisson", poisson)
register("negative_binomial", negative_binomial)
register("ordinal", ordinal)

__all__ = ["get"]
