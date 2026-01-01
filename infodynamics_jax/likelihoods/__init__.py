# infodynamics_jax/likelihoods/__init__.py

from .base import register, get

from .bernoulli import bernoulli
from .poisson import poisson
from .negative_binomial import negative_binomial
from .ordinal import ordinal

register("bernoulli", bernoulli)
register("poisson", poisson)
register("negative_binomial", negative_binomial)
register("ordinal", ordinal)

__all__ = [
    "get",
    "bernoulli",
    "poisson",
    "negative_binomial",
    "ordinal",
]

