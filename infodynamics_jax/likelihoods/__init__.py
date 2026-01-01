from .base import register, get

from .bernoulli import bernoulli
from .poisson import poisson
from .negative_binomial import negative_binomial
from .ordinal import ordinal
from .gaussian import gaussian

register("bernoulli", bernoulli)
register("poisson", poisson)
register("negative_binomial", negative_binomial)
register("ordinal", ordinal)
register("gaussian", gaussian)

__all__ = ["get"]
