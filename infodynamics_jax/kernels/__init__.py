from .base import register, get

from .rbf import rbf
from .matern12 import matern12
from .matern32 import matern32
from .matern52 import matern52
from .linear import linear
from .polynomial import polynomial
from .periodic import periodic
from .rational_quadratic import rational_quadratic
from .white import white

register("rbf", rbf)
register("matern12", matern12)
register("matern32", matern32)
register("matern52", matern52)
register("linear", linear)
register("polynomial", polynomial)
register("periodic", periodic)
register("rq", rational_quadratic)
register("white", white)

__all__ = ["get"]