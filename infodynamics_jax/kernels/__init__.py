# infodynamics_jax/kernels/__init__.py

from .base import register, get

# primitive kernels
from .rbf import rbf
from .matern12 import matern12
from .matern32 import matern32
from .matern52 import matern52
from .linear import linear
from .polynomial import polynomial
from .periodic import periodic
from .rational_quadratic import rational_quadratic
from .white import white

# composite kernels
from .composite import sum_kernel, product_kernel, scale_kernel

# --------------------------------------------------
# Registry
# --------------------------------------------------
register("rbf", rbf)
register("matern12", matern12)
register("matern32", matern32)
register("matern52", matern52)
register("linear", linear)
register("polynomial", polynomial)
register("periodic", periodic)
register("rq", rational_quadratic)
register("white", white)

# composites (optional but nice)
register("sum", sum_kernel)
register("product", product_kernel)
register("scale", scale_kernel)

__all__ = [
    "get",
    "rbf",
    "matern12",
    "matern32",
    "matern52",
    "linear",
    "polynomial",
    "periodic",
    "rational_quadratic",
    "white",
    "sum_kernel",
    "product_kernel",
    "scale_kernel",
]
