from .rbf import rbf
from .base import register, get

register("rbf", rbf)

__all__ = ["get", "rbf"]