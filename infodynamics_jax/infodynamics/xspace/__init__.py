"""
X-space dynamics kernels (array-only).

These are intended for latent/image spaces rather than hyperparameter pytrees.
"""

from .levy import levy_rejuvenate

__all__ = ["levy_rejuvenate"]
