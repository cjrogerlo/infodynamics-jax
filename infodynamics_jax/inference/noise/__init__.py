# infodynamics_jax/inference/noise/__init__.py
"""
Noise distributions for inference dynamics.

This module provides noise samplers for various stochastic dynamics:
  - levy: α-stable Lévy noise (generalizes Gaussian)
  - gaussian: Standard Gaussian noise (convenience)
"""
from .levy import alpha_stable_noise, levy_noise

__all__ = [
    "alpha_stable_noise",
    "levy_noise",
]
