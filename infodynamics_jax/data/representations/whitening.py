from __future__ import annotations
from typing import Dict, Tuple

import jax.numpy as jnp


def whiten_features(Y: jnp.ndarray, eps: float = 1e-6) -> Tuple[jnp.ndarray, Dict]:
    """Simple per-dimension whitening for observation features."""
    mu = jnp.mean(Y, axis=0, keepdims=True)
    std = jnp.std(Y, axis=0, keepdims=True) + eps
    Yw = (Y - mu) / std
    return Yw, {"mu": mu, "std": std}
