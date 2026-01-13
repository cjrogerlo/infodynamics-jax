# infodynamics_jax/core/data.py
"""
Data view layer.

This module provides functional data views for energy functions.
It deliberately contains no model assumptions, no inference logic,
and no probabilistic decisions.

Design principle:
  Data is input to energy, but not part of inference state.
  These are containers that provide views (batch, prefix) for
  different inference scenarios (supervised, latent, online, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp


@dataclass(frozen=True)
class SupervisedData:
    """
    Supervised learning data view.
    
    For standard GP regression / classification:
    - X: inputs (N, Q)
    - Y: observations (N,) or (N, D)
    
    This is a functional view container. It does not make any
    probabilistic assumptions or decisions.
    """
    X: jnp.ndarray  # (N, Q)
    Y: jnp.ndarray  # (N,) or (N, D)

    def batch(self, idx: Union[jnp.ndarray, slice]) -> SupervisedData:
        """
        Create a batched view.
        
        Args:
            idx: Index array or slice
        
        Returns:
            New SupervisedData with selected samples
        """
        return SupervisedData(self.X[idx], self.Y[idx])

    def prefix(self, t: int) -> SupervisedData:
        """
        Create a prefix view (first t samples).
        
        Useful for:
        - IBIS / online inference
        - Annealed SMC with sequential data
        - Streaming inference
        
        Args:
            t: Number of samples to include (0 < t <= N)
        
        Returns:
            New SupervisedData with first t samples
        """
        return SupervisedData(self.X[:t], self.Y[:t])

    def __len__(self) -> int:
        """Return number of samples."""
        return self.X.shape[0]


# Backward-compatible alias for older code and scripts.
SupervisedDataset = SupervisedData


@dataclass(frozen=True)
class LatentData:
    """
    Latent variable model data view.
    
    For GPLVM / LLGP / EGPF:
    - Y: observed data (N, D)
    - X_init: optional initial guess for latent X (N, Q)
    
    Note: X is latent and will be inferred, so it's not stored here.
    Only the observed Y and optional initialization are provided.
    """
    Y: jnp.ndarray  # (N, D)
    X_init: Optional[jnp.ndarray] = None  # (N, Q) optional initial guess

    def batch(self, idx: Union[jnp.ndarray, slice]) -> LatentData:
        """
        Create a batched view.
        
        Args:
            idx: Index array or slice
        
        Returns:
            New LatentData with selected samples
        """
        X_init_batch = self.X_init[idx] if self.X_init is not None else None
        return LatentData(self.Y[idx], X_init_batch)

    def prefix(self, t: int) -> LatentData:
        """
        Create a prefix view (first t samples).
        
        Args:
            t: Number of samples to include (0 < t <= N)
        
        Returns:
            New LatentData with first t samples
        """
        X_init_prefix = self.X_init[:t] if self.X_init is not None else None
        return LatentData(self.Y[:t], X_init_prefix)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.Y.shape[0]


__all__ = [
    "SupervisedData",
    "SupervisedDataset",  # Backward-compatible alias
    "LatentData",
]
