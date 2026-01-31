"""
Public reference implementations of plugin interfaces.

These serve as:
1. Examples for plugin developers
2. Baseline implementations for testing
3. Fallback options when proprietary plugins unavailable

All implementations here use only public data sources and
standard statistical models.
"""

import numpy as np
import jax.numpy as jnp
from jax import Array, grad, random
from typing import Dict, Any, Optional, Tuple

from .interfaces import (
    DataStreamPlugin,
    ObservationEnergyPlugin,
    RiskEvaluatorPlugin,
)
from .registry import register_plugin


class PublicCSVDataStream(DataStreamPlugin):
    """
    Simple CSV data stream for public examples.

    This is the baseline implementation that works with
    standard CSV files (time, observation columns).
    """

    def __init__(self, file_path: str, obs_columns: list[str]):
        """
        Args:
            file_path: Path to CSV file
            obs_columns: Column names for observations
        """
        import pandas as pd

        self.data = pd.read_csv(file_path)
        self.obs_columns = obs_columns
        self.current_idx = 0

        # Validate columns exist
        missing = set(obs_columns) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

    def next(self) -> Tuple[float, Array, Optional[Dict[str, Any]]]:
        if self.current_idx >= len(self.data):
            raise StopIteration

        row = self.data.iloc[self.current_idx]

        # Extract time (assume first column or 'time'/'t' column)
        if 'time' in self.data.columns:
            t = float(row['time'])
        elif 't' in self.data.columns:
            t = float(row['t'])
        else:
            t = float(self.current_idx)  # Use index as time

        # Extract observations
        y_t = jnp.array([row[col] for col in self.obs_columns])

        # Optional metadata (remaining columns)
        meta_cols = set(self.data.columns) - set(self.obs_columns) - {'time', 't'}
        meta = {col: row[col] for col in meta_cols} if meta_cols else None

        self.current_idx += 1
        return t, y_t, meta

    def reset(self) -> None:
        self.current_idx = 0

    @property
    def observation_dim(self) -> int:
        return len(self.obs_columns)


class GaussianObservationEnergy(ObservationEnergyPlugin):
    """
    Standard Gaussian observation model: E_obs = ||y - f(x)||² / (2σ²)

    This is the baseline implementation used in most examples.
    Private projects may replace with Student-t, mixture models, etc.
    """

    def __init__(self, observation_fn: Optional[callable] = None):
        """
        Args:
            observation_fn: Optional function to map latent state to observations.
                          If None, assumes direct observation (y = x + noise).
        """
        self.observation_fn = observation_fn or (lambda x, g: x)

    def energy(
        self,
        y_t: Array,
        x_t: Array,
        graph: Any,
        params: Dict[str, Array],
    ) -> float:
        """
        Gaussian negative log-likelihood (up to constant):
        E = ||y - μ(x)||² / (2σ²)
        """
        noise_scale = params.get('observation_noise', 1.0)

        # Predicted observation
        y_pred = self.observation_fn(x_t, graph)

        # Squared error
        residual = y_t - y_pred
        squared_error = jnp.sum(residual ** 2)

        # Energy (negative log-likelihood)
        energy = squared_error / (2 * noise_scale ** 2)

        return energy

    def grad_x(
        self,
        y_t: Array,
        x_t: Array,
        graph: Any,
        params: Dict[str, Array],
    ) -> Array:
        """
        Gradient of Gaussian energy w.r.t. latent state.

        For direct observation (y = x + noise):
            ∇_x E = -(y - x) / σ²

        For general observation function y = f(x) + noise:
            ∇_x E = -∇_x f(x)ᵀ (y - f(x)) / σ²
        """
        # Use JAX automatic differentiation
        grad_fn = grad(lambda x: self.energy(y_t, x, graph, params))
        return grad_fn(x_t)

    def sample_observation(
        self,
        x_t: Array,
        graph: Any,
        params: Dict[str, Array],
        key: Array,
    ) -> Array:
        """
        Sample from Gaussian observation model: y ~ N(f(x), σ²I)
        """
        noise_scale = params.get('observation_noise', 1.0)

        y_mean = self.observation_fn(x_t, graph)
        noise = random.normal(key, shape=y_mean.shape) * noise_scale

        return y_mean + noise


class StandardRiskEvaluator(RiskEvaluatorPlugin):
    """
    Standard statistical evaluation metrics.

    Computes:
    - CRPS (Continuous Ranked Probability Score)
    - Coverage (for prediction intervals)
    - Mean Absolute Error
    - Root Mean Squared Error
    """

    def evaluate(
        self,
        predictions: Dict[str, Array],
        actuals: Array,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute standard metrics.

        Args:
            predictions: Must contain 'mean' and optionally 'samples' or 'quantiles'
            actuals: Ground truth (shape: [T, D])
            metadata: Not used in standard evaluator

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        pred_mean = predictions['mean']

        # Point prediction metrics
        errors = actuals - pred_mean
        metrics['mae'] = float(jnp.mean(jnp.abs(errors)))
        metrics['rmse'] = float(jnp.sqrt(jnp.mean(errors ** 2)))

        # CRPS (if samples available)
        if 'samples' in predictions:
            samples = predictions['samples']  # Shape: [num_samples, T, D]
            crps = self._compute_crps(samples, actuals)
            metrics['crps'] = float(crps)

        # Coverage (if quantiles available)
        if 'quantiles' in predictions:
            quantiles = predictions['quantiles']  # Dict: {0.05: array, 0.95: array}

            for level in [0.90, 0.95, 0.99]:
                lower_q = (1 - level) / 2
                upper_q = 1 - lower_q

                if lower_q in quantiles and upper_q in quantiles:
                    lower = quantiles[lower_q]
                    upper = quantiles[upper_q]

                    coverage = jnp.mean(
                        (actuals >= lower) & (actuals <= upper)
                    )
                    metrics[f'coverage_{int(level*100)}'] = float(coverage)

        return metrics

    def _compute_crps(self, samples: Array, actuals: Array) -> float:
        """
        Compute empirical CRPS from posterior samples.

        CRPS = E[|Y - X|] - 0.5 * E[|X - X'|]
        where Y ~ actual, X, X' ~ forecast (independent)
        """
        num_samples = samples.shape[0]

        # E[|Y - X|]
        term1 = jnp.mean(jnp.abs(samples - actuals[None, :, :]))

        # E[|X - X'|] (pairwise differences)
        # For efficiency, subsample if too many samples
        if num_samples > 100:
            idx = np.random.choice(num_samples, 100, replace=False)
            samples_sub = samples[idx]
        else:
            samples_sub = samples

        # Pairwise absolute differences
        term2 = 0.0
        n = samples_sub.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                term2 += jnp.mean(jnp.abs(samples_sub[i] - samples_sub[j]))

        term2 = term2 * 2 / (n * (n - 1))  # Average over all pairs

        crps = term1 - 0.5 * term2
        return crps


# Auto-register public implementations
register_plugin('data_stream', 'public_csv', PublicCSVDataStream)
register_plugin('obs_energy', 'gaussian', GaussianObservationEnergy)
register_plugin('risk_evaluator', 'standard', StandardRiskEvaluator)
