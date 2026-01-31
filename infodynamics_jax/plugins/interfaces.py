"""
Abstract base classes for infodynamics-jax plugins.

These interfaces define the contracts that plugin implementations must satisfy.
They are designed to be:
1. Minimal - only essential methods required by core algorithms
2. Composable - plugins can be mixed and matched
3. JAX-friendly - compatible with JIT compilation and automatic differentiation
4. Type-safe - use type hints for better IDE support and static checking

All proprietary implementations should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Callable
import jax.numpy as jnp
from jax import Array


class DataStreamPlugin(ABC):
    """
    Abstract interface for data streaming.

    This allows private projects to implement custom data loaders
    (e.g., LOB data, proprietary market data, medical imaging streams)
    while the core library only depends on this interface.

    Example implementations:
    - PublicCSVStream (in core library, for examples)
    - ProprietaryLOBStream (in private hedge fund repo)
    - MedicalImageStream (in private medical imaging repo)
    """

    @abstractmethod
    def next(self) -> Tuple[float, Array, Optional[Dict[str, Any]]]:
        """
        Get next observation from the stream.

        Returns:
            t: Timestamp (float)
            y_t: Observation at time t (shape: [D_obs])
            meta: Optional metadata (e.g., market conditions, image tags)

        Raises:
            StopIteration: When stream is exhausted
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset stream to beginning."""
        pass

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Dimensionality of observations."""
        pass

    @property
    def metadata_schema(self) -> Optional[Dict[str, type]]:
        """
        Optional: Define metadata schema for validation.

        Example:
            {'market_regime': str, 'volatility': float}
        """
        return None


class ObservationEnergyPlugin(ABC):
    """
    Abstract interface for observation energy (likelihood) models.

    This is where proprietary modeling happens:
    - Custom noise models (Student-t for heavy tails, etc.)
    - Structured likelihoods (order book dynamics, option surfaces)
    - Domain-specific observation models

    The core library provides standard implementations (Gaussian, etc.),
    but private projects can inject sophisticated models here.
    """

    @abstractmethod
    def energy(
        self,
        y_t: Array,
        x_t: Array,
        graph: Any,  # GraphState from core library
        params: Dict[str, Array],
    ) -> float:
        """
        Compute observation energy E_obs(y_t | x_t, G).

        Lower energy = higher likelihood.

        Args:
            y_t: Observation at time t (shape: [D_obs])
            x_t: Latent state at time t (shape: [Q])
            graph: Graph structure (node positions, edges)
            params: Model parameters (e.g., noise_scale, degrees_of_freedom)

        Returns:
            Scalar energy value (float)
        """
        pass

    @abstractmethod
    def grad_x(
        self,
        y_t: Array,
        x_t: Array,
        graph: Any,
        params: Dict[str, Array],
    ) -> Array:
        """
        Gradient of energy w.r.t. latent state: ∇_x E_obs.

        Used by oEGPF during particle updates.

        Returns:
            Gradient array (shape: [Q])
        """
        pass

    def sample_observation(
        self,
        x_t: Array,
        graph: Any,
        params: Dict[str, Array],
        key: Array,
    ) -> Array:
        """
        Optional: Sample observation given latent state.

        Useful for:
        - Posterior predictive checks
        - Simulating forward scenarios
        - Testing

        Args:
            key: JAX random key

        Returns:
            Sampled observation (shape: [D_obs])
        """
        raise NotImplementedError("Sampling not implemented for this energy model")


class DynamicsEnergyPlugin(ABC):
    """
    Abstract interface for dynamics energy (state transition) models.

    Examples:
    - Gaussian process dynamics (already in core)
    - Regime-switching models (proprietary)
    - Jump diffusions (proprietary)
    """

    @abstractmethod
    def energy(
        self,
        x_t: Array,
        x_prev: Array,
        graph: Any,
        params: Dict[str, Array],
    ) -> float:
        """
        Compute dynamics energy E_dyn(x_t | x_{t-1}, G).

        Args:
            x_t: Current state (shape: [Q])
            x_prev: Previous state (shape: [Q])
            graph: Graph structure
            params: Dynamics parameters (e.g., GP kernel hyperparameters)

        Returns:
            Scalar energy value
        """
        pass

    @abstractmethod
    def grad_x_t(
        self,
        x_t: Array,
        x_prev: Array,
        graph: Any,
        params: Dict[str, Array],
    ) -> Array:
        """
        Gradient w.r.t. current state: ∇_{x_t} E_dyn.

        Returns:
            Gradient (shape: [Q])
        """
        pass


class RiskEvaluatorPlugin(ABC):
    """
    Abstract interface for risk evaluation and model assessment.

    Public library provides standard metrics (CRPS, coverage, ESS).
    Private projects can add domain-specific evaluators:
    - Trading P&L simulation
    - Portfolio risk metrics (VaR, CVaR, tail risk)
    - Medical diagnosis accuracy
    """

    @abstractmethod
    def evaluate(
        self,
        predictions: Dict[str, Array],
        actuals: Array,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate prediction quality.

        Args:
            predictions: Dictionary containing:
                - 'mean': Point predictions
                - 'samples': Posterior samples (optional)
                - 'quantiles': Quantile predictions (optional)
            actuals: Ground truth observations
            metadata: Optional context (e.g., market regime for conditional metrics)

        Returns:
            Dictionary of metrics (e.g., {'crps': 1.23, 'coverage_95': 0.94})
        """
        pass


class ConfigPlugin(ABC):
    """
    Abstract interface for configuration management.

    Allows private projects to extend configuration schema
    while maintaining compatibility with public core.
    """

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ConfigPlugin':
        """Load configuration from dictionary."""
        pass


# Type aliases for plugin factories
DataStreamFactory = Callable[..., DataStreamPlugin]
ObservationEnergyFactory = Callable[..., ObservationEnergyPlugin]
DynamicsEnergyFactory = Callable[..., DynamicsEnergyPlugin]
RiskEvaluatorFactory = Callable[..., RiskEvaluatorPlugin]
