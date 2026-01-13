"""
Synthetic functions for generating test data.
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Callable


class SyntheticFunctions:
    """Collection of synthetic functions for testing."""
    
    def __init__(self):
        self._functions = {
            'sine_mix': (
                lambda x: jnp.sin(2 * x) + 0.3 * jnp.cos(5 * x),
                'Sine Mix',
                'sin(2x) + 0.3*cos(5x)',
                'periodic'
            ),
            'sine': (
                lambda x: jnp.sin(2 * jnp.pi * x),
                'Sine',
                'sin(2πx)',
                'periodic'
            ),
            'cosine': (
                lambda x: jnp.cos(2 * jnp.pi * x),
                'Cosine',
                'cos(2πx)',
                'periodic'
            ),
            'linear': (
                lambda x: x,
                'Linear',
                'x',
                'linear'
            ),
            'quadratic': (
                lambda x: x ** 2,
                'Quadratic',
                'x²',
                'polynomial'
            ),
            'gaussian': (
                lambda x: jnp.exp(-0.5 * (x / 0.5) ** 2),
                'Gaussian',
                'exp(-0.5*(x/0.5)²)',
                'smooth'
            ),
        }
    
    def get(self, name: str) -> Tuple[Callable, str, str, str]:
        """Get a function by name.
        
        Returns:
            (function, title, description, category)
        """
        if name not in self._functions:
            raise ValueError(f"Unknown function: {name}. Available: {list(self._functions.keys())}")
        return self._functions[name]
    
    def sample(
        self,
        name: str,
        N: int,
        noise: float,
        domain: Tuple[float, float],
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample data from a synthetic function.
        
        Args:
            name: Function name
            N: Number of samples
            noise: Noise standard deviation
            domain: (x_min, x_max)
            key: JAX random key
            
        Returns:
            (X, Y, Y_clean) where Y = Y_clean + noise
        """
        fn, _, _, _ = self.get(name)
        x_min, x_max = domain
        
        # Generate X uniformly in domain
        key_x, key_noise = jax.random.split(key)
        X = jax.random.uniform(key_x, (N,), minval=x_min, maxval=x_max)
        X = jnp.sort(X)  # Sort for easier visualization
        
        # Generate clean Y
        Y_clean = fn(X)
        
        # Add noise
        noise_samples = jax.random.normal(key_noise, (N,)) * noise
        Y = Y_clean + noise_samples
        
        return X, Y, Y_clean


# Global instance
synthetic = SyntheticFunctions()
