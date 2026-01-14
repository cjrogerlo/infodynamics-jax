"""
Synthetic functions for generating test data.
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Dict


class SyntheticFunctions:
    """Collection of synthetic functions for testing."""
    
    def __init__(self):
        self._functions = {
            # --- SMOOTH / MULTISCALE ---
            'sine_mix': (
                lambda x: jnp.sin(3 * x) + 0.5 * jnp.cos(5 * x),
                'Sine Mix',
                'sin(3x) + 0.5*cos(5x)',
                'smooth_multiscale'
            ),
            'nonstationary_frequency': (
                lambda x: jnp.sin(2 * x) + 0.5 * jnp.sin(20 * x) * jnp.exp(-5 * x**2),
                'Nonstationary Frequency',
                'sin(2x) + 0.5*sin(20x)*exp(-5x²)',
                'smooth_multiscale'
            ),
            'oscillatory_pocket': (
                lambda x: 0.5 * x + jnp.exp(-10 * (x - 0.5)**2) * jnp.cos(30 * x),
                'Oscillatory Pocket',
                '0.5x + exp(-10(x-0.5)²)*cos(30x)',
                'smooth_multiscale'
            ),
            'multimodal_sines': (
                lambda x: 0.5 * jnp.sin(4 * x) + 0.3 * jnp.sin(13 * x) + 0.1 * jnp.sin(40 * x),
                'Multimodal Sines',
                '0.5*sin(4x) + 0.3*sin(13x) + 0.1*sin(40x)',
                'smooth_multiscale'
            ),
            'highfreq_composite': (
                lambda x: jnp.sin(4 * x) + 0.5 * jnp.cos(15 * x) + 0.3 * jnp.sin(40 * x),
                'High-Freq Composite',
                'sin(4x) + 0.5*cos(15x) + 0.3*sin(40x)',
                'smooth_multiscale'
            ),
            'fractal_mcos': (
                lambda x: sum(0.6**k * jnp.cos(6**k * x) for k in range(1, 6)),
                'Fractal Cosine Mixture',
                'sum(0.6^k * cos(6^k * x))',
                'smooth_multiscale'
            ),
            'multiscale_cos': (
                lambda x: jnp.cos(x) + 0.5 * jnp.cos(3 * x) + 0.25 * jnp.cos(9 * x) + 0.1 * jnp.cos(27 * x),
                'Multi-scale Cosine',
                'cos(x) + 0.5*cos(3x) + 0.25*cos(9x) + 0.1*cos(27x)',
                'smooth_multiscale'
            ),

            # --- PIECEWISE CONTINUOUS ---
            'piecewise_kink': (
                lambda x: jnp.where(x < 0, -0.5 * x + 0.2, 0.8 * x - 0.4),
                'Piecewise Kink',
                'L: -0.5x+0.2, R: 0.8x-0.4',
                'piecewise_continuous'
            ),
            'abs_kink': (
                lambda x: jnp.abs(x) + 0.3 * jnp.sin(4 * x),
                'Absolute Kink',
                '|x| + 0.3*sin(4x)',
                'piecewise_continuous'
            ),
            'piecewise_three_region': (
                lambda x: jnp.where(
                    x < -0.3,
                    jnp.sin(4 * x),
                    jnp.where(
                        x < 0.4,
                        0.5 * x + 0.3,
                        jnp.cos(6 * x) + 1.0
                    )
                ),
                'Three-region Piecewise',
                'sin -> linear -> cos',
                'piecewise_continuous'
            ),
            'heaviside_wiggle': (
                lambda x: jnp.where(
                    x < 0,
                    0.2 * jnp.sin(10 * x),
                    1.0 + 0.2 * jnp.sin(10 * x)
                ),
                'Heaviside Wiggle',
                'Step + sin(10x)',
                'piecewise_continuous'
            ),

            # --- DISCONTINUOUS ---
            'step_local_variation': (
                lambda x: jnp.where(x < 0, jnp.sin(3 * x), jnp.sin(3 * x) + 2.0),
                'Step Local Variation',
                'sin(3x) (+2 if x>0)',
                'discontinuous'
            ),
            'multi_discontinuity': (
                lambda x: jnp.where(
                    x < -1,
                    jnp.sin(2 * x),
                    jnp.where(
                        x < 0,
                        1.5,
                        jnp.where(
                            x < 1,
                            jnp.cos(5 * x),
                            -2.0
                        )
                    )
                ),
                'Multiple Discontinuities',
                'Four distinct regimes',
                'discontinuous'
            ),
            'square_wave': (
                lambda x: jnp.sign(jnp.sin(5 * x)),
                'Square Wave',
                'sign(sin(5x))',
                'discontinuous'
            ),
            'sawtooth': (
                lambda x: x - jnp.floor(x),
                'Sawtooth Wave',
                'x - floor(x)',
                'discontinuous'
            ),
            'wiggly_discontinuous': (
                lambda x: jnp.where(
                    x < 0,
                    jnp.sin(8 * x),
                    jnp.sin(8 * x) + 0.7 * jnp.sign(jnp.sin(3 * x))
                ),
                'Wiggly Discontinuous',
                'sin(8x) + jumpy sign',
                'discontinuous'
            ),

            # --- SPIKY LOCALISED ---
            'local_bump': (
                lambda x: jnp.exp(-40 * (x - 0.3)**2) + 0.1 * jnp.sin(10 * x),
                'Local Bump',
                'exp(-40(x-0.3)²) + small sin',
                'spiky_localised'
            ),
            'two_bumps': (
                lambda x: jnp.exp(-30 * (x + 0.5)**2) + 0.8 * jnp.exp(-50 * (x - 0.4)**2),
                'Two Bumps',
                'Two separated Gaussians',
                'spiky_localised'
            ),
            'spike_train': (
                lambda x: jnp.exp(-200 * (x + 0.7)**2) + jnp.exp(-200 * (x + 0.2)**2) + \
                          jnp.exp(-200 * (x - 0.4)**2) + jnp.exp(-200 * (x - 0.8)**2),
                'Spike Train',
                'Multiple sharp impulses',
                'spiky_localised'
            ),

            # --- STRUCTURAL ---
            'rational_peak': (
                lambda x: 1 / (0.1 + x**2),
                'Rational Peak',
                '1 / (0.1 + x²)',
                'structural'
            ),
            'log_singularity': (
                lambda x: jnp.log(jnp.abs(x) + 0.05),
                'Log Singularity',
                'log(|x| + 0.05)',
                'structural'
            ),

            # --- LEGACY ---
            'sine': (lambda x: jnp.sin(2 * jnp.pi * x), 'Sine', 'sin(2πx)', 'periodic'),
            'cosine': (lambda x: jnp.cos(2 * jnp.pi * x), 'Cosine', 'cos(2πx)', 'periodic'),
            'linear': (lambda x: x, 'Linear', 'x', 'linear'),
            'quadratic': (lambda x: x ** 2, 'Quadratic', 'x²', 'polynomial'),
            'gaussian': (lambda x: jnp.exp(-0.5 * (x / 0.5)**2), 'Gaussian', 'exp(-0.5*(x/0.5)²)', 'smooth'),
        }
    
    def get(self, name: str) -> Tuple[Callable, str, str, str]:
        """Get a function by name."""
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
        """Sample data from a synthetic function."""
        fn, _, _, _ = self.get(name)
        x_min, x_max = domain
        
        key_x, key_noise = jax.random.split(key)
        X = jax.random.uniform(key_x, (N,), minval=x_min, maxval=x_max)
        X = jnp.sort(X)
        
        Y_clean = fn(X)
        noise_samples = jax.random.normal(key_noise, (N,)) * noise
        Y = Y_clean + noise_samples
        
        return X, Y, Y_clean


# Global instance
synthetic = SyntheticFunctions()
