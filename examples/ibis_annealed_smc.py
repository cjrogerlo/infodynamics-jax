# examples/ibis_annealed_smc.py
"""
IBIS (Iterated Batch Importance Sampling) with Annealed SMC.

This script demonstrates online inference using:
  - SupervisedData.prefix() for sequential data views
  - AnnealedSMC for posterior approximation at each time step
  - runner.run() for clean orchestration

IBIS processes data sequentially:
  t=1: p(phi | y_1)
  t=2: p(phi | y_1, y_2)
  ...
  t=T: p(phi | y_1, ..., y_T)

At each step, we use Annealed SMC to approximate the posterior.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import List, Dict, Any

from infodynamics_jax.core import SupervisedData
from infodynamics_jax.core.phi import Phi
from infodynamics_jax.infodynamics import RunOut
from infodynamics_jax.inference.particle import AnnealedSMC, AnnealedSMCCFG
from infodynamics_jax.energy import TargetEnergy, InertialEnergy, PriorEnergy


def init_particles_fn(key: jax.random.KeyArray, n_particles: int, phi_template: Phi) -> Any:
    """
    Initialize particles for SMC.
    
    Args:
        key: PRNG key
        n_particles: Number of particles
        phi_template: Template Phi to determine structure
    
    Returns:
        Stacked particles pytree with shape [n_particles, ...]
    """
    keys = jax.random.split(key, n_particles)
    
    def init_one(key_i):
        # Initialize each particle from prior or around phi_template
        # This is a simple example - you may want to add noise or sample from prior
        # For now, just return the template (all particles start the same)
        return phi_template
    
    # Use vmap to stack particles
    # Note: vmap expects the function to work on a single key, then stacks results
    particles = jax.vmap(init_one)(keys)
    return particles


def ibis_annealed_smc(
    data: SupervisedData,
    energy: Any,  # EnergyTerm (using Any to avoid import issues)
    phi_init: Phi,
    *,
    key: jax.random.KeyArray,
    n_particles: int = 128,
    n_steps_per_time: int = 20,
    ess_threshold: float = 0.5,
    min_time_steps: int = 10,  # Start IBIS after this many samples
    max_time_steps: int | None = None,  # None = use all data
) -> Dict[str, Any]:
    """
    Run IBIS with Annealed SMC.
    
    Args:
        data: SupervisedData with (X, Y)
        energy: TargetEnergy (inertial + prior)
        phi_init: Initial parameter state
        key: PRNG key
        n_particles: Number of SMC particles
        n_steps_per_time: Annealing steps per time point
        ess_threshold: ESS threshold for resampling
        min_time_steps: Minimum number of samples before starting IBIS
        max_time_steps: Maximum number of time steps (None = all)
    
    Returns:
        Dictionary with:
        - particles_trace: List of particle sets at each time
        - logw_trace: List of log weights at each time
        - ess_trace: List of ESS values at each time
        - logZ_trace: List of log normalizing constant estimates
        - diagnostics: Per-time diagnostics
    """
    max_time_steps = max_time_steps or len(data)
    max_time_steps = min(max_time_steps, len(data))
    
    # Initialize storage
    particles_trace: List[Any] = []
    logw_trace: List[jnp.ndarray] = []
    ess_trace: List[jnp.ndarray] = []
    logZ_trace: List[float] = []
    diagnostics_list: List[Dict[str, Any]] = []
    
    # Initialize particles from prior (or around phi_init)
    # For first time step, initialize from phi_init
    key, subkey = jax.random.split(key)
    particles = init_particles_fn(subkey, n_particles, phi_init)
    logw = jnp.zeros(n_particles)
    
    # IBIS loop: process data sequentially
    for t in range(min_time_steps, max_time_steps + 1):
        print(f"IBIS step {t}/{max_time_steps} (using {t} samples)")
        
        # Get prefix data (first t samples)
        data_prefix = data.prefix(t)
        
        # Configure Annealed SMC for this time step
        smc_cfg = AnnealedSMCCFG(
            n_particles=n_particles,
            n_steps=n_steps_per_time,
            ess_threshold=ess_threshold,
            rejuvenation="hmc",
            rejuvenation_steps=1,
            jit=True,
        )
        method = AnnealedSMC(cfg=smc_cfg)
        
        # Create init_particles function that uses current particles
        # For IBIS, we want to continue from previous particles (possibly resampled)
        def init_from_current(key_init, n_p):
            # Use current particles - ensure they have the right batch dimension
            # particles should already be a stacked pytree [n_particles, ...]
            # Check if particles is already batched
            from jax.tree_util import tree_flatten
            particles_flat, _ = tree_flatten(particles)
            if len(particles_flat) > 0 and particles_flat[0].shape[0] == n_p:
                # Already has correct batch dimension
                return particles
            else:
                # Need to stack - this shouldn't happen in normal IBIS flow
                # But handle it just in case
                from jax.tree_util import tree_map
                return tree_map(lambda x: jnp.stack([x] * n_p), particles)
        
        # Run Annealed SMC with prefix data
        # Note: AnnealedSMC.run() requires init_particles_fn, not phi_init
        key, subkey = jax.random.split(key)
        
        # We need to call method.run() directly since runner expects phi_init
        # but AnnealedSMC expects init_particles_fn
        # For now, we'll call it directly and wrap the result
        smc_result = method.run(
            energy=energy,
            init_particles_fn=init_from_current,
            key=subkey,
            energy_args=(data_prefix.X, data_prefix.Y),
        )
        
        # Extract results directly from smc_result
        particles = smc_result.particles
        logw = smc_result.logw
        
        # Store results
        particles_trace.append(particles)
        logw_trace.append(logw)
        ess_trace.append(smc_result.ess_trace)
        logZ_trace.append(smc_result.logZ_est)
        
        # Create diagnostics
        diagnostics = {
            "method": "AnnealedSMC",
            "ess_final": float(smc_result.ess_trace[-1]) if len(smc_result.ess_trace) > 0 else 0.0,
            "logZ_est": float(smc_result.logZ_est),
        }
        diagnostics_list.append(diagnostics)
        
        # Print diagnostics
        ess_final = smc_result.ess_trace[-1] if len(smc_result.ess_trace) > 0 else 0.0
        print(f"  ESS: {ess_final:.2f}, logZ: {smc_result.logZ_est:.2f}")
    
    return {
        "particles_trace": particles_trace,
        "logw_trace": logw_trace,
        "ess_trace": ess_trace,
        "logZ_trace": logZ_trace,
        "diagnostics": diagnostics_list,
    }


def main():
    """
    Example usage of IBIS with Annealed SMC.
    
    Minimal working example with synthetic data.
    """
    key = jax.random.key(42)
    
    # Generate synthetic data
    N, Q, D = 50, 2, 1  # Smaller for testing
    key, subkey = jax.random.split(key)
    X = jax.random.normal(subkey, (N, Q))
    key, subkey = jax.random.split(key)
    Y = jax.random.normal(subkey, (N,)).squeeze()
    data = SupervisedData(X, Y)
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"Running IBIS with Annealed SMC...")
    print(f"Will process {len(data)} samples sequentially")
    
    # Create minimal Phi
    from infodynamics_jax.gp.kernels.params import KernelParams
    # Note: kernels are in gp/kernels, but the get function might be in a different location
    # For now, we'll create a simple kernel function directly
    from infodynamics_jax.gp.kernels.rbf import rbf as rbf_kernel
    
    # Simple kernel function wrapper (rbf signature: rbf(X, Z, params))
    def kernel_fn(X1, X2, params):
        return rbf_kernel(X1, X2, params)
    
    kernel_params = KernelParams(
        lengthscale=jnp.array(1.0),
        variance=jnp.array(1.0),
    )
    
    # Simple inducing points (use subset of X)
    M = min(10, N)
    Z = X[:M].copy()
    
    phi_init = Phi(
        kernel_params=kernel_params,
        Z=Z,
        likelihood_params={"noise_var": jnp.array(0.1)},
        jitter=1e-8,
    )
    
    # Create minimal energy
    # For now, we'll use a simple energy that just returns a constant
    # In a real scenario, you'd use InertialEnergy + PriorEnergy
    from infodynamics_jax.energy.base import EnergyTerm
    
    class SimpleEnergy(EnergyTerm):
        """Minimal energy for testing."""
        def __call__(self, phi, X, Y, key=None):
            # Simple energy: just return a constant for testing
            # In real use, this would be InertialEnergy + PriorEnergy
            return jnp.array(0.0)
    
    energy = SimpleEnergy()
    
    print(f"\nInitializing with {M} inducing points")
    print(f"Starting IBIS...")
    
    # Run IBIS with minimal settings for testing
    try:
        results = ibis_annealed_smc(
            data=data,
            energy=energy,
            phi_init=phi_init,
            key=key,
            n_particles=16,  # Small for testing
            n_steps_per_time=5,  # Small for testing
            min_time_steps=5,
            max_time_steps=15,  # Just test first 15 steps
        )
        
        # Analyze results
        print(f"\nIBIS completed!")
        print(f"Processed {len(results['particles_trace'])} time steps")
        if len(results['logZ_trace']) > 0:
            print(f"Final logZ estimate: {results['logZ_trace'][-1]:.2f}")
        print(f"ESS trace lengths: {[len(e) for e in results['ess_trace']]}")
        
    except Exception as e:
        print(f"Error during IBIS: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
