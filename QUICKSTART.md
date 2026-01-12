# Quick Start Guide for infodynamics-jax

Get started with `infodynamics-jax` in 5 minutes!

## Installation

### Step 1: Clone or Navigate to the Repository

```bash
cd /path/to/infodynamics-jax
```

### Step 2: Install Dependencies

**Option A: Using pip (recommended)**
```bash
pip install -e .
```

This installs the package in "editable" mode, so changes to the source code are immediately reflected.

**Option B: Manual installation**
```bash
pip install jax jaxlib optax numpy scipy matplotlib jupyter
```

## Running Examples

### Method 1: Using Jupyter Notebooks (Recommended)

```bash
# Install Jupyter if not already installed
pip install jupyter matplotlib

# Start Jupyter from the examples directory
cd examples
jupyter notebook

# Open any of the notebooks:
# - notebook_01_basic_regression.ipynb
# - notebook_02_different_kernels.ipynb
# - notebook_03_classification.ipynb
# - notebook_04_annealed_smc.ipynb
# - notebook_05_online_ibis.ipynb
```

### Method 2: Running Python Scripts

```bash
# From project root
python examples/01_basic_regression_map2.py
```

### Method 3: Running Tests

```bash
# Run basic pipeline tests
python test_pipeline_basic.py
python test_pipeline_map2.py
python test_pipeline_annealed_smc.py
```

## Your First Inference

Here's a minimal example to get you started:

```python
import jax
import jax.numpy as jnp
from infodynamics_jax.core import Phi
from infodynamics_jax.gp.kernels.params import KernelParams
from infodynamics_jax.gp.kernels.rbf import rbf as rbf_kernel
from infodynamics_jax.gp.likelihoods import get as get_likelihood
from infodynamics_jax.energy import InertialEnergy, InertialCFG
from infodynamics_jax.inference.optimisation import MAP2, MAP2CFG
from infodynamics_jax.infodynamics import run, RunCFG

# Generate toy data
key = jax.random.key(0)
X = jnp.linspace(-3, 3, 30)[:, None]
Y = jnp.sin(X[:, 0]) + 0.1 * jax.random.normal(key, (30,))

# Initialize model
phi_init = Phi(
    kernel_params=KernelParams(lengthscale=jnp.array(1.0), variance=jnp.array(1.0)),
    Z=jnp.linspace(-3, 3, 10)[:, None],  # 10 inducing points
    likelihood_params={"noise_var": jnp.array(0.1)},
    jitter=1e-5,
)

# Create energy
energy = InertialEnergy(
    kernel_fn=rbf_kernel,
    likelihood=get_likelihood("gaussian"),
    cfg=InertialCFG(estimator="gh", gh_n=20),
)

# Run MAP-II optimization
method = MAP2(cfg=MAP2CFG(steps=100, lr=1e-2))
out = run(key=key, method=method, energy=energy, phi_init=phi_init, energy_args=(X, Y))

# Get optimized parameters
phi_opt = out.result.phi
print(f"Optimized lengthscale: {phi_opt.kernel_params.lengthscale:.3f}")
print(f"Optimized variance: {phi_opt.kernel_params.variance:.3f}")
```

## Common Issues and Solutions

### Issue 1: `ModuleNotFoundError: No module named 'infodynamics_jax'`

**Solution**: Make sure you've installed the package or set the Python path:

```bash
# Option 1: Install in editable mode
pip install -e .

# Option 2: Set PYTHONPATH
export PYTHONPATH=$PWD
```

For Jupyter notebooks, add this at the top:
```python
import sys
sys.path.insert(0, '..')
```

### Issue 2: JAX Platform Warnings

**Solution**: Force CPU mode:

```python
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
jax.config.update('jax_platform_name', 'cpu')
```

### Issue 3: NaN Values in Optimization

**Solutions**:
1. Increase `jitter` in Phi: `jitter=1e-4`
2. Reduce learning rate: `MAP2CFG(..., lr=1e-3)`
3. Enable gradient clipping: `MAP2CFG(..., clip_grad_norm=1.0)`
4. Normalize your data to [-1, 1] or [0, 1]

### Issue 4: Slow Performance

**Solutions**:
1. Enable JIT: `cfg=RunCFG(jit=True)`
2. Reduce inducing points: Use M = 10-20 for testing
3. Use fewer optimization steps initially
4. For SMC, reduce particle count: `n_particles=32`

## Next Steps

1. **📚 Read the Examples**: Work through `examples/notebook_01_basic_regression.ipynb`
2. **📖 Check Documentation**: See `docs/design/` for architecture details
3. **🔬 Try Different Kernels**: Explore `notebook_02_different_kernels.ipynb`
4. **🎯 Understand Non-Conjugate**: Read `notebook_03_classification.ipynb`
5. **🎲 Learn SMC**: Study `notebook_04_annealed_smc.ipynb`

## Getting Help

- **Examples**: `examples/README.md`
- **Design Docs**: `docs/design/`
- **Theory**: `docs/theory/`
- **Contributing**: `docs/contributing_energy.md`

## Key Concepts

### The Three Pillars

1. **Energy** (`energy/`): What to optimize
   - `InertialEnergy`: Data-dependent term
   - `PriorEnergy`: Regularization
   - `TargetEnergy`: Combined objective

2. **Inference** (`inference/`): How to optimize
   - `MAP2`: Point estimate (fast)
   - `AnnealedSMC`: Full posterior (robust)
   - `HMC/NUTS/MALA`: MCMC samplers

3. **Infodynamics** (`infodynamics/`): Orchestration
   - `run()`: Unified interface
   - Handles hyperpriors automatically
   - Provides diagnostics

### Workflow

```
Data → Phi (model) → Energy (objective) → Inference (optimization) → Results
```

## Quick Reference

### Common Configurations

**Fast prototyping (MAP-II)**:
```python
MAP2CFG(steps=100, lr=1e-2, jit=False)
```

**Production (MAP-II)**:
```python
MAP2CFG(steps=500, lr=5e-3, jit=True, constrain_params=True)
```

**Uncertainty quantification (SMC)**:
```python
AnnealedSMCCFG(n_particles=64, n_steps=20, rejuvenation="hmc", jit=True)
```

### Common Kernel Configs

**RBF (smooth functions)**:
```python
KernelParams(lengthscale=1.0, variance=1.0)
```

**Matérn 3/2 (less smooth)**:
```python
KernelParams(lengthscale=1.0, variance=1.0)
```

**Periodic (seasonal data)**:
```python
KernelParams(lengthscale=1.0, variance=1.0, period=2.0)
```

Happy inferencing! 🎉
