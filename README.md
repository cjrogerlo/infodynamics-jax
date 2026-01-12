# infodynamics-jax

**Inference as infodynamics** — A general-purpose library for Bayesian inference over energy landscapes.

## Overview

`infodynamics-jax` is a **general-purpose inference library** designed to support a wide range of applications:

- **Supervised learning**: GP regression, classification, etc.
- **Unsupervised learning**: GPLVM, latent GP models, etc.
- **Online inference**: IBIS, streaming data, etc.
- **Future applications**: RL, EGPF, RJVMC, etc.

The library provides a clean separation between:
- **Energy landscapes** (`energy/`): Define the inference objective
- **Inference dynamics** (`inference/`): Algorithms (MCMC, SMC, optimization)
- **Orchestration** (`infodynamics/`): Execution layer that composes energy + inference

## Design Principles

1. **Protocol-based**: Uses `EnergyTerm` and `InferenceMethod` protocols for maximum flexibility
2. **Application-agnostic**: No hardcoded assumptions about supervised/unsupervised/online scenarios
3. **Composable**: Energy functions, inference methods, and data views can be freely combined
4. **JAX-native**: Fully JIT-compatible, supports automatic differentiation

## Quick Start

**New to the library?** See [QUICKSTART.md](QUICKSTART.md) for a 5-minute introduction.

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

# Generate data
key = jax.random.key(0)
X = jnp.linspace(-3, 3, 30)[:, None]
Y = jnp.sin(X[:, 0]) + 0.1 * jax.random.normal(key, (30,))

# Initialize model
phi_init = Phi(
    kernel_params=KernelParams(lengthscale=jnp.array(1.0), variance=jnp.array(1.0)),
    Z=jnp.linspace(-3, 3, 10)[:, None],
    likelihood_params={"noise_var": jnp.array(0.1)},
    jitter=1e-5,
)

# Create energy and run MAP-II
energy = InertialEnergy(
    kernel_fn=rbf_kernel,
    likelihood=get_likelihood("gaussian"),
    cfg=InertialCFG(estimator="gh", gh_n=20),
)

method = MAP2(cfg=MAP2CFG(steps=100, lr=1e-2))
out = run(key=key, method=method, energy=energy, phi_init=phi_init, energy_args=(X, Y))
phi_opt = out.result.phi  # Optimized hyperparameters
```

## Architecture

```
infodynamics_jax/
├── core/          # Core data structures (Phi, SupervisedData, LatentData)
├── energy/        # Energy functionals (EnergyTerm protocol)
├── gp/            # GP components (kernels, likelihoods, ansatz)
├── inference/     # Inference dynamics (InferenceMethod protocol)
│   ├── optimisation/  # VGA, MAP2
│   ├── particle/      # AnnealedSMC, IBIS
│   └── sampling/      # HMC, NUTS, MALA, Slice
└── infodynamics/  # Orchestration layer (runner.run)
```

## Key Features

- **Multiple inference methods**: HMC, NUTS, MALA, Slice, Annealed SMC, IBIS, VGA, MAP2
- **Flexible energy composition**: InertialEnergy, PriorEnergy, TargetEnergy, etc.
- **Non-conjugate support**: Gauss-Hermite and Monte Carlo estimators
- **Hyperprior utilities**: L2, log-L2 priors on kernel/likelihood parameters
- **Data views**: SupervisedData, LatentData with batch/prefix operations

## Examples

Check out our comprehensive example notebooks in the `examples/` directory:

1. **[Basic GP Regression](examples/notebook_01_basic_regression.ipynb)** - Introduction to MAP-II optimization
2. **[Different Kernels](examples/notebook_02_different_kernels.ipynb)** - Comparing RBF, Matérn, Periodic kernels
3. **[GP Classification](examples/notebook_03_classification.ipynb)** - Non-conjugate likelihoods
4. **[Annealed SMC](examples/notebook_04_annealed_smc.ipynb)** - Full Bayesian inference
5. **[Online IBIS](examples/notebook_05_online_ibis.ipynb)** - Streaming data processing

See [examples/README.md](examples/README.md) for setup instructions.

## Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Examples README](examples/README.md)** - How to run examples
- [Energy Design](docs/energy_design.md)
- [Design Documents](docs/design/)
- [Contributing Guide](docs/contributing_energy.md)

## Installation

### Option 1: Development Mode (Recommended)

```bash
git clone <repository-url>
cd infodynamics-jax
pip install -e .
```

### Option 2: Manual Dependencies

```bash
pip install jax jaxlib optax numpy scipy matplotlib jupyter
```

## License

[Add your license here]

## Related Projects

This library is designed to be used by application-specific projects:
- `probabilistic-image-synthesis`: Image synthesis application (uses this library)
- Future: RL, EGPF, RJVMC applications
