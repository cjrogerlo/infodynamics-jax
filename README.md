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

```python
from infodynamics_jax.infodynamics import run
from infodynamics_jax.inference.sampling import HMC, HMCCFG
from infodynamics_jax.energy import TargetEnergy, InertialEnergy, PriorEnergy

# Define energy
inertial = InertialEnergy(kernel="rbf", likelihood="gaussian")
prior = PriorEnergy([...])
target = TargetEnergy(inertial=inertial, prior=prior)

# Run inference
method = HMC(cfg=HMCCFG(step_size=1e-2, n_samples=256))
out = run(
    key=key,
    method=method,
    energy=target,
    phi_init=phi_init,
    energy_args=(X, Y),
)
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

## Documentation

- [Energy Design](docs/energy_design.md)
- [Design Documents](docs/design/)
- [Contributing Guide](docs/contributing_energy.md)

## Installation

```bash
pip install jax jaxlib optax numpy scipy
```

## License

[Add your license here]

## Related Projects

This library is designed to be used by application-specific projects:
- `probabilistic-image-synthesis`: Image synthesis application (uses this library)
- Future: RL, EGPF, RJVMC applications
