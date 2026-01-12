# Hyperprior Design

## Overview

Hyperprior is the prior on structural hyperparameters φ:
- `kernel_params`: kernel hyperparameters
- `Z`: inducing point locations
- `likelihood_params`: likelihood hyperparameters

**Important**: Hyperprior is **not** EnergyTerm, because it does not conform to the definition of energy (see `docs/design/hyperprior-not-energy.md`).

## Design Principles

1. **Hyperprior is not Energy**:
   - Energy = `E[-log p(y|f,phi)]` (data-dependent)
   - Hyperprior = `-log p(phi)` (data-independent)

2. **General utility functions**:
   - Provide utility functions (not EnergyTerm)
   - Can be used in all inference methods

3. **Selective application**:
   - Different kernels/likelihoods use different parameter subsets
   - Selectively apply priors through `fields` and `keys` parameters

## Location

Hyperprior utility functions are located at:
- `infodynamics_jax/infodynamics/hyperprior.py`

This is at the `infodynamics/` level, because:
- All inference methods need it (MAP2, HMC, NUTS, SMC, IBIS)
- Not energy layer (does not conform to energy definition)
- Not a single inference method (general utility)

## API

### Utility Functions

```python
from infodynamics_jax.infodynamics import make_hyperprior

# Create hyperprior function
hyperprior = make_hyperprior(
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance"],  # RBF kernel
    likelihood_log_lambda=0.1,
    likelihood_keys=["noise_var"],  # Gaussian likelihood
)
```

### Usage

#### Method 1: Add at runner level (Recommended)

```python
from infodynamics_jax.infodynamics import run, make_hyperprior
from infodynamics_jax.inference.sampling import HMC, HMCCFG

hyperprior = make_hyperprior(
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance"],
)

out = run(
    key=key,
    method=HMC(cfg=HMCCFG(...)),
    energy=target_energy,
    phi_init=phi_init,
    energy_args=(X, Y),
    hyperprior=hyperprior,  # ✅ All methods support
)
```

#### Method 2: Add through TargetEnergy.extra

```python
from infodynamics_jax.energy import TargetEnergy
from infodynamics_jax.infodynamics import make_hyperprior

hyperprior = make_hyperprior(...)

target = TargetEnergy(
    inertial=inertial_energy,
    extra=[hyperprior],  # As extra term
)
```

## Handling Different Kernels/Likelihoods

### Problem

Different kernels/likelihoods use different parameters:
- **RBF kernel**: `lengthscale`, `variance`
- **Matern kernel**: `lengthscale`, `variance`, `nu`
- **Periodic kernel**: `lengthscale`, `variance`, `period`
- **Gaussian likelihood**: `noise_var`
- **Bernoulli likelihood**: (may have no hyperparameters)

### Solution: Selective Fields/Keys

Selectively apply priors through `fields` and `keys` parameters:

```python
# RBF kernel: only apply prior to lengthscale and variance
hyperprior = make_hyperprior(
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance"],  # Only select these two
)

# Matern kernel: apply prior to lengthscale, variance, nu
hyperprior = make_hyperprior(
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance", "nu"],  # Include nu
)

# Gaussian likelihood: apply prior to noise_var
hyperprior = make_hyperprior(
    likelihood_log_lambda=0.1,
    likelihood_keys=["noise_var"],
)

# Bernoulli likelihood: may not need hyperprior (no hyperparameters)
# Or can apply prior to other parameters (if any)
```

### Design Principles

1. **Selective application**: Select parameter subsets through `fields`/`keys`
2. **Backward compatible**: If `fields`/`keys` is None, use defaults
3. **Type safe**: Use `hasattr` and `in` to check if parameters exist
4. **Modular**: Different prior types (L2, log-L2) can be composed

## Supported Prior Types

### L2 Prior

```python
kernel_l2_hyperprior(phi, fields=["lengthscale"], lam=1.0)
# = 0.5 * lam * sum(phi.kernel_params.lengthscale ** 2)
```

### Log-L2 Prior (Log-Normal)

```python
kernel_log_l2_hyperprior(phi, fields=["lengthscale"], lam=1.0, mu={"lengthscale": 0.0})
# = 0.5 * lam * sum((log(lengthscale) - mu) ** 2)
```

Suitable for positive parameters (e.g., `lengthscale`, `variance`, `noise_var`).

### Z Prior

```python
z_l2_hyperprior(phi, lam=1.0)
# = 0.5 * lam * sum(phi.Z ** 2)
```

## Usage Examples

### Complete Example

```python
from infodynamics_jax.infodynamics import run, make_hyperprior
from infodynamics_jax.inference.optimisation import MAP2, MAP2CFG
from infodynamics_jax.energy import TargetEnergy, InertialEnergy

# Create hyperprior (select parameters based on kernel/likelihood)
hyperprior = make_hyperprior(
    # RBF kernel priors
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance"],
    
    # Gaussian likelihood prior
    likelihood_log_lambda=0.1,
    likelihood_keys=["noise_var"],
    
    # Inducing points prior
    z_lambda=0.01,
)

# Create energy
inertial = InertialEnergy(...)
target = TargetEnergy(inertial=inertial)

# MAP2 optimization (hyperprior added at runner level)
method = MAP2(cfg=MAP2CFG(steps=200, lr=1e-2))
out = run(
    key=key,
    method=method,
    energy=target,
    phi_init=phi_init,
    energy_args=(X, Y),
    hyperprior=hyperprior,  # ✅ All methods support
)
```

## Extensibility

If new prior types are needed, can:
1. Add new atomic functions in `infodynamics/hyperprior.py` (e.g., `kernel_l1_hyperprior`)
2. Add corresponding parameters in `make_hyperprior`
3. Add handling logic in `make_hyperprior` implementation

This design maintains:
- ✅ Generality (applicable to all kernels/likelihoods)
- ✅ Flexibility (selective application)
- ✅ Extensibility (easy to add new prior types)

## Differences from Old Design

### Old Design (Incorrect)

```python
# ❌ HyperpriorEnergy inherits EnergyTerm
class HyperpriorEnergy(EnergyTerm):
    ...
```

**Problems**:
- Hyperprior is not energy
- Violates energy layer design principles

### New Design (Correct)

```python
# ✅ Utility function (not EnergyTerm)
def make_hyperprior(...) -> Callable[[Phi], jnp.ndarray]:
    ...
```

**Advantages**:
- Conforms to design principles (hyperprior is not energy)
- General (all inference methods can use)
- Flexible (can add through runner or TargetEnergy.extra)
