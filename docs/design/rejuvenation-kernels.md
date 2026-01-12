# Rejuvenation Kernels Design

## Current Situation

### Implemented Rejuvenation Kernels

Currently only **HMC** as a rejuvenation kernel:

1. **`annealed.py`**:
   - `_hmc_kernel()`: Internally implemented HMC kernel
   - Target: tempered energy `U_beta = beta * E(phi)`
   - Configuration: `rejuvenation: str = "hmc"`

2. **`ibis.py`**:
   - `_hmc_kernel()`: Internally implemented HMC kernel
   - Target: full posterior `p(phi | y_{1:t})`
   - Configuration: `rejuvenation: str = "hmc"` or `None`

### Problems

1. **Code duplication**:
   - Both `annealed.py` and `ibis.py` have their own `_hmc_kernel` implementations
   - The two implementations are almost identical, only energy handling is slightly different

2. **Missing other kernels**:
   - No MALA, NUTS, Slice, etc. as rejuvenation kernels
   - But these are already implemented in `inference/sampling/`

3. **Cannot reuse**:
   - Kernels in `inference/sampling/` are implemented as `InferenceMethod`
   - Not directly reusable kernel functions

## Design Options

### Option 1: Keep Current State (Simple but Duplicated)

**Advantages**:
- ✅ Simple, no refactoring needed
- ✅ Each method can independently adjust kernel

**Disadvantages**:
- ❌ Code duplication
- ❌ Cannot reuse other kernels (MALA, NUTS, Slice)

### Option 2: Extract Shared Rejuvenation Kernel Module (Recommended)

Create `inference/particle/rejuvenation.py`:

```python
# inference/particle/rejuvenation.py
"""
Rejuvenation kernels for particle-based methods.

These kernels are used to refresh particles after resampling in SMC methods.
They target either:
  - Tempered distribution: π_β(φ) ∝ p(φ) p(y|φ)^β (for Annealed SMC)
  - Full posterior: p(φ | y_{1:t}) (for IBIS)
"""

def hmc_rejuvenate(key, particles, energy_fn, step_size, n_leapfrog, n_steps):
    """HMC rejuvenation kernel."""
    ...

def mala_rejuvenate(key, particles, energy_fn, step_size, n_steps):
    """MALA rejuvenation kernel."""
    ...

def nuts_rejuvenate(key, particles, energy_fn, step_size, n_steps):
    """NUTS rejuvenation kernel."""
    ...
```

**Advantages**:
- ✅ Eliminates code duplication
- ✅ Can add various rejuvenation kernels
- ✅ Can reuse logic from `inference/sampling/`

**Disadvantages**:
- ⚠️ Requires refactoring existing code

### Option 3: Reuse Kernels from `inference/sampling/`

**Problem**:
- Kernels in `inference/sampling/` are `InferenceMethod`, not functions
- Their interface is `run(energy, phi_init, ...)`, not `kernel(key, particles, ...)`

**Solution**:
- Extract core kernel logic from `inference/sampling/` as functions
- Or create adapter to convert `InferenceMethod` to rejuvenation kernel

**Advantages**:
- ✅ Maximizes code reuse
- ✅ Unifies all MCMC kernels

**Disadvantages**:
- ⚠️ Requires larger refactoring
- ⚠️ May be over-engineered

## Recommended Approach

**Option 2 (Extract Shared Module)**, because:

1. **Moderate abstraction**:
   - Don't need to refactor entire `inference/sampling/`
   - Only need to extract parts needed for rejuvenation

2. **Clear responsibilities**:
   - `inference/sampling/`: Standalone MCMC methods
   - `inference/particle/rejuvenation.py`: Rejuvenation kernels in SMC

3. **Easy to extend**:
   - Can easily add new rejuvenation kernels
   - Don't need to modify `inference/sampling/`

## Implementation Suggestions

### 1. Create `rejuvenation.py`

```python
# inference/particle/rejuvenation.py
"""
Rejuvenation kernels for particle-based methods.
"""

def hmc_rejuvenate(
    key,
    particles,  # pytree stacked [P, ...]
    energy_fn,  # function(phi) -> scalar
    step_size: float = 1e-2,
    n_leapfrog: int = 4,
    n_steps: int = 1,
) -> Any:
    """
    HMC rejuvenation kernel.
    
    Args:
        key: PRNG key
        particles: Stacked particles pytree [P, ...]
        energy_fn: Energy function (phi) -> scalar
        step_size: HMC step size
        n_leapfrog: Number of leapfrog steps
        n_steps: Number of HMC steps per particle
    
    Returns:
        rejuvenated_particles: pytree stacked [P, ...]
    """
    ...
```

### 2. Update `annealed.py` and `ibis.py`

```python
# annealed.py
from .rejuvenation import hmc_rejuvenate

# In run():
if rejuvenation == "hmc":
    def energy_fn(phi):
        return beta * energy(phi, *energy_args, **energy_kwargs)
    particles = hmc_rejuvenate(
        key_rejuv, particles, energy_fn,
        step_size=self.cfg.step_size,
        n_leapfrog=self.cfg.n_leapfrog,
        n_steps=self.cfg.rejuvenation_steps,
    )
```

### 3. Add Other Kernels (Optional)

```python
# Can add in the future:
if rejuvenation == "mala":
    particles = mala_rejuvenate(...)
elif rejuvenation == "nuts":
    particles = nuts_rejuvenate(...)
```

## Configuration Updates

```python
@dataclass(frozen=True)
class AnnealedSMCCFG:
    ...
    rejuvenation: str = "hmc"  # "hmc", "mala", "nuts", None
    rejuvenation_steps: int = 1
    step_size: float = 1e-2  # For HMC/MALA
    n_leapfrog: int = 4  # For HMC
```

## Conclusion

**Current situation**:
- ✅ Only HMC as rejuvenation kernel
- ❌ Code duplication (`annealed.py` and `ibis.py` both have `_hmc_kernel`)

**Recommendation**:
- Create `inference/particle/rejuvenation.py` to extract shared logic
- Can add MALA, NUTS, etc. as options in the future
- Keep independence from `inference/sampling/` (different use cases)
