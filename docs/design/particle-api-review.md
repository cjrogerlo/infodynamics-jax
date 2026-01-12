# Particle Inference API Review Report

## Conclusion

✅ **All files are clean, responsibilities are clearly separated, API design is consistent**

---

## Responsibility Separation

### 1. `annealed.py` - β-annealed SMC
**Responsibilities:**
- Implements β-annealing (thermodynamic path)
- Temperature annealing on fixed dataset
- Weight update: `Δlogw = -Δβ * E(φ)`

**Not responsible for:**
- ❌ Data streaming
- ❌ IBIS logic
- ❌ SVGP-specific inference

### 2. `ibis.py` - IBIS (Iterated Batch Importance Sampling)
**Responsibilities:**
- Implements data streaming (Bayesian filtering path)
- Processes data stream, updates `p(φ | y_{1:t})`
- Weight update: `logw += log p(y_t | φ)`

**Not responsible for:**
- ❌ β-annealing
- ❌ Fixed dataset processing

### 3. `resampling.py` - SMC Core Utility Functions
**Responsibilities:**
- `multinomial_resample`: Multinomial resampling
- `effective_sample_size`: ESS computation

**Design principles:**
- Only contains pure functions, stateless
- Can be reused by multiple particle methods

---

## API Design

### `InferenceMethod` Protocol Compliance

Both implement the `InferenceMethod` protocol, but `run()` signatures differ (this is allowed):

#### `AnnealedSMC.run()`
```python
def run(
    self, 
    energy: EnergyTerm, 
    init_particles_fn: Callable[[jax.random.PRNGKey, int], Any], 
    *, 
    key, 
    energy_args=(), 
    energy_kwargs=None
) -> SMCRun
```

**Features:**
- Accepts `energy_args` (fixed dataset)
- Does not need `data_stream`
- Returns `SMCRun` (contains `betas`)

#### `IBIS.run()`
```python
def run(
    self,
    energy: EnergyTerm,
    init_particles_fn: Callable[[jax.random.PRNGKey, int], Any],
    data_stream: Union[Iterator[SupervisedData], list[SupervisedData]],
    *,
    key: jax.random.PRNGKey,
    energy_kwargs: Optional[dict] = None,
) -> IBISRun
```

**Features:**
- Accepts `data_stream` (data stream)
- Does not need `energy_args` (data is in stream)
- Returns `IBISRun` (contains `logZ_trace`, `time_steps`)

**Design rationale:**
- ✅ Signature differences reflect essential method differences
- ✅ Both follow the spirit of `InferenceMethod` protocol
- ✅ Both accept `energy: EnergyTerm` (core contract)

---

## Code Duplication Handling

### ✅ Extracted to `resampling.py`
- `multinomial_resample()`
- `effective_sample_size()`

### ✅ Preserved differences (reasonable)
- `_hmc_kernel()` exists in both files, but:
  - `annealed.py`: targets `β * E(φ)` (tempered)
  - `ibis.py`: targets `E(φ; y_{1:t})` (full posterior)
  - Implementation differences are necessary, should not merge

---

## Export Check

### `inference/particle/__init__.py`
```python
from .annealed import AnnealedSMC, AnnealedSMCCFG, SMCRun
from .ibis import IBIS, IBISCFG, IBISRun

__all__ = [
    "AnnealedSMC", "AnnealedSMCCFG", "SMCRun",
    "IBIS", "IBISCFG", "IBISRun",
]
```
✅ Correctly exports all public APIs

### `inference/__init__.py`
```python
from .particle import (
    AnnealedSMC, AnnealedSMCCFG, SMCRun,
    IBIS, IBISCFG, IBISRun
)

__all__ = [
    ...
    "AnnealedSMC", "AnnealedSMCCFG", "SMCRun",
    "IBIS", "IBISCFG", "IBISRun",
]
```
✅ Correctly exports to top level

---

## Documentation Completeness

### ✅ Module-level docstring
- `annealed.py`: Clearly states it is not IBIS, not SVGP
- `ibis.py`: Clearly states differences from β-annealing

### ✅ Class-level docstring
- Both clearly state:
  - Target distribution
  - Weight update formula
  - Evolution axis (thermodynamic vs data streaming)

### ✅ Method-level docstring
- `run()` methods have complete parameter descriptions
- Key differences are annotated

---

## Potential Improvements (Optional)

### 1. HMC Kernel Sharing (Low Priority)
Currently two `_hmc_kernel` implementations are similar but different. Could consider:
- Extract base HMC logic to shared function
- But preserve different energy function wrappers
- **Recommendation: Keep current state** (differences are necessary)

### 2. Type Hint Enhancement (Optional)
- `data_stream` could have more explicit type
- But current `Union[Iterator, list]` is sufficient

---

## Summary

### ✅ Responsibility Separation
- Both have clear responsibilities, no overlap
- Different theoretical foundations (thermodynamic vs Bayesian filtering)

### ✅ API Design
- Complies with `InferenceMethod` protocol
- Signature differences reflect method essence
- Exports are complete and consistent

### ✅ Code Quality
- Duplicated code extracted to `resampling.py`
- Necessary differences preserved (HMC kernel)
- Documentation is complete and clear

### ✅ Architectural Consistency
- Both use `EnergyTerm` as input
- Both follow same design principles
- Consistent with overall architecture (energy/, core/, infodynamics/)

**Conclusion: All files are clean, responsibilities are clear, API design is consistent and reasonable.**
