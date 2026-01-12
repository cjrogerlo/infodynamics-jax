# Algorithm Layers and Ansatz Usage

## Core Problem

Different algorithm layers need different abstractions:

1. **`sampling/` (HMC, NUTS, MALA)**
   - **Layer**: Pure sampler
   - **Needs**: `energy(phi, ...) -> scalar`
   - **Ansatz**: ✅ Already handled internally through `InertialEnergy`
   - **Does not care**: How energy is computed internally

2. **`optimisation/map2.py`**
   - **Layer**: Optimize energy
   - **Needs**: `energy(phi, ...) -> scalar`
   - **Ansatz**: ✅ Already handled internally through `InertialEnergy`
   - **Does not care**: How energy is computed internally

3. **`particle/annealed.py`**
   - **Layer**: Uses energy for β-annealing
   - **Needs**: `energy(phi, ...) -> scalar`
   - **Ansatz**: ✅ Already handled internally through `InertialEnergy`
   - **Does not care**: How energy is computed internally

4. **`particle/ibis.py`** ⚠️
   - **Layer**: Needs `log p(y|φ)`, not just energy
   - **Problem**: `E[-log p(y|f,φ)] ≠ -log p(y|φ)` (non-conjugate)
   - **Ansatz**: ❌ Needs to **re-call** ansatz to compute `log p(y|φ)`

## Key Problem

### What Does IBIS Need?

IBIS weight update:
```
logw += log p(y_t | φ)
```

But `InertialEnergy` provides:
```
E(φ) = E_{q(f|φ)}[-log p(y|f,φ)]
```

For non-conjugate:
- `E[-log p(y|f,φ)]` ≠ `-log E[p(y|f,φ)]` (Jensen's inequality)
- Need to compute `log ∫ p(y|f,φ) q(f|φ) df`

### What Does Ansatz Already Do?

`InertialEnergy` internally calls ansatz to compute:
```
E[-log p(y|f,φ)] = Σ_i E_{q(f_i|φ)}[-log p(y_i|f_i,φ)]
```

But IBIS needs:
```
log p(y|φ) = log ∫ p(y|f,φ) q(f|φ) df
```

This is a **different computation**!

## Solutions

### Option 1: IBIS Directly Calls Ansatz (Recommended)

Let IBIS access `InertialEnergy`'s internal components to compute `log p(y|φ)`:

```python
# IBIS needs to be able to:
# 1. Get q(f_i|φ) marginals (ansatz already computed)
# 2. Compute log ∫ p(y_i|f_i,φ) q(f_i|φ) df_i (needs new ansatz function)
# 3. Sum over all i
```

**Problem**: This requires exposing ansatz internals, violates encapsulation

### Option 2: Add `log_likelihood` Method to `InertialEnergy`

```python
class InertialEnergy(EnergyTerm):
    def __call__(self, phi, X, Y, key=None):
        # Existing energy computation
        ...
    
    def log_likelihood(self, phi, X, Y, key=None):
        # Compute log p(y|φ) using ansatz
        # For Gaussian: use analytic
        # For non-conjugate: use ansatz
        ...
```

**Problem**: Violates energy layer design principles (does not provide marginal likelihood)

### Option 3: Create Dedicated `LogLikelihoodTerm` (Recommended)

```python
class LogLikelihoodTerm:
    """Dedicated log likelihood computation for IBIS"""
    def __init__(self, inertial_energy: InertialEnergy):
        # Reuse InertialEnergy's configuration (kernel, likelihood, estimator)
        ...
    
    def __call__(self, phi, X, Y, key=None):
        # Use ansatz to compute log p(y|φ)
        # Reuse InertialEnergy's internal logic
        ...
```

**Advantages**:
- ✅ Does not violate energy layer design (this is new term, not energy)
- ✅ Reuses `InertialEnergy`'s configuration and ansatz logic
- ✅ IBIS can accept `LogLikelihoodTerm` or `EnergyTerm`

### Option 4: Let IBIS Accept `InertialEnergy` and Call Ansatz Internally

```python
class IBIS:
    def _compute_log_likelihood_from_inertial(
        self, 
        inertial: InertialEnergy,
        phi, X, Y, key
    ):
        # Check estimator
        if inertial.cfg.estimator == "analytic":
            # Gaussian: use -energy
            return -inertial(phi, X, Y, key=key)
        else:
            # Non-conjugate: need to call ansatz
            # Reuse InertialEnergy's internal logic
            state = inertial._solve_inner(phi, X, Y)
            # Compute log p(y|φ) instead of E[-log p(y|f,φ)]
            ...
```

**Problem**: Requires accessing `InertialEnergy`'s private methods

## Recommended Solution

**Hybrid of Option 3 + Option 4**:

1. Create `LogLikelihoodTerm` as new protocol/interface
2. Let `InertialEnergy` be convertible to `LogLikelihoodTerm`
3. IBIS accepts `LogLikelihoodTerm` or `EnergyTerm` (backward compatible)

This way:
- ✅ Maintains energy layer design principles
- ✅ IBIS can correctly handle non-conjugate
- ✅ Reuses existing ansatz logic
- ✅ Clear abstraction layers
