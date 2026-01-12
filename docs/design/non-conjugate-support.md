# Non-Conjugate Support in Particle Methods

## Problem Analysis

### Current Status

1. **`annealed.py`**: ✅ **Supports non-conjugate**
   - Directly uses `energy(phi, *args, **kwargs)`
   - If `energy` is `InertialEnergy`, it already handles non-conjugate (through GH/MC estimators)
   - No additional processing needed

2. **`ibis.py`**: ⚠️ **Partially supports, but has issues**
   - Uses `-energy` as approximation for log likelihood
   - For Gaussian: `log p(y|φ) ≈ -E[-log p(y|f,φ)]` ✅
   - For non-conjugate: this is an **approximation**, not accurate ❌

3. **`sampling/` methods**: ✅ **Supports non-conjugate**
   - Only needs energy function
   - Does not care how energy is computed internally (analytic/GH/MC)
   - This is correct design: inference layer should not care about energy's internal implementation

### Core Problem

**IBIS needs `log p(y_t | φ)`, but energy layer only provides `E[-log p(y|f,φ)]`**

For non-conjugate:
- `E[-log p(y|f,φ)] ≠ -log p(y|φ)`
- Due to Jensen's inequality: `E[-log p(y|f,φ)] ≥ -log E[p(y|f,φ)]`

---

## Design Layer Analysis

### Algorithm Layer Differences

1. **`sampling/` (HMC, NUTS, MALA)**
   - **Layer**: Pure sampler, only cares about energy landscape
   - **Input**: `energy(phi, ...) -> scalar`
   - **Does not care**: How energy is computed (analytic/GH/MC)
   - **Supports non-conjugate**: ✅ Automatically supported (through energy layer)

2. **`particle/annealed.py`**
   - **Layer**: Uses energy for β-annealing
   - **Input**: `energy(phi, ...) -> scalar`
   - **Does not care**: How energy is computed
   - **Supports non-conjugate**: ✅ Automatically supported (through energy layer)

3. **`particle/ibis.py`**
   - **Layer**: Needs log likelihood, not just energy
   - **Input**: `energy(phi, ...) -> scalar`, but needs `log p(y|φ)`
   - **Problem**: Energy layer does not provide `log p(y|φ)`
   - **Supports non-conjugate**: ⚠️ Currently approximate

---

## Solutions

### Option 1: Add log_likelihood Method to Energy Layer (Not Recommended)

**Problem**: Violates design principles
- Energy layer explicitly does not provide `marginal_likelihood` or `log_evidence`
- This would break the "inference-as-dynamics" design philosophy

### Option 2: Accept log_likelihood_fn in IBIS (Recommended)

**Design**: IBIS accepts optional `log_likelihood_fn`, uses it if provided, otherwise falls back to `-energy` approximation

```python
def run(
    self,
    energy: EnergyTerm,
    init_particles_fn: Callable,
    data_stream: ...,
    *,
    key: jax.random.PRNGKey,
    log_likelihood_fn: Optional[Callable] = None,  # New
    energy_kwargs: Optional[dict] = None,
) -> IBISRun:
```

**Advantages**:
- Does not violate energy layer design principles
- For Gaussian, can compute through `vfe_objective` or analytic
- For non-conjugate, users can provide their own implementation
- Backward compatible (default uses `-energy` approximation)

### Option 3: Use TargetEnergy Composition (Recommended)

**Design**: IBIS accepts `TargetEnergy`, which already composes inertial + prior

```python
# User composes
target_energy = TargetEnergy(
    inertial=InertialEnergy(...),  # Handles non-conjugate
    prior=PriorEnergy(...)
)

# IBIS uses
ibis.run(
    energy=target_energy,  # Already handles non-conjugate
    ...
)
```

**But problem**: IBIS still needs `log p(y|φ)`, not just `E[-log p(y|f,φ)]`

---

## Final Recommendation

### For `annealed.py`: ✅ No modification needed
- Already correctly uses energy
- Automatically supports non-conjugate (through energy layer)

### For `ibis.py`: Add `log_likelihood_fn` parameter

**Implementation strategy**:
1. If `log_likelihood_fn` is provided, use it
2. Otherwise, use `-energy` as approximation (backward compatible)
3. Clearly document:
   - Gaussian: `-energy` is accurate
   - Non-conjugate: Need to provide `log_likelihood_fn` or accept approximation

**Design rationale**:
- Does not violate energy layer design principles
- Maintains inference layer abstraction (does not care about energy internals)
- Gives users flexibility to provide accurate log likelihood (if needed)

---

## Summary

1. **`annealed.py`**: ✅ Supports non-conjugate (through energy layer)
2. **`ibis.py`**: ⚠️ Needs improvement (add `log_likelihood_fn` parameter)
3. **`sampling/`**: ✅ Supports non-conjugate (through energy layer)

**Key insight**: 
- `sampling/` and `annealed.py` only need energy, so automatically support non-conjugate
- `ibis.py` needs log likelihood, which is beyond energy layer's scope
- Solution: Let IBIS accept optional `log_likelihood_fn`, keeping design layers clear
