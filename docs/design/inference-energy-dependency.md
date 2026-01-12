# Inference Layer's Dependency on Energy Layer

## Current Situation

### Does Inference Layer Know About Energy?

**Yes, but this is part of the design and is reasonable.**

### Dependency Relationship

1. **Protocol level**:
   - `inference/base.py` defines `InferenceMethod` protocol
   - All inference methods must accept `energy: EnergyTerm`
   - This is **interface dependency**, not implementation dependency

2. **Concrete implementation**:
   - All inference methods import `EnergyTerm`:
     - `sampling/hmc.py`, `mala.py`, `nuts.py`, `slice.py`
     - `optimisation/map2.py`, `vga.py`
     - `particle/annealed.py`, `ibis.py`
   - But **only use `EnergyTerm` protocol**, do not depend on concrete implementation

### Design Principles (from `base.py`)

```python
class InferenceMethod(Protocol):
    """
    - An InferenceMethod consumes an EnergyTerm and performs inference.
    - It MUST treat EnergyTerm as a black box (no inspection of internal structure).
    - It MUST NOT assume any specific energy signature beyond EnergyTerm contract.
    """
```

## Is This a Reasonable Design?

### ✅ Yes, this is reasonable

**Reasons**:

1. **Dependency injection pattern**:
   - Inference needs to know how to call energy (through protocol)
   - But should not know energy's internal implementation
   - This is standard dependency injection pattern

2. **Clear layering**:
   - `energy/`: Defines energy landscape
   - `inference/`: Defines dynamics (needs to know how to call energy)
   - `infodynamics/`: Composes both

3. **Conforms to design principles**:
   - Inference only depends on `EnergyTerm` protocol (interface)
   - Does not depend on concrete implementation (e.g., `InertialEnergy`, `TargetEnergy`)
   - This is correct application of **interface segregation principle**

### ❌ Dependencies That Should Not Exist

Check results show inference layer **does not have** the following inappropriate dependencies:
- ❌ Does not directly import `InertialEnergy`
- ❌ Does not directly import `TargetEnergy`
- ❌ Does not directly import `PriorEnergy`
- ❌ Does not inspect energy's internal structure

**All inference methods only use `EnergyTerm` protocol**.

## Comparison: If Inference Doesn't Know About Energy

### Option 1: Inference Doesn't Know About Energy (Not Recommended)

```python
# If inference doesn't know about energy
class InferenceMethod(Protocol):
    def run(self, objective_fn: Callable, *args, **kwargs) -> Any:
        # Use generic Callable, not EnergyTerm
        ...
```

**Problems**:
- ❌ Lose type safety
- ❌ Cannot express "this is energy" semantics
- ❌ Cannot define contract at protocol level

### Option 2: Inference Knows Energy Protocol (Current Design, Recommended)

```python
# Current design
class InferenceMethod(Protocol):
    def run(self, energy: EnergyTerm, *args, **kwargs) -> Any:
        # Clearly express: this is energy, not arbitrary function
        ...
```

**Advantages**:
- ✅ Type safe
- ✅ Clear semantics (this is energy, not arbitrary function)
- ✅ Can define contract at protocol level
- ✅ Still maintains abstraction (does not depend on concrete implementation)

## Conclusion

### Current Design is Correct

1. **Inference knows Energy Protocol**:
   - ✅ This is necessary (needs to know how to call)
   - ✅ This is interface dependency, not implementation dependency
   - ✅ Conforms to dependency injection pattern

2. **Inference doesn't know Energy implementation**:
   - ✅ Does not depend on concrete classes like `InertialEnergy`, `TargetEnergy`
   - ✅ Only uses `EnergyTerm` protocol
   - ✅ Treats energy as black box

3. **Clear layering**:
   - `energy/`: Defines energy landscape (implementation)
   - `inference/`: Defines dynamics (uses energy protocol)
   - `infodynamics/`: Composes both (orchestration)

### Design Principle Summary

```
energy/          → Defines EnergyTerm protocol + implementation
     ↓ (protocol dependency)
inference/       → Uses EnergyTerm protocol (black box)
     ↓ (composition)
infodynamics/    → Composes energy + inference
```

This is **correct layered architecture**:
- Upper layer can depend on lower layer's **interface** (protocol)
- Upper layer should not depend on lower layer's **implementation** (concrete classes)

## Check Results

All inference methods:
- ✅ Only import `EnergyTerm` (protocol)
- ✅ Do not import concrete implementation (e.g., `InertialEnergy`, `TargetEnergy`)
- ✅ Treat energy as black box (only call, do not inspect internal)

**Conclusion: Current design is correct, no need to modify.**
