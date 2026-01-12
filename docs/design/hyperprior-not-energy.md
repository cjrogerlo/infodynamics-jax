# Hyperprior is Not Energy

## Problem

Hyperprior should not be `EnergyTerm`, because it does not conform to the definition of energy.

## Definition of Energy

According to `docs/energy_design.md`:

```
E(phi) = E_{q(f | phi)}[ -log p(y | f, phi) ]
```

This is **inertial energy**, which is data-dependent.

## What is Hyperprior?

Hyperprior is `-log p(phi)`, i.e., the prior on hyperparameters.

This is **not** energy, because:
- Energy is data-dependent: `E[-log p(y|f,phi)]`
- Hyperprior is data-independent: `-log p(phi)`

## Design Problem

### Current Design (Incorrect)

```python
class HyperpriorEnergy(EnergyTerm):  # ❌ Should not inherit EnergyTerm
    ...
```

**Problems**:
- Hyperprior is not energy
- Violates energy layer design principles
- Confuses conceptual layers

### Correct Design

Hyperprior should:
1. **Not be EnergyTerm** (does not conform to energy definition)
2. **Can be added as term to TargetEnergy** (through composition)
3. **Or should be handled in inference layer** (MAP-II can add hyperprior)

## Solutions

### Option 1: As Independent Term (Recommended)

```python
# Does not inherit EnergyTerm, but can be used as term
class HyperpriorTerm:
    """Hyperprior on φ (not an energy, but a regularization term)."""
    def __call__(self, phi, X=None, Y=None, key=None):
        return -log p(phi)  # or negative of log p(phi)
```

Then in `TargetEnergy`:
```python
@dataclass(frozen=True)
class TargetEnergy(EnergyTerm):
    inertial: EnergyTerm
    prior: Optional[EnergyTerm] = None  # Prior on X
    hyperprior: Optional[Callable] = None  # -log p(phi), not EnergyTerm
    extra: Optional[Sequence[EnergyTerm]] = None
```

### Option 2: Handle in Inference Layer

```python
# map2.py can accept hyperprior as additional parameter
def run(
    self,
    energy: EnergyTerm,
    phi_init,
    *,
    hyperprior: Optional[Callable] = None,  # -log p(phi)
    ...
):
    def objective(phi):
        E = energy(phi, *energy_args, **energy_kwargs)
        if hyperprior is not None:
            E = E + hyperprior(phi)
        return E
    ...
```

### Option 3: Through `extra` Parameter (Currently Feasible)

```python
# Not called HyperpriorEnergy, but as ordinary function
def hyperprior_term(phi, X, Y, key=None):
    return -log p(phi)

target = TargetEnergy(
    inertial=inertial_energy,
    extra=[hyperprior_term],  # As extra term
)
```

## Recommended Solution

**Hybrid of Option 1 + Option 3**:

1. **Do not create `HyperpriorEnergy` class** (violates design principles)
2. **Provide utility functions** (not EnergyTerm)
3. **Add through `TargetEnergy.extra` or `hyperprior` parameter**

```python
# Utility function (not EnergyTerm)
def hyperprior_l2(phi, fields=None, lam=1.0):
    """L2 hyperprior on kernel params (not an energy)."""
    ...

# Usage
target = TargetEnergy(
    inertial=inertial_energy,
    extra=[lambda phi, X, Y, key=None: hyperprior_l2(phi)],
)
```

Or:

```python
# Handle in inference layer
map2.run(
    energy=target_energy,
    phi_init=phi_init,
    hyperprior=lambda phi: hyperprior_l2(phi),  # Add in inference layer
    ...
)
```

## Key Distinctions

| Concept | Definition | Is Energy? |
|---------|------------|------------|
| Inertial Energy | `E[-log p(y|f,phi)]` | ✅ Yes |
| Prior on X | `-log p(X)` | ⚠️ Can be EnergyTerm (but only depends on X) |
| Hyperprior | `-log p(phi)` | ❌ No (data-independent, not energy) |

## Conclusion

Hyperprior **should not be EnergyTerm**, because:
1. It does not conform to energy definition (not `E[-log p(y|f,phi)]`)
2. It is data-independent (energy is data-dependent)
3. It should be handled in inference layer or as composition term

**Recommendation**:
- Remove `HyperpriorEnergy` class
- Provide utility functions (not EnergyTerm)
- Add through `TargetEnergy.extra` or inference layer
