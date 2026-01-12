# SMC vs Rejuvenation: Should They Be Merged?

## Current Situation

### `resampling.py` (53 lines)
- `multinomial_resample`: Core SMC operation (resampling)
- `effective_sample_size`: Core SMC operation (ESS computation)
- **Purpose**: **Required** operations for SMC algorithms

### `rejuvenation.py` (448 lines)
- `hmc_rejuvenate`, `mala_rejuvenate`, `nuts_rejuvenate`: Rejuvenation kernels
- `kinetic_energy`: Helper function
- **Purpose**: **Optional** step for SMC (rejuvenation)

## Considerations for Merging

### Advantages

1. **Unified management**:
   - All particle-related utilities in one place
   - Reduces number of files

2. **Relevance**:
   - Both belong to SMC/particle methods
   - Both are in `inference/particle/` directory

### Disadvantages

1. **File size**:
   - Merged would exceed 500 lines (53 + 448 = 501 lines)
   - May be too long, not conducive to maintenance

2. **Conceptual differences**:
   - `resampling.py`: SMC **core** operations (resampling, ESS)
   - `rejuvenation.py`: SMC **optional** step (rejuvenation)
   - Conceptually slightly different

3. **Reusability**:
   - Rejuvenation kernels can theoretically be used by other methods
   - Although currently only used in particle methods

4. **Separation of concerns**:
   - `resampling.py`: Pure SMC operations (does not depend on energy)
   - `rejuvenation.py`: Depends on energy (needs energy_fn)

## Recommendation

### Option 1: Keep Separate (Recommended)

**Reasons**:
- ✅ Clear concepts: core SMC operations vs optional rejuvenation
- ✅ Reasonable file sizes: `resampling.py` (53 lines) and `rejuvenation.py` (448 lines)
- ✅ Separation of concerns: `resampling.py` does not depend on energy, `rejuvenation.py` depends on energy
- ✅ Easy to maintain: each file focuses on one concept

**Current structure**:
```
inference/particle/
  ├── resampling.py   # Core SMC operations (resampling, ESS)
  ├── rejuvenation.py # Optional rejuvenation kernels
  ├── annealed.py     # β-annealed SMC
  └── ibis.py         # IBIS
```

### Option 2: Merge into `resampling.py`

**If merged**:
- File would exceed 500 lines
- Mixes core operations and optional steps
- But all particle utilities in one place

## Conclusion

**Recommend keeping separate**, because:

1. **Clear concepts**:
   - `resampling.py`: Core SMC operations (required)
   - `rejuvenation.py`: Rejuvenation kernels (optional)

2. **File size**:
   - Current separation is easier to maintain
   - Merged would exceed 500 lines

3. **Separation of concerns**:
   - `resampling.py` does not depend on energy
   - `rejuvenation.py` depends on energy

4. **Future extensions**:
   - If other methods need rejuvenation in the future, can reuse
   - Keeping separate is more flexible

**Current design is reasonable, no need to merge.**
