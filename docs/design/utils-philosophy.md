# Utils Design Philosophy

## Current Utils

### 1. `inference/particle/resampling.py`
- `multinomial_resample`: Resampling function needed for SMC core
- `effective_sample_size`: ESS computation needed for SMC core

**These are core utilities specific to SMC algorithms, not problem-specific.**

### 2. `gp/kernels/utils.py`
- `scaled_sqdist`: Internal utility function for kernel computation

**This is needed at the model level, not problem-specific.**

## Utils That Should Not Be Added

### Plotting / Visualization
**Reasons**:
- ❌ Very problem-specific (different problems need different plots)
- ❌ Adds dependencies (matplotlib, seaborn, etc.)
- ❌ Violates "library should be lightweight" principle
- ✅ Users can use external libraries in `examples/` or their own code

**Recommendation**:
- Show how to use matplotlib/seaborn for visualization in `examples/`
- Should not provide plotting tools in core library

### Metrics / Evaluation
**Reasons**:
- ❌ Very problem-specific (different problems need different metrics)
- ❌ Adds dependencies (scikit-learn, etc.)
- ❌ Violates "library should focus on inference" principle
- ✅ Users can compute metrics in their own code

**Recommendation**:
- Show how to compute common metrics (e.g., RMSE, accuracy) in `examples/`
- Should not provide metrics tools in core library

### Diagnostics Extensions
**Current design**:
- `runner.py` already provides basic diagnostics (`accept_rate`, `energy_trace`, `grad_norm_trace`)
- These are **general, algorithm-level** diagnostics

**Should not add**:
- ❌ Problem-specific diagnostics (e.g., regression metrics, classification metrics)
- ❌ Complex diagnostic tools (these should be in user code)

## Design Principles

### Utils That Should Be Included
1. **Needed for algorithm core** (e.g., `multinomial_resample`, `effective_sample_size`)
2. **Needed for model computation** (e.g., `scaled_sqdist`)
3. **General, non-problem-specific**

### Utils That Should Not Be Included
1. **Plotting / Visualization** (problem-specific)
2. **Metrics / Evaluation** (problem-specific)
3. **Complex diagnostic tools** (should be in user code)

## Suggested Structure

```
infodynamics_jax/
  ├── inference/particle/resampling.py  ✅ SMC core utilities
  ├── gp/kernels/utils.py               ✅ Model computation utilities
  └── (should not have)
      ├── utils/plotting.py        ❌ Should not
      ├── utils/metrics.py         ❌ Should not
      └── utils/diagnostics.py     ❌ Should not (already have basic diagnostics in runner.py)

examples/
  ├── ibis_annealed_smc.py         ✅ Can include simple plotting examples
  └── (can add)
      ├── plot_results.py          ✅ Show how to use matplotlib
      └── compute_metrics.py       ✅ Show how to compute metrics
```

## Conclusion

**Current design is correct**:
- ✅ Only includes utils needed for algorithm/model core
- ✅ Keeps library lightweight and general
- ✅ Does not include problem-specific tools

**If users need plotting/metrics**:
- Provide examples in `examples/`
- Or use external libraries in their own code (matplotlib, seaborn, scikit-learn, etc.)

**This conforms to the library's design principles**:
- Focus on inference dynamics
- Do not assume specific problem types
- Keep lightweight and composable
