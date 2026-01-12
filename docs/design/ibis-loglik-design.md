# IBIS log_likelihood_fn Design Analysis

## Problem

IBIS needs `log p(y_t | φ)` to update weights:
```
logw += log p(y_t | φ)
```

But energy layer only provides:
```
E(φ) = E_{q(f|φ)}[-log p(y|f,φ)]
```

For non-conjugate, due to Jensen's inequality:
```
E[-log p(y|f,φ)] ≠ -log p(y|φ)
```

## Current Design: `log_likelihood_fn` Parameter

### Advantages
- ✅ Does not violate energy layer design principles (does not require energy to provide marginal likelihood)
- ✅ Gives users flexibility
- ✅ Backward compatible (default uses `-energy` approximation)

### Disadvantages
- ⚠️ Requires users to provide additional function
- ⚠️ For Gaussian, theoretically not needed (but in practice `-energy` is sufficient)

## Alternative Solutions

### Solution 1: Completely Remove `log_likelihood_fn`, Only Use `-energy`

**Reasoning**:
- For Gaussian: `-energy` is accurate (up to constant)
- For non-conjugate: using approximation is reasonable (IBIS itself is an approximate method)

**Problem**:
- Cannot satisfy scenarios that need accurate log likelihood

### Solution 2: Let `InertialEnergy` Provide Optional `log_likelihood` Method

**Problem**:
- ❌ Violates energy layer design principles
- ❌ Energy layer explicitly does not provide `marginal_likelihood`

### Solution 3: Use Negative of `vfe_objective` (Gaussian Only)

**Problem**:
- ❌ `vfe_objective` is an optimization objective, not model energy
- ❌ Only applicable to Gaussian
- ❌ Mixes optimization and inference layers

### Solution 4: Keep Current Design, But Improve Documentation

**Recommendation**:
- Clearly state when `log_likelihood_fn` is needed
- Provide usage examples
- Explain that for Gaussian, `-energy` is sufficient

## Recommended Solution

**Keep current design (Solution 4)**, but:

1. **Improve default behavior**:
   - For Gaussian likelihood, automatically use `-energy` (accurate)
   - For non-conjugate, use `-energy` as approximation (unless `log_likelihood_fn` is provided)

2. **Documentation improvements**:
   - Clearly state: Gaussian does not need `log_likelihood_fn`
   - Explain: non-conjugate can use approximation, or provide accurate function

3. **Optional enhancement**:
   - Detect `InertialEnergy`'s `estimator`, if "analytic" (Gaussian), automatically use `-energy`
   - But this requires inspecting energy internal structure, may violate abstraction

## Conclusion

Current design is reasonable, but can:
1. Improve documentation, clearly state usage scenarios
2. Consider automatic detection of Gaussian case (if feasible and does not violate abstraction)
3. Provide usage examples

**Key point**: `log_likelihood_fn` is **optional**, default use of `-energy` approximation is sufficient for most scenarios.
