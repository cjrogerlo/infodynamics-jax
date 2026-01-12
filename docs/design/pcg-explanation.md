# PCG (Preconditioned Conjugate Gradient) Explanation

## What is PCG?

**PCG = Preconditioned Conjugate Gradient**

This is an **iterative method for solving linear systems**, used to solve:
```
K * x = b
```
where K is a large sparse or structured matrix.

## Why is PCG needed?

### Traditional Method: Cholesky Decomposition

For `K * x = b`, the standard approach is:
1. Compute `L = cholesky(K)` (O(M³) time, O(M²) space)
2. Solve `L * y = b`, then `L^T * x = y`

**Problems**:
- When M is large (> 10,000), Cholesky is very slow and memory-intensive
- Need to store the full L matrix (M × M)

### PCG Method

PCG is an **iterative method**:
- Does not need to build the full Cholesky decomposition
- Only needs to be able to compute `K * v` (matrix-vector product)
- Uses a preconditioner to accelerate convergence

**Advantages**:
- For large M, faster than Cholesky
- More memory efficient (does not need to store full L)
- Can stop early (when sufficient accuracy is reached)

## Application in GP

### The Problem We Need to Solve

In sparse GP, we often need:
```
K_uu^{-1} * v
```

where:
- `K_uu` is the kernel matrix between inducing points (M × M)
- `v` is some vector (M,)

### Current Implementation (Cholesky)

```python
# Our current implementation
L = jnp.linalg.cholesky(K_uu)  # O(M³)
x = jax.scipy.linalg.cho_solve((L, True), v)  # O(M²)
```

### If Using PCG

```python
# PCG implementation (pseudocode)
def pcg_solve(K_fn, b, preconditioner, max_iter=100, tol=1e-6):
    """
    K_fn: function v -> K * v (does not need to build full K)
    b: right-hand side vector
    preconditioner: preconditioner (accelerates convergence)
    """
    x = initial_guess
    r = b - K_fn(x)  # residual
    p = preconditioner(r)  # preconditioned residual
    
    for i in range(max_iter):
        Ap = K_fn(p)
        alpha = (r^T * preconditioner(r)) / (p^T * Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        if ||r_new|| < tol:
            break
        
        beta = (r_new^T * preconditioner(r_new)) / (r^T * preconditioner(r))
        p = preconditioner(r_new) + beta * p
        r = r_new
    
    return x
```

## Do We Need PCG?

### Current Situation

**Our typical M (number of inducing points)**:
- Small to medium scale: M < 1,000
- For M < 1,000, Cholesky typically:
  - ✅ Faster (O(M³) but small constant)
  - ✅ More stable (high numerical precision)
  - ✅ Simpler (JAX built-in support)

### When is PCG Needed?

**Cases that need PCG**:
- M > 10,000 (very large scale)
- Memory constrained (cannot store full L)
- Only need approximate solution (can stop early)

**Our situation**:
- ❌ Typically M < 1,000, PCG not needed
- ❌ We need exact solutions (for energy computation), cannot stop early
- ❌ JAX's Cholesky is already efficient (GPU accelerated)

## Why Does GPyTorch Need PCG?

### GPyTorch's Use Cases

**GPyTorch is a general-purpose library** that needs to support:
1. **Very large datasets**: N > 100,000
2. **Many inducing points**: M > 10,000
3. **Various application scenarios**: from small research to large industrial applications

**GPyTorch's goals**:
- Become "the most general GP library"
- Support as many application scenarios as possible
- Handle "extreme scale" problems

### Why GPyTorch Needs PCG

1. **Handling very large M**:
   - GPyTorch users may use M = 10,000+ inducing points
   - Cholesky becomes very slow for M > 10,000 (O(M³))
   - PCG can handle M = 100,000+ cases

2. **Memory efficiency**:
   - Large applications may be memory constrained
   - PCG does not need to store the full Cholesky decomposition
   - Can handle larger problems

3. **Generality**:
   - As a general library, needs to support various scales
   - PCG allows GPyTorch to handle "any scale" problems

## Why Don't We Need It?

### Our Design Goals Are Different

**Our library is a foundational library**, not a general-purpose application library:
1. **Focus on core functionality**: inference methods, energy functions
2. **Moderate scale**: M typically < 1,000 (from examples, M = 10)
3. **Conceptual clarity**: prioritize theoretical clarity over extreme scale

### Our Typical Scale

From code and examples:
- **Tests**: M = 6
- **Examples**: M = 10
- **Real applications**: M typically between 50-500

For M < 1,000:
- ✅ Cholesky is very fast (< 1 second)
- ✅ Memory requirements are reasonable (< 10 MB)
- ✅ Numerically stable (JAX optimized)

### Key Differences

| Feature | GPyTorch | Our Library |
|---------|----------|-------------|
| **Goal** | General GP library | Foundational inference library |
| **Typical M** | 100 - 10,000+ | 10 - 1,000 |
| **Need PCG** | ✅ Yes (supports very large scale) | ❌ No (moderate scale) |
| **Design Focus** | Generality, scalability | Conceptual clarity, theoretical correctness |

## Conclusion

**Why GPyTorch needs PCG**:
- As a general library, needs to support very large scale (M > 10,000)
- Needs to handle various application scenarios (from small to industrial scale)
- PCG allows it to handle "any scale" problems

**Why we don't need it**:
- Our M is typically < 1,000, Cholesky is sufficient
- We focus on conceptual clarity, not extreme scale
- Our design goals are different (foundational library vs general library)

**This is not a defect, but a design choice**:
- GPyTorch: Generality first → needs PCG
- Our library: Conceptual clarity first → Cholesky is sufficient

**Implementation Priority**:
- 🔴 Low priority: Not needed currently
- 🟡 Medium priority: If we need very large scale (M > 10,000) in the future, implement then
- 🟢 High priority: Keep current Cholesky implementation (already very good)
