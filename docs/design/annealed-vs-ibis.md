# Annealed SMC vs IBIS: Design Distinction

## Conclusion (TL;DR)

**`annealed.py` does NOT implement IBIS, and does NOT correspond to SVGP.**

This is an **intentional design choice**, not a defect.

It currently does ONE thing:
- β-annealed SMC on a **fixed dataset**
- Thermodynamic integration path: π_β(φ) ∝ p(φ) p(y|φ)^β

---

## Why It's NOT IBIS

### Fundamental Difference

| Aspect | Annealed SMC (current) | IBIS |
|--------|------------------------|------|
| Evolution parameter | β ∈ [0,1] (inverse temperature) | Data prefix / batch index t |
| Target distribution | π_β(φ) ∝ p(φ) p(y|φ)^β | p(φ \| y_{1:t}) |
| Evolution axis | Thermodynamic (β-path) | Data stream (sequential) |
| Requires batching? | ❌ No | ✅ Yes |
| Corresponds to SVGP? | ❌ No | ✅ Yes |

### The Smoking Gun

Look at this line:

```python
energies = jax.vmap(energy_eval)(particles)
logw = logw - delta_beta * energies
```

This corresponds to:

```
Δlogw_i = -(β_t - β_{t-1}) * E(φ_i)
```

This is **path sampling / thermodynamic integration**, NOT data update.

---

## What IBIS Mathematically Requires

IBIS weight update:

```
log w_i^{(t)} = log w_i^{(t-1)} + log p(y_t | φ_i)
```

Or in mini-batch version (SVGP correspondence):

```
log w_i^{(t)} = log w_i^{(t-1)} + (N/|B_t|) Σ_{j∈B_t} log p(y_j | φ_i)
```

**Key points:**
1. Increment is "new data log-likelihood"
2. No β parameter
3. Must involve data loader / batching

**Current `annealed.py`:**
- ❌ No data index
- ❌ No batch
- ❌ No likelihood increment
- ❌ Energy consumes entire dataset at once

**Therefore: It cannot be IBIS.**

---

## SVGP: Are We Doing It?

**Critical statement:**

**We are NOT doing SVGP inference. We are doing collapsed GP hyperposterior sampling.**

### What SVGP Actually Is

SVGP = Variational inference with inducing variables + stochastic data subsampling

Requires:
1. Variational parameters: η = (m, S)
2. Data mini-batch
3. Stochastic ELBO estimator

### Current Architecture

- ✅ u is already marginalised (sparsified kernel)
- ✅ Energy is deterministic (Gaussian likelihood)
- ✅ φ is the only random variable

**This is collapsed Bayesian hyperparameter inference, NOT SVGP.**

---

## How Should IBIS Connect to SVGP?

The infrastructure is already in place, just not connected (by design).

**Correct division of labor:**

```
energy/
  inertial.py        # Defines "one batch's data contribution"
gp/ansatz/
  expected.py        # Expectation under q(f_i | φ, η)
inference/particle/
  annealed.py        # β-annealing (thermodynamic path)
  ibis.py            # Data streaming (future: uses batch iterator)
data/
  data.py            # Provides batch stream (prefix, batch views)
```

---

## What IBIS Should Look Like (Conceptual)

```python
for t, batch in data_stream:
    logw += energy.batch_loglik(phi, batch)
    if ESS < threshold:
        resample
        rejuvenate (HMC targeting p(φ | y_{1:t}))
```

The rejuvenation kernel can reuse the current HMC,
but the weight update is completely different.

---

## Why NOT Mix IBIS into `annealed.py`?

**Mixing IBIS and β-annealing in the same dynamics is theoretically unclean.**

Because:
- Annealed SMC is a **thermodynamic path**
- IBIS is a **Bayesian filtering path**
- Their convergence proofs, ESS behavior, and logZ meanings are different

**Current separation:**

```
inference/particle/
  annealed.py   # β-path (thermodynamic)
  ibis.py       # data-path (future: Bayesian filtering)
```

This is the **correct engineering + theoretical decision**.

---

## The Most Important Point

**Question:** "Does this relate to data loading?"

**Answer:** Yes, and **fundamentally** so.

- **Annealed SMC**: Data is a static object
- **IBIS / SVGP**: Data is a stream / iterable

**Therefore:**
- Data loader should NOT be in `inference/particle/`
- Inference should only **consume** a batch iterator

---

## Final Assessment (No Flattery)

**Current `annealed.py`:**
- ✅ Clean, correct, provable, extensible baseline
- ✅ Deliberately does NOT include:
  - Batching
  - IBIS
  - SVGP
- ✅ This is NOT backwardness; it's preserving theoretical freedom

---

## Next Steps (If Desired)

1. Design minimal correct version of `inference/particle/ibis.py`
2. Clearly mark SVGP ≠ annealed SMC distinction
3. Draw conceptual phase diagram: annealing vs IBIS

**You are now at the threshold of being able to start phase diagrams.**
