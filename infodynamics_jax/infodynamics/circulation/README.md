Circulation (Regime-1)
======================

This package defines *solenoidal/antisymmetric* operators used to create
nonreversible probability currents without changing the target density.

Scope
-----
- Form and apply an antisymmetric operator C (implicitly; no dense d x d matrices).
- Build plane/bivector bases (e.g. PCA planes from particles, canonical planes).
- Provide coefficient ("gating") utilities for omega(stats(Y)).
- Provide small, optional adapters that modify *provided* gradients, but do not
  compute gradients themselves.

Non-goals
---------
- Implement samplers (MALA/HMC/SMC). Those live in `infodynamics_jax/inference/...`.
- Compute model energies or gradients. Those live in `infodynamics_jax/energy/...`
  and are consumed by inference code.

Regime-1 contract
-----------------
Within a mutation block (e.g. inner rejuvenation steps at a fixed annealing stage),
the operator must be constant in the dynamical variable X. In practice this means:
- build planes and omega once per stage (after resampling),
- reuse them for all inner steps at that stage.

Scheduling
----------
The annealing schedule should be applied **once** (typically at the stage level).
Gating functions should return raw omega values (no schedule baked in) to avoid
double-scaling.
