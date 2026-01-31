# Circulation Layout (Regime-1)

This note codifies the circulation module boundaries and naming conventions so the code stays consistent and easy to extend.

## Scope

Circulation defines how the antisymmetric operator **C** is formed and how it is applied to a vector **v**. It does **not**:
- compute model energies or gradients
- implement samplers or mutation kernels

Sampling logic lives in `infodynamics_jax/inference/...`.

## Naming

- Use `lam` for the stage/annealing parameter (not `beta`).
- Use `v` for a generic vector; do not assume it is a gradient.
- If a function expects a gradient, that expectation belongs in the inference layer, not circulation.

## Schedule placement (single source)

Scheduling should be applied **once** at the stage level.

- Gating functions return **raw** `omega` (no schedule baked in).
- `StageFrozenCirculation.apply(v, lam)` applies the schedule and returns `C_lam v`.

This avoids double-scaling.

## Regime-1 contract

Within a mutation block (e.g., inner rejuvenation steps at fixed `lam`):
- `C` must be constant in the dynamical variable `X`.
- Build planes/omega once per stage (after resampling), reuse for all inner steps.

## Structural guidance for C

Conceptually, `C` acts on the vectorized latent (`NQ`), but implementations should avoid dense matrices:

- **Planes/bivectors**: low-rank implicit `C` via plane basis (current default).
- **N-axis**: `C = C_N ⊗ I_Q`, apply as `C_N @ g` (shape `N x Q`).
- **Q-axis**: `C = I_N ⊗ C_Q`, apply as `g @ C_Q^T`.

Choose the structure that matches the inductive bias and compute budget.

## Package layout

`infodynamics_jax/infodynamics/circulation/`:
- `operator.py`: apply `C v` without dense matrices
- `planes.py`: plane/basis builders (PCA, canonical, etc.)
- `gating.py`: `omega(stats)` parameterisations (no schedules)
- `stage.py`: stage-frozen cache + schedule application
- `stats.py`: data statistics containers (precomputed, pooled)
- `README.md`: user-facing overview
