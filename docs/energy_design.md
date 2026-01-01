# Energy Layer Design

This library treats **inertial energy** as the primitive object of inference.

We do **not** perform marginal likelihood estimation, free-energy estimation,
or evidence-based model comparison.

---

## Definition

The only energy quantity defined in this library is

    E(phi) = E_{q(f | phi)}[ -log p(y | f, phi) ]

where:
- phi denotes all structural parameters (kernel, inducing inputs, likelihood
  hyperparameters, etc.);
- q(f | phi) is an inner equilibrium distribution induced by q(u | phi);
- the expectation is taken **inside** the logarithm.

This quantity is referred to as **inertial energy** throughout the codebase.

---

## What Energy Is NOT

The following quantities are intentionally *not* represented:

- marginal likelihood
- log evidence
- free energy of the form  -log E[p(y | f, phi)]
- ELBO as a terminal objective

These objects are incompatible with the inference-as-dynamics perspective
adopted in this framework.

---

## Rao–Blackwellisation Principle

Although q(f | phi) does not factorise across data points after integrating
out inducing variables, each one-dimensional marginal q(f_i | phi) is
tractable.

All non-conjugacy is localised to scalar expectations under q(f_i | phi),
while global coupling is retained in inducing space.

This Rao–Blackwellisation step is mandatory for non-conjugate likelihoods.

---

## Estimators

Different estimators correspond to different *evaluators* of the same energy:

- conjugate: closed-form Gaussian expectation
- gh: Gauss–Hermite quadrature
- mc: Monte Carlo estimator

They do not define different objectives.

---

## Invariants

The following invariants must hold everywhere in the codebase:

1. All likelihood evaluations explicitly depend on phi.
2. Energy is always a scalar functional of phi (and optional inner state).
3. Intermediate objects such as q(f_i | phi) are not exposed outside the
   energy layer.

