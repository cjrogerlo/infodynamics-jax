# Contributing to the Energy Layer

This document describes what is allowed and what is forbidden when modifying
the energy layer.

---

## Allowed

- Adding new likelihoods with a method:
      neg_loglik_1d(y, f, phi)
- Adding new estimators for:
      E_{q(f_i|phi)}[-log p(y_i|f_i,phi)]
- Adding alternative Rao–Blackwellised constructions

---

## Forbidden

The following are not allowed:

- Introducing standalone modules for:
    - MC sampling
    - q(f) or q(f_i) marginals
- Defining or exposing:
    - free_energy
    - log_evidence
    - marginal_likelihood
    - ELBO as a final objective
- Omitting phi from likelihood or energy interfaces

---

## Design Rule of Thumb

If a quantity:
- is not part of the inference state,
- is not directly optimised or sampled,
- only exists to compute the inertial energy,

then it must live *inside* the energy layer and must not be exposed as a
standalone module.

When in doubt: collapse it.

