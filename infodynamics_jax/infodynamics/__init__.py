# infodynamics_jax/infodynamics/__init__.py
"""
Infodynamics layer.

This package provides *execution-level composition* of:
  - an energy functional (energy.*),
  - an inference dynamics (inference.*),
  - and data / randomness.

It deliberately contains no model assumptions and no inference logic.
The runner orchestrates the composition but does not define either
the energy landscape or the dynamics operators.

Design principle:
  Infodynamics = energy landscape + inference dynamics composition law
  - energy/ defines the landscape
  - inference/ defines the dynamics
  - infodynamics/ defines the composition / execution

Submodules:
  - likelihoods: Gradient-domain Gaussian likelihoods with separable covariance
  - circulation: Curl transforms and beta schedules
"""
from .runner import RunCFG, RunOut, run
from .hyperprior import (
    make_hyperprior,
    kernel_l2_hyperprior,
    kernel_log_l2_hyperprior,
    z_l2_hyperprior,
    likelihood_l2_hyperprior,
    likelihood_log_l2_hyperprior,
)
from .circulation import (
    PCAPlanes,
    apply_skew_from_planes,
    beta_schedule,
    flatten_latent,
    unflatten_latent,
    make_curl_transform,
)
from . import likelihoods

__all__ = [
    "RunCFG", "RunOut", "run",
    "make_hyperprior",
    "kernel_l2_hyperprior",
    "kernel_log_l2_hyperprior",
    "z_l2_hyperprior",
    "likelihood_l2_hyperprior",
    "likelihood_log_l2_hyperprior",
    "PCAPlanes",
    "apply_skew_from_planes",
    "beta_schedule",
    "flatten_latent",
    "unflatten_latent",
    "make_curl_transform",
    "likelihoods",
]
