# infodynamics_jax/inference/optimisation/vga.py
"""
Variational Gaussian Approximation (VGA).

VGA performs type-II maximum likelihood by optimising the VFE objective
using gradient descent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_leaves

from ...energy.base import EnergyTerm
from ..base import InferenceMethod


@dataclass(frozen=True)
class VGACFG:
    """Configuration for VGA."""
    steps: int = 200
    lr: float = 1e-2
    clip_grad_norm: Optional[float] = None
    jit: bool = True


@dataclass
class VGARun:
    """VGA run results."""
    phi: Any
    energy_trace: jnp.ndarray  # shape [steps]
    grad_norm_trace: jnp.ndarray  # shape [steps]

    @property
    def loss(self) -> jnp.ndarray:
        """Backward-compatible alias for energy_trace."""
        return self.energy_trace


class VGA(InferenceMethod):
    """
    Variational Gaussian Approximation.
    
    Performs type-II ML by optimising the VFE objective using gradient descent.
    """
    
    def __init__(self, cfg: VGACFG = VGACFG()):
        self.cfg = cfg

    def run(
        self,
        energy: EnergyTerm | None,
        phi_init,
        *args,
        key=None,
        energy_kwargs=None,
        objective: EnergyTerm | None = None,
        energy_args=None,
        energy_args_kw=None,
    ) -> VGARun:
        """
        Run VGA optimisation.
        
        Args:
            energy: Energy term (typically VFE objective)
            phi_init: Initial parameter state
            key: PRNG key (optional, for stochastic energy)
            energy_args: Additional arguments for energy
            energy_kwargs: Additional keyword arguments for energy
            objective: Backward-compatible alias for energy
        
        Returns:
            VGARun with optimised phi, energy trace, and grad norm trace
        """
        if objective is not None:
            if energy is not None and callable(energy):
                raise TypeError("Provide only one of energy or objective.")
            if energy is not None and not callable(energy):
                key = energy
            energy = objective

        if energy is None:
            raise TypeError("Missing required energy/objective function.")

        if args:
            if energy_args is not None or energy_args_kw is not None:
                raise TypeError("Provide energy_args positionally or via energy_args/energy_args_kw, not both.")
            energy_args = args

        if energy_args_kw is not None and energy_args is not None:
            raise TypeError("Provide only one of energy_args or energy_args_kw.")
        if energy_args is None:
            if energy_args_kw is not None:
                energy_args = tuple(energy_args_kw)
            else:
                energy_args = ()
        else:
            energy_args = tuple(energy_args)

        if energy_kwargs is None:
            energy_kwargs = {}

        steps = self.cfg.steps
        lr = self.cfg.lr
        clip_grad_norm = self.cfg.clip_grad_norm
        jit = self.cfg.jit

        def energy_fn(phi):
            if key is None:
                return energy(phi, *energy_args, **energy_kwargs)
            else:
                # include key only if key is not None
                return energy(phi, *energy_args, key=key, **energy_kwargs)

        value_and_grad_fn = jax.value_and_grad(energy_fn)

        def global_grad_norm(grad):
            sq_sum = 0.0
            for leaf in tree_leaves(grad):
                sq_sum += jnp.sum(leaf ** 2)
            return jnp.sqrt(sq_sum)

        def clip_grads(grad):
            if clip_grad_norm is None:
                return grad
            norm = global_grad_norm(grad)
            factor = jnp.minimum(1.0, clip_grad_norm / (norm + 1e-16))
            return tree_map(lambda g: g * factor, grad)

        def step(carry, _):
            phi, energy_trace, grad_norm_trace, i = carry
            val, grad = value_and_grad_fn(phi)
            grad = clip_grads(grad)
            grad_norm = global_grad_norm(grad)
            phi = tree_map(lambda p, g: p - lr * g, phi, grad)
            energy_trace = energy_trace.at[i].set(val)
            grad_norm_trace = grad_norm_trace.at[i].set(grad_norm)
            return (phi, energy_trace, grad_norm_trace, i + 1), None

        energy_trace = jnp.zeros(steps)
        grad_norm_trace = jnp.zeros(steps)

        if jit:
            (phi_final, energy_trace, grad_norm_trace, _), _ = lax.scan(
                step, (phi_init, energy_trace, grad_norm_trace, 0), None, length=steps
            )
        else:
            phi = phi_init
            etrace = energy_trace
            gtrace = grad_norm_trace
            i = 0
            for _ in range(steps):
                val, grad = value_and_grad_fn(phi)
                grad = clip_grads(grad)
                grad_norm = global_grad_norm(grad)
                phi = tree_map(lambda p, g: p - lr * g, phi, grad)
                etrace = etrace.at[i].set(val)
                gtrace = gtrace.at[i].set(grad_norm)
                i += 1
            phi_final = phi
            energy_trace = etrace
            grad_norm_trace = gtrace

        return VGARun(phi=phi_final, energy_trace=energy_trace, grad_norm_trace=grad_norm_trace)
