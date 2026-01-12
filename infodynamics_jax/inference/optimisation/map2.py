# infodynamics_jax/inference/optimisation/map2.py
"""
Maximum A Posteriori Type-II (MAP-II) optimisation.

MAP-II optimises the energy function (or negative log-likelihood) with respect to
hyperparameters phi, finding the point estimate that maximizes the posterior:

    phi* = argmax_phi p(phi | y) = argmin_phi E(phi)

where E(phi) is the energy function (typically negative log-likelihood plus prior).

This is a general-purpose optimiser that can use different optimisers (SGD, Adam, etc.)
and can work with any EnergyTerm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.tree_util import tree_map, tree_leaves

from ...energy.base import EnergyTerm
from ..base import InferenceMethod


@dataclass(frozen=True)
class MAP2CFG:
    """Configuration for MAP-II optimisation."""
    steps: int = 200
    lr: float = 1e-2
    optimizer: Literal["sgd", "adam", "rmsprop"] = "adam"
    clip_grad_norm: Optional[float] = None
    jit: bool = True


@dataclass
class MAP2Run:
    """MAP-II run results."""
    phi: Any
    energy_trace: jnp.ndarray  # shape [steps]
    grad_norm_trace: jnp.ndarray  # shape [steps]


class MAP2(InferenceMethod):
    """
    Maximum A Posteriori Type-II optimiser.
    
    MAP-II finds the hyperparameters that minimize the energy function:
        phi* = argmin_phi E(phi)
    
    This can be used with any EnergyTerm, including:
    - InertialEnergy (data-dependent energy)
    - TargetEnergy (with prior terms)
    - VFE objective (for variational inference)
    
    The optimiser supports different optimisers (SGD, Adam, RMSprop) and
    can work with or without stochastic energy terms.
    """
    
    def __init__(self, cfg: MAP2CFG = MAP2CFG()):
        self.cfg = cfg

    def _get_optimizer(self, lr: float):
        """Get optimizer based on configuration."""
        if self.cfg.optimizer == "sgd":
            return optax.sgd(lr)
        elif self.cfg.optimizer == "adam":
            return optax.adam(lr)
        elif self.cfg.optimizer == "rmsprop":
            return optax.rmsprop(lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    def run(
        self, 
        energy: EnergyTerm, 
        phi_init, 
        *, 
        key=None, 
        energy_args=(), 
        energy_kwargs=None
    ) -> MAP2Run:
        """
        Run MAP-II optimisation.
        
        Args:
            energy: Energy term to minimize (can be any EnergyTerm)
            phi_init: Initial parameter state
            key: PRNG key (optional, for stochastic energy)
            energy_args: Additional arguments for energy
            energy_kwargs: Additional keyword arguments for energy
        
        Note:
            Hyperpriors should be added via infodynamics.run(..., hyperprior=...)
            or through TargetEnergy.extra. This method only optimizes the energy.
        
        Returns:
            MAP2Run with optimised phi, energy trace, and grad norm trace
        
        Examples:
            >>> # Using with InertialEnergy
            >>> from infodynamics_jax.energy import InertialEnergy, TargetEnergy
            >>> inertial = InertialEnergy(...)
            >>> target = TargetEnergy(inertial=inertial, prior=prior)
            >>> map2 = MAP2(MAP2CFG(steps=500, optimizer="adam"))
            >>> result = map2.run(target, phi_init, key=key, energy_args=(X, Y))
            
            >>> # Using with VFE objective
            >>> from infodynamics_jax.inference.optimisation import make_vfe_objective
            >>> vfe_obj = make_vfe_objective(kernel_fn)
            >>> result = map2.run(vfe_obj, phi_init, energy_args=(X, Y))
        """
        if energy_kwargs is None:
            energy_kwargs = {}

        cfg = self.cfg
        steps = cfg.steps
        lr = cfg.lr
        clip_grad_norm = cfg.clip_grad_norm
        jit = cfg.jit

        # Create energy function
        def energy_fn(phi):
            if key is None:
                return energy(phi, *energy_args, **energy_kwargs)
            else:
                # For stochastic energy, we need to split the key each time
                # This is a limitation: we can't use the same key for all steps
                # In practice, users should pass a key that gets split appropriately
                return energy(phi, *energy_args, key=key, **energy_kwargs)

        value_and_grad_fn = jax.value_and_grad(energy_fn)

        # Get optimizer
        optimizer = self._get_optimizer(lr)
        opt_state = optimizer.init(phi_init)

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
            phi, opt_state, energy_trace, grad_norm_trace, i = carry
            val, grad = value_and_grad_fn(phi)
            grad = clip_grads(grad)
            grad_norm = global_grad_norm(grad)
            
            # Update using optimizer
            updates, opt_state = optimizer.update(grad, opt_state, params=phi)
            phi = optax.apply_updates(phi, updates)
            
            energy_trace = energy_trace.at[i].set(val)
            grad_norm_trace = grad_norm_trace.at[i].set(grad_norm)
            return (phi, opt_state, energy_trace, grad_norm_trace, i + 1), None

        energy_trace = jnp.zeros(steps)
        grad_norm_trace = jnp.zeros(steps)

        if jit:
            (phi_final, _, energy_trace, grad_norm_trace, _), _ = lax.scan(
                step, (phi_init, opt_state, energy_trace, grad_norm_trace, 0), None, length=steps
            )
        else:
            phi = phi_init
            opt_state_local = opt_state
            etrace = energy_trace
            gtrace = grad_norm_trace
            i = 0
            for _ in range(steps):
                val, grad = value_and_grad_fn(phi)
                grad = clip_grads(grad)
                grad_norm = global_grad_norm(grad)
                
                # Update using optimizer
                updates, opt_state_local = optimizer.update(grad, opt_state_local, params=phi)
                phi = optax.apply_updates(phi, updates)
                
                etrace = etrace.at[i].set(val)
                gtrace = gtrace.at[i].set(grad_norm)
                i += 1
            phi_final = phi
            energy_trace = etrace
            grad_norm_trace = gtrace

        return MAP2Run(phi=phi_final, energy_trace=energy_trace, grad_norm_trace=grad_norm_trace)
