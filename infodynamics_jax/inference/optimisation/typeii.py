# infodynamics_jax/inference/optimisation/typeii.py
"""
Type-II optimisation (ML-II and MAP-II).

This module provides a general-purpose Type-II optimiser that can be used for:
- **ML-II (Maximum Likelihood Type-II)**: Optimize VFE objective (no hyperprior)
- **MAP-II (Maximum A Posteriori Type-II)**: Optimize VFE + hyperprior

The optimiser finds hyperparameters that minimize the energy function:
    phi* = argmin_phi E(phi)

where E(phi) can be:
- VFE objective (for ML-II or MAP-II)
- InertialEnergy (for Bayesian inference)
- TargetEnergy (with prior terms)

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
from ...core.phi import Phi
from ...gp.kernels.params import KernelParams


@dataclass(frozen=True)
class TypeIICFG:
    """Configuration for Type-II optimisation (ML-II or MAP-II)."""
    steps: int = 200
    lr: float = 1e-2
    optimizer: Literal["sgd", "adam", "rmsprop"] = "adam"
    clip_grad_norm: Optional[float] = None
    jit: bool = True
    # Parameter constraints
    constrain_params: bool = True  # Apply constraints to ensure positive parameters
    min_lengthscale: float = 1e-6  # Minimum value for lengthscale
    min_variance: float = 1e-6  # Minimum value for variance
    min_noise_var: float = 1e-3  # Minimum value for noise_var in likelihood_params
    # Increased from 1e-6 to 1e-3 to prevent noise variance collapse


@dataclass
class TypeIIRun:
    """Type-II run results."""
    phi: Any
    energy_trace: jnp.ndarray  # shape [steps]
    grad_norm_trace: jnp.ndarray  # shape [steps]


class TypeII(InferenceMethod):
    """
    Type-II optimiser (for ML-II and MAP-II).
    
    This optimiser finds hyperparameters that minimize the energy function:
        phi* = argmin_phi E(phi)
    
    Can be used for:
    - **ML-II**: Optimize VFE objective (no hyperprior)
    - **MAP-II**: Optimize VFE + hyperprior (pass hyperprior to run())
    
    This can be used with any EnergyTerm, including:
    - VFE objective (for type-II inference)
    - InertialEnergy (data-dependent energy)
    - TargetEnergy (with prior terms)
    
    The optimiser supports different optimisers (SGD, Adam, RMSprop) and
    can work with or without stochastic energy terms.
    
    Examples:
        >>> # ML-II: Optimize VFE only
        >>> from infodynamics_jax.inference.optimisation.vfe import make_vfe_objective
        >>> vfe_obj = make_vfe_objective(kernel_fn)
        >>> method = TypeII(TypeIICFG(steps=200, lr=1e-2))
        >>> result = method.run(vfe_obj, phi_init, energy_args=(X, Y))
        >>> # No hyperprior → ML-II
        >>> 
        >>> # MAP-II: Optimize VFE + hyperprior
        >>> from infodynamics_jax.infodynamics import make_hyperprior
        >>> hyperprior = make_hyperprior(...)
        >>> # Pass hyperprior to run() via infodynamics.run(..., hyperprior=...)
    """
    
    def __init__(self, cfg: TypeIICFG = TypeIICFG()):
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

    def _apply_constraints(self, phi: Phi) -> Phi:
        """
        Apply parameter constraints to ensure positive values.
        
        This ensures:
        - lengthscale >= min_lengthscale
        - variance >= min_variance
        - noise_var >= min_noise_var (if present in likelihood_params)
        
        Args:
            phi: Parameter state to constrain
            
        Returns:
            Constrained parameter state
        """
        if not self.cfg.constrain_params:
            return phi
        
        # Constrain kernel parameters
        lengthscale = jnp.maximum(phi.kernel_params.lengthscale, self.cfg.min_lengthscale)
        variance = jnp.maximum(phi.kernel_params.variance, self.cfg.min_variance)
        
        # Preserve other kernel parameters
        kernel_params = KernelParams(
            lengthscale=lengthscale,
            variance=variance,
            nu=phi.kernel_params.nu,
            alpha=phi.kernel_params.alpha,
            period=phi.kernel_params.period,
            offset=phi.kernel_params.offset,
            degree=phi.kernel_params.degree,
        )
        
        # Constrain likelihood parameters
        # Use dict comprehension to preserve structure
        likelihood_params = {
            key: jnp.maximum(value, self.cfg.min_noise_var) if key == "noise_var" else value
            for key, value in phi.likelihood_params.items()
        }
        
        return Phi(
            kernel_params=kernel_params,
            Z=phi.Z,
            likelihood_params=likelihood_params,
            jitter=phi.jitter,
        )

    def run(
        self, 
        energy: EnergyTerm, 
        phi_init, 
        *, 
        key=None, 
        energy_args=(), 
        energy_kwargs=None
    ) -> TypeIIRun:
        """
        Run Type-II optimisation (ML-II or MAP-II).
        
        Args:
            energy: Energy term to minimize (can be any EnergyTerm)
            phi_init: Initial parameter state
            key: PRNG key (optional, for stochastic energy)
            energy_args: Additional arguments for energy
            energy_kwargs: Additional keyword arguments for energy
        
        Note:
            - For **ML-II**: Pass VFE objective without hyperprior
            - For **MAP-II**: Pass VFE objective and add hyperprior via 
              `infodynamics.run(..., hyperprior=...)` or through TargetEnergy.extra
        
        Returns:
            TypeIIRun with optimised phi, energy trace, and grad norm trace
        
        Examples:
            >>> # ML-II: Using with VFE objective
            >>> from infodynamics_jax.inference.optimisation.vfe import make_vfe_objective
            >>> vfe_obj = make_vfe_objective(kernel_fn)
            >>> method = TypeII(TypeIICFG(steps=200, lr=1e-2))
            >>> result = method.run(vfe_obj, phi_init, energy_args=(X, Y))
            >>> 
            >>> # MAP-II: Using with VFE + hyperprior (via run())
            >>> from infodynamics_jax.infodynamics import run, make_hyperprior
            >>> hyperprior = make_hyperprior(...)
            >>> result = run(method=method, energy=vfe_obj, ..., hyperprior=hyperprior)
        """
        if energy_kwargs is None:
            energy_kwargs = {}

        cfg = self.cfg
        steps = cfg.steps
        lr = cfg.lr
        clip_grad_norm = cfg.clip_grad_norm
        jit = cfg.jit

        # Create energy function with constraints applied
        def energy_fn(phi):
            # Apply constraints before computing energy (so gradients are computed on constrained params)
            phi_constrained = self._apply_constraints(phi)
            if key is None:
                return energy(phi_constrained, *energy_args, **energy_kwargs)
            else:
                # For stochastic energy, we need to split the key each time
                # This is a limitation: we can't use the same key for all steps
                # In practice, users should pass a key that gets split appropriately
                return energy(phi_constrained, *energy_args, key=key, **energy_kwargs)

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
            
            # Check for NaN/Inf in energy or gradients
            val = jnp.where(jnp.isfinite(val), val, jnp.inf)
            grad = jax.tree_util.tree_map(
                lambda g: jnp.where(jnp.isfinite(g), g, 0.0),
                grad
            )
            
            grad = clip_grads(grad)
            grad_norm = global_grad_norm(grad)
            
            # Update using optimizer
            updates, opt_state = optimizer.update(grad, opt_state, params=phi)
            phi = optax.apply_updates(phi, updates)
            
            # Note: Constraints are applied inside energy_fn, so phi may have unconstrained values
            # but energy and gradients are computed on constrained values
            # We still apply constraints here to keep phi in valid range
            phi = self._apply_constraints(phi)
            
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
                
                # Note: Constraints are applied inside energy_fn, so phi may have unconstrained values
                # but energy and gradients are computed on constrained values
                # We still apply constraints here to keep phi in valid range
                phi = self._apply_constraints(phi)
                
                etrace = etrace.at[i].set(val)
                gtrace = gtrace.at[i].set(grad_norm)
                i += 1
            phi_final = phi
            energy_trace = etrace
            grad_norm_trace = gtrace

        # Apply constraints to final result
        phi_final = self._apply_constraints(phi_final)

        return TypeIIRun(phi=phi_final, energy_trace=energy_trace, grad_norm_trace=grad_norm_trace)
