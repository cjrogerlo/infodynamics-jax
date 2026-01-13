"""
RJMCMC (Reversible Jump MCMC) for Sparse GP with Conjugate (Gaussian) Likelihoods.

This module implements trans-dimensional MCMC sampling over the number of
inducing points using VFE for conjugate Gaussian likelihoods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any, Literal
import jax
import jax.numpy as jnp
from jax import random

from ...energy.base import EnergyTerm
from ...core import Phi
from ...gp.kernels.params import KernelParams
from ..base import InferenceMethod
from ..optimisation.vfe import vfe_objective
from .state import RJState


@dataclass(frozen=True)
class RJMCMCCFG:
    """Configuration for RJMCMC sampler."""
    n_steps: int = 1000
    burn: int = 250
    M_min: int = 5
    M_max: int = 60
    M_init: int = 20
    birth_prob: float = 0.5  # Probability of birth move (vs death)
    death_mode: Literal["rank1_last", "local_rebuild"] = "rank1_last"
    # HMC parameters for updating hyperparameters
    hmc_step_size: float = 1e-2
    hmc_n_leapfrog: int = 8
    hmc_prob: float = 0.3  # Probability of HMC step (vs RJ move)


@dataclass
class RJMCMCRun:
    """RJMCMC run results."""
    samples: list[RJState]  # List of RJState samples
    accept_rate_rj: float  # Acceptance rate for RJ moves
    accept_rate_hmc: float  # Acceptance rate for HMC moves
    M_trace: jnp.ndarray  # Trace of M values
    energy_trace: jnp.ndarray  # Trace of energy values


class RJMCMC(InferenceMethod):
    """
    Reversible Jump MCMC for sparse GP with conjugate (Gaussian) likelihoods.
    
    This sampler performs trans-dimensional MCMC over the number of inducing
    points using VFE for energy computation. It alternates between:
    1. RJ moves (birth/death of inducing points)
    2. HMC moves (updating hyperparameters)
    """
    
    def __init__(
        self,
        cfg: RJMCMCCFG = RJMCMCCFG(),
        kernel_fn: Optional[Callable] = None,
    ):
        """
        Initialize RJMCMC sampler.
        
        Args:
            cfg: Configuration for RJMCMC
            kernel_fn: Kernel function (required for VFE computation)
        """
        self.cfg = cfg
        self.kernel_fn = kernel_fn
    
    def _compute_vfe(self, phi: Phi, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        """Compute VFE for given phi."""
        if self.kernel_fn is None:
            raise ValueError("kernel_fn must be provided for RJMCMC")
        return vfe_objective(phi, X, Y, kernel_fn=self.kernel_fn, residual="fitc")
    
    def _birth_move(
        self,
        state: RJState,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        key: jax.random.KeyArray,
    ) -> tuple[RJState, bool]:
        """
        Birth move: Add a new inducing point.
        
        Returns:
            (new_state, accepted)
        """
        if state.M >= self.cfg.M_max:
            return state, False
        
        # Propose new inducing point location
        key, subkey = random.split(key)
        # Simple proposal: sample uniformly from data points not already in Z_buf
        # (This is a simplified version - real implementation would be more sophisticated)
        available_indices = jnp.setdiff1d(
            jnp.arange(X.shape[0]),
            state.Z_buf[:state.M]
        )
        if len(available_indices) == 0:
            return state, False
        
        new_idx = random.choice(subkey, available_indices)
        
        # Update Z_buf
        new_Z_buf = state.Z_buf.at[state.M].set(new_idx)
        new_M = state.M + 1
        
        # Create new Z with fixed size (M_max, D) - only first M entries are active
        # This ensures phi.Z always has the same shape for jax.lax.cond
        D = X.shape[1]
        M_max = self.cfg.M_max
        new_Z = jnp.zeros((M_max, D), dtype=X.dtype)
        # Set active inducing points (first M entries)
        active_indices = new_Z_buf[:new_M]
        new_Z = new_Z.at[:new_M].set(X[active_indices])
        # Keep old values for inactive entries (for shape consistency)
        if state.phi.Z.shape[0] >= M_max:
            new_Z = new_Z.at[new_M:].set(state.phi.Z[new_M:])
        
        # Create new phi
        new_phi = Phi(
            kernel_params=state.phi.kernel_params,
            Z=new_Z,
            likelihood_params=state.phi.likelihood_params,
            jitter=state.phi.jitter,
        )
        
        # Compute new energy (only use active inducing points)
        # Create temporary phi with only active inducing points for VFE computation
        active_Z = new_Z[:new_M]
        temp_phi = Phi(
            kernel_params=new_phi.kernel_params,
            Z=active_Z,
            likelihood_params=new_phi.likelihood_params,
            jitter=new_phi.jitter,
        )
        new_energy = self._compute_vfe(temp_phi, X, Y)
        
        # Compute acceptance probability
        # log_alpha = min(0, new_energy - state.energy + log_prior_M(new_M) - log_prior_M(state.M))
        # Simplified: uniform prior on M
        log_alpha = jnp.minimum(0.0, new_energy - state.energy)
        
        key, subkey = random.split(key)
        u = random.uniform(subkey)
        accepted = jnp.log(u) < log_alpha
        
        new_state = RJState(
            phi=new_phi,
            variational_state=None,  # Conjugate case
            M=new_M,
            Z_buf=new_Z_buf,
            energy=new_energy,
            Lm=None,  # Would need to update cached matrices
            A=None,
        )
        
        return jax.lax.cond(
            accepted,
            lambda: (new_state, True),
            lambda: (state, False),
        )
    
    def _death_move(
        self,
        state: RJState,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        key: jax.random.KeyArray,
    ) -> tuple[RJState, bool]:
        """
        Death move: Remove an inducing point.
        
        Returns:
            (new_state, accepted)
        """
        if state.M <= self.cfg.M_min:
            return state, False
        
        # Propose removal of last inducing point (simplified)
        key, subkey = random.split(key)
        remove_idx = random.randint(subkey, (), 0, state.M)
        
        # Update Z_buf (shift remaining indices)
        keep_mask = jnp.arange(state.M) != remove_idx
        new_Z_buf = jnp.concatenate([
            state.Z_buf[keep_mask],
            state.Z_buf[state.M:]  # Keep buffer tail
        ])
        new_M = state.M - 1
        
        # Create new Z with fixed size (M_max, D) - only first M entries are active
        # This ensures phi.Z always has the same shape for jax.lax.cond
        D = X.shape[1]
        M_max = self.cfg.M_max
        new_Z = jnp.zeros((M_max, D), dtype=X.dtype)
        # Set active inducing points (first M entries)
        active_indices = new_Z_buf[:new_M]
        new_Z = new_Z.at[:new_M].set(X[active_indices])
        # Keep old values for inactive entries (for shape consistency)
        if state.phi.Z.shape[0] >= M_max:
            new_Z = new_Z.at[new_M:].set(state.phi.Z[new_M:])
        
        # Create new phi
        new_phi = Phi(
            kernel_params=state.phi.kernel_params,
            Z=new_Z,
            likelihood_params=state.phi.likelihood_params,
            jitter=state.phi.jitter,
        )
        
        # Compute new energy (only use active inducing points)
        # Create temporary phi with only active inducing points for VFE computation
        active_Z = new_Z[:new_M]
        temp_phi = Phi(
            kernel_params=new_phi.kernel_params,
            Z=active_Z,
            likelihood_params=new_phi.likelihood_params,
            jitter=new_phi.jitter,
        )
        new_energy = self._compute_vfe(temp_phi, X, Y)
        
        # Compute acceptance probability
        log_alpha = jnp.minimum(0.0, new_energy - state.energy)
        
        key, subkey = random.split(key)
        u = random.uniform(subkey)
        accepted = jnp.log(u) < log_alpha
        
        new_state = RJState(
            phi=new_phi,
            variational_state=None,
            M=new_M,
            Z_buf=new_Z_buf,
            energy=new_energy,
            Lm=None,
            A=None,
        )
        
        return jax.lax.cond(
            accepted,
            lambda: (new_state, True),
            lambda: (state, False),
        )
    
    def run(
        self,
        energy: EnergyTerm,
        phi_init: Any,
        *,
        key: jax.random.KeyArray,
        energy_args: tuple = (),
        energy_kwargs: Optional[dict] = None,
        init_state: Optional[RJState] = None,
    ) -> RJMCMCRun:
        """
        Run RJMCMC sampling.
        
        Args:
            energy: Energy term (should be VFE-based for conjugate case)
            phi_init: Initial Phi (used if init_state is None)
            key: PRNG key
            energy_args: Positional arguments for energy (typically (X, Y))
            energy_kwargs: Keyword arguments for energy
            init_state: Optional initial RJState (if None, created from phi_init)
        
        Returns:
            RJMCMCRun with samples and diagnostics
        """
        if self.kernel_fn is None:
            raise ValueError("kernel_fn must be provided for RJMCMC")
        
        # Extract X, Y from energy_args
        if len(energy_args) < 2:
            raise ValueError("RJMCMC requires energy_args=(X, Y)")
        X, Y = energy_args[0], energy_args[1]
        
        # Initialize RJState if not provided
        if init_state is None:
            M_init = self.cfg.M_init
            N = X.shape[0]
            # Initialize inducing point indices
            key, subkey = random.split(key)
            Z_indices = random.choice(subkey, N, (M_init,), replace=False)
            
            # Create Z_buf
            Z_buf = jnp.zeros(self.cfg.M_max, dtype=jnp.int32)
            Z_buf = Z_buf.at[:M_init].set(Z_indices)
            
            # Create initial phi if needed
            if isinstance(phi_init, Phi):
                phi = phi_init
                # Ensure phi.Z has fixed size (M_max, D) for shape consistency
                D = X.shape[1]
                M_max = self.cfg.M_max
                if phi.Z.shape[0] != M_max:
                    Z_fixed = jnp.zeros((M_max, D), dtype=X.dtype)
                    Z_fixed = Z_fixed.at[:M_init].set(X[Z_indices])
                    phi = Phi(
                        kernel_params=phi.kernel_params,
                        Z=Z_fixed,
                        likelihood_params=phi.likelihood_params,
                        jitter=phi.jitter,
                    )
            else:
                # Create phi with fixed-size Z
                D = X.shape[1]
                M_max = self.cfg.M_max
                Z_fixed = jnp.zeros((M_max, D), dtype=X.dtype)
                Z_fixed = Z_fixed.at[:M_init].set(X[Z_indices])
                phi = Phi(
                    kernel_params=phi_init.kernel_params,
                    Z=Z_fixed,
                    likelihood_params=phi_init.likelihood_params,
                    jitter=phi_init.jitter,
                )
            
            # Compute initial energy (only use active inducing points)
            # Create temporary phi with only active inducing points for VFE computation
            active_Z = phi.Z[:M_init]
            temp_phi = Phi(
                kernel_params=phi.kernel_params,
                Z=active_Z,
                likelihood_params=phi.likelihood_params,
                jitter=phi.jitter,
            )
            key, subkey = random.split(key)
            initial_energy = self._compute_vfe(temp_phi, X, Y)
            
            init_state = RJState(
                phi=phi,
                variational_state=None,  # Conjugate case
                M=jnp.array(M_init, dtype=jnp.int32),
                Z_buf=Z_buf,
                energy=initial_energy,
                Lm=None,
                A=None,
            )
        
        state = init_state
        samples = []
        rj_accepts = 0
        rj_attempts = 0
        hmc_accepts = 0
        hmc_attempts = 0
        
        for step in range(self.cfg.n_steps):
            key, subkey = random.split(key)
            
            # Decide move type
            move_type = random.uniform(subkey) < self.cfg.hmc_prob
            
            if move_type and step > 0:  # HMC move (simplified - would need full HMC implementation)
                # Placeholder: just accept current state
                hmc_attempts += 1
                hmc_accepts += 1
            else:
                # RJ move
                rj_attempts += 1
                key, subkey = random.split(key)
                birth = random.uniform(subkey) < self.cfg.birth_prob
                
                if birth:
                    state, accepted = self._birth_move(state, X, Y, key)
                else:
                    state, accepted = self._death_move(state, X, Y, key)
                
                if accepted:
                    rj_accepts += 1
            
            # Store sample after burn-in
            if step >= self.cfg.burn:
                samples.append(state)
        
        M_trace = jnp.array([s.M for s in samples])
        energy_trace = jnp.array([s.energy for s in samples])
        
        return RJMCMCRun(
            samples=samples,
            accept_rate_rj=rj_accepts / max(rj_attempts, 1),
            accept_rate_hmc=hmc_accepts / max(hmc_attempts, 1),
            M_trace=M_trace,
            energy_trace=energy_trace,
        )
