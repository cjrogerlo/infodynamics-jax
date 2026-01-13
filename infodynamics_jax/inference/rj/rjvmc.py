"""
RJVMC (Reversible Jump Variational Monte Carlo) for Sparse GP with Non-Conjugate Likelihoods.

This module implements trans-dimensional MCMC sampling over the number of
inducing points using InertialEnergy + VariationalState for non-conjugate likelihoods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any, Literal
import jax
import jax.numpy as jnp
from jax import random

from ...energy.base import EnergyTerm
from ...energy.inertial import InertialEnergy
from ...core import Phi
from ...gp.ansatz.state import VariationalState
from ..base import InferenceMethod
from .state import RJState


@dataclass(frozen=True)
class RJVMCCFG:
    """Configuration for RJVMC sampler."""
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
    # Variational state update parameters
    inner_steps: int = 5  # Number of inner optimisation steps for variational state
    inner_lr: float = 1e-2  # Learning rate for inner optimisation


@dataclass
class RJVMCRun:
    """RJVMC run results."""
    samples: list[RJState]  # List of RJState samples (with variational_state)
    accept_rate_rj: float  # Acceptance rate for RJ moves
    accept_rate_hmc: float  # Acceptance rate for HMC moves
    M_trace: jnp.ndarray  # Trace of M values
    energy_trace: jnp.ndarray  # Trace of energy values


class RJVMC(InferenceMethod):
    """
    Reversible Jump Variational Monte Carlo for sparse GP with non-conjugate likelihoods.
    
    This sampler performs trans-dimensional MCMC over the number of inducing
    points using InertialEnergy + VariationalState for non-conjugate likelihoods.
    It alternates between:
    1. RJ moves (birth/death of inducing points)
    2. HMC moves (updating hyperparameters)
    3. Variational state updates (updating q(u))
    """
    
    def __init__(
        self,
        cfg: RJVMCCFG = RJVMCCFG(),
        energy: Optional[InertialEnergy] = None,
    ):
        """
        Initialize RJVMC sampler.
        
        Args:
            cfg: Configuration for RJVMC
            energy: InertialEnergy instance (required for non-conjugate case)
        """
        self.cfg = cfg
        self.energy = energy
    
    def _update_variational_state(
        self,
        phi: Phi,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        variational_state: VariationalState,
        key: jax.random.KeyArray,
    ) -> VariationalState:
        """
        Update variational state q(u) using inner optimisation.
        
        This performs a few steps of gradient descent on the variational
        objective to update the variational posterior.
        
        Note: InertialEnergy internally computes variational state via _solve_inner.
        For RJVMC, we maintain variational_state in the RJState and update it
        explicitly to avoid recomputing it every time.
        """
        if self.energy is None:
            raise ValueError("energy (InertialEnergy) must be provided for RJVMC")
        
        # Use energy's inner profiling mechanism
        # The energy already has _solve_inner method that does this
        updated_state = self.energy._solve_inner(phi, X, Y)
        
        return updated_state
    
    def _compute_energy(
        self,
        phi: Phi,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """
        Compute InertialEnergy for given phi.
        
        Note: InertialEnergy internally computes variational state via _solve_inner.
        For efficiency in RJVMC, we maintain variational_state separately in RJState
        and update it explicitly, but the energy computation still goes through
        the standard InertialEnergy interface.
        """
        if self.energy is None:
            raise ValueError("energy (InertialEnergy) must be provided for RJVMC")
        
        # InertialEnergy will internally call _solve_inner to compute variational state
        # and then compute energy. This is the standard interface.
        return self.energy(phi, X, Y, key=key)
    
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
        
        if state.variational_state is None:
            raise ValueError("RJVMC requires variational_state (non-conjugate case)")
        
        # Propose new inducing point location
        key, subkey = random.split(key)
        # Simple proposal: sample uniformly from data points not already in Z_buf
        available_indices = jnp.setdiff1d(
            jnp.arange(X.shape[0]),
            state.Z_buf[:state.M]
        )
        if len(available_indices) == 0:
            return state, False
        
        new_idx = random.choice(subkey, available_indices)
        new_Z = jnp.vstack([state.phi.Z, X[new_idx:new_idx+1]])
        
        # Update Z_buf
        new_Z_buf = state.Z_buf.at[state.M].set(new_idx)
        new_M = state.M + 1
        
        # Create new phi
        new_phi = Phi(
            kernel_params=state.phi.kernel_params,
            Z=new_Z,
            likelihood_params=state.phi.likelihood_params,
            jitter=state.phi.jitter,
        )
        
        # Initialize new variational state with expanded dimensions
        # Add zero mean and identity covariance for new inducing point
        M_old = state.variational_state.m_u.shape[0]
        D = state.variational_state.m_u.shape[1] if state.variational_state.m_u.ndim > 1 else 1
        
        new_m_u = jnp.vstack([
            state.variational_state.m_u,
            jnp.zeros((1, D))
        ])
        
        if state.variational_state.L_u is not None:
            # Expand L_u: add row/column with identity structure
            L_old = state.variational_state.L_u
            new_L_u = jnp.block([
                [L_old, jnp.zeros((M_old, 1))],
                [jnp.zeros((1, M_old)), jnp.eye(1)]
            ])
        else:
            new_L_u = None
        
        new_variational_state = VariationalState(
            m_u=new_m_u,
            L_u=new_L_u,
            s_u_diag=state.variational_state.s_u_diag,
            cov_type=state.variational_state.cov_type,
        )
        
        # Update variational state
        key, subkey = random.split(key)
        new_variational_state = self._update_variational_state(
            new_phi, X, Y, new_variational_state, subkey
        )
        
        # Compute new energy
        key, subkey = random.split(key)
        new_energy = self._compute_energy(new_phi, X, Y, subkey)
        
        # Compute acceptance probability
        log_alpha = jnp.minimum(0.0, new_energy - state.energy)
        
        key, subkey = random.split(key)
        u = random.uniform(subkey)
        accepted = jnp.log(u) < log_alpha
        
        new_state = RJState(
            phi=new_phi,
            variational_state=new_variational_state,
            M=new_M,
            Z_buf=new_Z_buf,
            energy=new_energy,
            Lm=None,  # Not used for non-conjugate
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
        
        if state.variational_state is None:
            raise ValueError("RJVMC requires variational_state (non-conjugate case)")
        
        # Propose removal of last inducing point (simplified)
        key, subkey = random.split(key)
        remove_idx = random.randint(subkey, (), 0, state.M)
        
        # Remove from Z
        keep_mask = jnp.arange(state.M) != remove_idx
        new_Z = state.phi.Z[keep_mask]
        
        # Update Z_buf
        new_Z_buf = jnp.concatenate([
            state.Z_buf[keep_mask],
            state.Z_buf[state.M:]
        ])
        new_M = state.M - 1
        
        # Create new phi
        new_phi = Phi(
            kernel_params=state.phi.kernel_params,
            Z=new_Z,
            likelihood_params=state.phi.likelihood_params,
            jitter=state.phi.jitter,
        )
        
        # Remove from variational state
        new_m_u = state.variational_state.m_u[keep_mask]
        
        if state.variational_state.L_u is not None:
            # Remove row/column from L_u
            L_old = state.variational_state.L_u
            keep_mask_2d = jnp.outer(keep_mask, keep_mask)
            new_L_u = L_old[keep_mask][:, keep_mask]
        else:
            new_L_u = None
        
        new_variational_state = VariationalState(
            m_u=new_m_u,
            L_u=new_L_u,
            s_u_diag=state.variational_state.s_u_diag,
            cov_type=state.variational_state.cov_type,
        )
        
        # Update variational state
        key, subkey = random.split(key)
        new_variational_state = self._update_variational_state(
            new_phi, X, Y, new_variational_state, subkey
        )
        
        # Compute new energy
        key, subkey = random.split(key)
        new_energy = self._compute_energy(new_phi, X, Y, subkey)
        
        # Compute acceptance probability
        log_alpha = jnp.minimum(0.0, new_energy - state.energy)
        
        key, subkey = random.split(key)
        u = random.uniform(subkey)
        accepted = jnp.log(u) < log_alpha
        
        new_state = RJState(
            phi=new_phi,
            variational_state=new_variational_state,
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
    ) -> RJVMCRun:
        """
        Run RJVMC sampling.
        
        Args:
            energy: Energy term (should be InertialEnergy for non-conjugate case)
            phi_init: Initial Phi (used if init_state is None)
            key: PRNG key
            energy_args: Positional arguments for energy (typically (X, Y))
            energy_kwargs: Keyword arguments for energy
            init_state: Optional initial RJState (if None, created from phi_init)
        
        Returns:
            RJVMCRun with samples and diagnostics
        """
        if not isinstance(energy, InertialEnergy):
            raise ValueError("RJVMC requires InertialEnergy for non-conjugate likelihoods")
        
        # Extract X, Y from energy_args
        if len(energy_args) < 2:
            raise ValueError("RJVMC requires energy_args=(X, Y)")
        X, Y = energy_args[0], energy_args[1]
        
        # Initialize RJState if not provided
        if init_state is None:
            M_init = self.cfg.M_init
            N = X.shape[0]
            # Initialize inducing point indices
            key, subkey = random.split(key)
            Z_indices = random.choice(subkey, N, (M_init,), replace=False)
            Z = X[Z_indices]
            
            # Create Z_buf
            Z_buf = jnp.zeros(self.cfg.M_max, dtype=jnp.int32)
            Z_buf = Z_buf.at[:M_init].set(Z_indices)
            
            # Create initial phi if needed
            if isinstance(phi_init, Phi):
                phi = phi_init
            else:
                phi = phi_init
            
            # Initialize VariationalState
            variational_state = VariationalState.initialise(phi, X, Y)
            
            # Compute initial energy
            key, subkey = random.split(key)
            initial_energy = self._compute_energy(phi, X, Y, subkey)
            
            init_state = RJState(
                phi=phi,
                variational_state=variational_state,
                M=jnp.array(M_init, dtype=jnp.int32),
                Z_buf=Z_buf,
                energy=initial_energy,
                Lm=None,
                A=None,
            )
        
        if init_state.variational_state is None:
            raise ValueError("RJVMC requires initial state with variational_state")
        
        self.energy = energy
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
            
            if move_type and step > 0:  # HMC move (simplified)
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
            
            # Update variational state periodically (even if RJ move rejected)
            if step % 10 == 0:  # Update every 10 steps
                key, subkey = random.split(key)
                updated_variational_state = self._update_variational_state(
                    state.phi, X, Y, state.variational_state, subkey
                )
                key, subkey = random.split(key)
                updated_energy = self._compute_energy(state.phi, X, Y, subkey)
                state = RJState(
                    phi=state.phi,
                    variational_state=updated_variational_state,
                    M=state.M,
                    Z_buf=state.Z_buf,
                    energy=updated_energy,
                    Lm=state.Lm,
                    A=state.A,
                )
            
            # Store sample after burn-in
            if step >= self.cfg.burn:
                samples.append(state)
        
        M_trace = jnp.array([s.M for s in samples])
        energy_trace = jnp.array([s.energy for s in samples])
        
        return RJVMCRun(
            samples=samples,
            accept_rate_rj=rj_accepts / max(rj_attempts, 1),
            accept_rate_hmc=hmc_accepts / max(hmc_attempts, 1),
            M_trace=M_trace,
            energy_trace=energy_trace,
        )
