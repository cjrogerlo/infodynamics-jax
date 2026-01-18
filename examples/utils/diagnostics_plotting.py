import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def plot_ess_kl_accept(smc_out, save_path=None):
    """Plot standard SMC diagnostics: ESS, KL, and Acceptance Rate."""
    # Use actual betas from the trace to match dimensionality
    # betas has n_steps + 1 elements [0, ..., 1.0]
    # traces have n_steps elements
    betas_mid = smc_out['betas'][1:]
    ess = smc_out['ess_trace']
    kl = smc_out['kl_trace']
    accept = smc_out['accept_trace']

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    
    ax[0].plot(betas_mid, ess, marker='.', alpha=0.7)
    ax[0].set_title('Effective Sample Size (ESS)')
    ax[0].set_xlabel(r'$\beta$')
    ax[0].set_ylim(0, max(ess)*1.1)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(betas_mid, kl, marker='.', color='orange', alpha=0.7)
    ax[1].set_title('Incremental KL Divergence')
    ax[1].set_xlabel(r'$\beta$')
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(betas_mid, accept, marker='.', color='green', alpha=0.7)
    ax[2].set_title('Rejuvenation Acceptance')
    ax[2].set_xlabel(r'$\beta$')
    ax[2].set_ylim(0, 1.1)
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()

def plot_response_map(smc_out, save_path=None):
    """Plot Δβ vs β - susceptibility measure of posterior contraction."""
    betas = smc_out['betas']
    delta_betas = np.diff(betas)
    betas_mid = betas[1:]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas_mid, delta_betas, marker='o', markersize=4, color='tab:red')
    ax.set_title(r'Susceptibility: $\Delta\beta_t$ vs $\beta$')
    ax.set_ylabel(r'Step Size $\Delta\beta$ (Inference Speed)')
    ax.set_xlabel(r'Annealing Temperature $\beta$')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Mirror susceptibility: 1/Δβ
    ax2 = ax.twinx()
    ax2.plot(betas_mid, 1.0/delta_betas, alpha=0.0) # Dummy for axis scaling
    ax2.set_ylabel(r'Susceptibility $(1/\Delta\beta)$', color='tab:red', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()

def plot_fisher_alignment(smc_out, save_path=None):
    """Validate Fisher-Rao alignment: Δβ⁻² vs Var(Energy)."""
    betas = smc_out['betas']
    delta_betas = np.diff(betas)
    
    # Energy variance is proportional to Heat Capacity C
    C = smc_out['C_trace']
    inv_delta_beta_sq = 1.0 / (delta_betas**2 + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(C, inv_delta_beta_sq, c=betas[1:], cmap='viridis', alpha=0.8, edgecolor='k', s=40)
    
    # Fit line to check proportionality
    try:
        from scipy.stats import linregress
        mask = (C > 0) & (np.isfinite(inv_delta_beta_sq))
        if np.sum(mask) > 2:
            res = linregress(C[mask], inv_delta_beta_sq[mask])
            x_fit = np.linspace(min(C), max(C), 100)
            ax.plot(x_fit, res.intercept + res.slope * x_fit, 'r--', alpha=0.5, label=f'R²={res.rvalue**2:.3f}')
            ax.legend()
    except ImportError:
        pass

    ax.set_title('Fisher-Rao Geodesic Alignment')
    ax.set_xlabel(r'Energy Variance $\propto C(\beta)$')
    ax.set_ylabel(r'Metric Proxy $\Delta\beta^{-2}$')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()

def plot_thermodynamic_plane(smc_out, save_path=None):
    """Plot Entropy vs Internal Energy (Thermodynamic Plane)."""
    U = smc_out['U_trace']
    logZ_trace = smc_out['logZ_trace']
    betas = smc_out['betas'][1:]
    
    # S = beta * U + logZ
    S = betas * U + logZ_trace

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(U, S, c=betas, cmap='magma', edgecolor='none', alpha=0.8)
    ax.plot(U, S, alpha=0.4, color='gray', lw=1)
    ax.set_title('Thermodynamic Plane')
    ax.set_xlabel('Internal Energy U')
    ax.set_ylabel('Entropy S')
    plt.colorbar(scatter, label=r'beta ($\beta$)')
    ax.grid(True, alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()

def plot_stability_map(ess_grid, n_grid, m_grid, title="SMC Stability (min ESS)", save_path=None):
    """Plot heatmap of min ESS across N and M."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ess_grid, origin='lower', cmap='RdYlGn',
                   extent=[min(m_grid), max(m_grid), min(n_grid), max(n_grid)],
                   aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Number of Inducing Points (M)')
    ax.set_ylabel('Number of Data Points (N)')
    plt.colorbar(im, label='min ESS')
    
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.show()
