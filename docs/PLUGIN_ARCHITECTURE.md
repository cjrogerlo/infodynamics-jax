# Plugin Architecture Guide

## Overview

The `infodynamics-jax` plugin system allows you to:

1. ✅ Keep the **public core library** open-source with standard implementations
2. ✅ Build **private projects** with proprietary data sources and models
3. ✅ **Gradually promote** successful private features back to public core
4. ✅ **Mix and match** public and private components at runtime

---

## Design Principles

### Public Core Responsibilities
The public `infodynamics-jax` library provides:
- Abstract plugin interfaces (`DataStreamPlugin`, `ObservationEnergyPlugin`, etc.)
- Core algorithms (oEGPF, SMC, GP dynamics)
- Standard implementations (Gaussian likelihoods, CSV data loaders)
- Plugin discovery and registry system

### Private Project Responsibilities
Your private repositories can provide:
- Proprietary data adaptors (LOB data, medical imaging, etc.)
- Custom likelihood models (Student-t, regime-switching, etc.)
- Domain-specific evaluators (trading P&L, risk metrics)
- Specialized features that shouldn't be public

### Key Invariant
**The public core NEVER imports or depends on private code.**

Private projects depend on public core:
```
infodynamics-jax (public)  ←──  hedge-fund-project (private)
                           ←──  medical-imaging-project (private)
                           ←──  quant-research-alpha (private)
```

---

## Architecture Patterns

### Pattern 1: Private Project as Downstream Package (Recommended)

**When to use:**
- Each project is a standalone application
- Different projects may use different versions of core library
- You want clean dependency management

**Structure:**
```
repos/
├── infodynamics-jax/          (public GitHub)
│   ├── infodynamics_jax/
│   │   ├── core/
│   │   ├── inference/
│   │   ├── plugins/           ← Plugin system (NEW)
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py  ← Abstract base classes
│   │   │   ├── registry.py    ← Plugin discovery
│   │   │   └── public_implementations.py
│   │   └── ...
│   └── setup.py
│
├── hedge-fund-alpha/          (private repo)
│   ├── hf_alpha/
│   │   ├── plugins/
│   │   │   ├── lob_data.py         ← LOBDataStream(DataStreamPlugin)
│   │   │   ├── student_t_obs.py    ← StudentTObsEnergy(...)
│   │   │   └── trading_eval.py     ← TradingRiskEvaluator(...)
│   │   ├── strategies/
│   │   └── execution/
│   ├── setup.py                     ← Depends on: infodynamics-jax
│   └── pyproject.toml
│
└── quant-research-beta/       (private repo)
    ├── qr_beta/
    │   ├── plugins/
    │   │   └── custom_gp.py         ← RegimeSwitchingGP(...)
    │   └── ...
    └── setup.py                     ← Depends on: infodynamics-jax
```

**Installation (private project):**
```bash
cd hedge-fund-alpha/

# Install public core
pip install infodynamics-jax

# Install private project (which extends core)
pip install -e .
```

**Usage in private project:**
```python
from infodynamics_jax.plugins import register_plugin, load_plugin
from hf_alpha.plugins.lob_data import LOBDataStream
from hf_alpha.plugins.student_t_obs import StudentTObsEnergy

# Plugins can auto-register via entry points (see below)
# or explicitly:
register_plugin('data_stream', 'lob', LOBDataStream)
register_plugin('obs_energy', 'student_t', StudentTObsEnergy)

# Now use in your pipeline
StreamClass = load_plugin('data_stream', 'lob')
data_stream = StreamClass(config)

EnergyClass = load_plugin('obs_energy', 'student_t')
obs_energy = EnergyClass()
```

---

### Pattern 2: Entry Points (Most Professional)

**Advantages:**
- Automatic plugin discovery
- No manual registration needed
- Follows Python packaging best practices
- Used by major projects (pytest, Django, Flask)

**Setup (in private project's `pyproject.toml`):**
```toml
[project.entry-points."infodynamics_jax.plugins"]
data_stream.lob = "hf_alpha.plugins.lob_data:LOBDataStream"
data_stream.options = "hf_alpha.plugins.options_data:OptionsStream"
obs_energy.student_t = "hf_alpha.plugins.student_t_obs:StudentTObsEnergy"
risk_evaluator.trading = "hf_alpha.plugins.trading_eval:TradingRiskEvaluator"
```

**Or in `setup.py`:**
```python
setup(
    name="hedge-fund-alpha",
    # ... other config ...
    entry_points={
        'infodynamics_jax.plugins': [
            'data_stream.lob = hf_alpha.plugins.lob_data:LOBDataStream',
            'obs_energy.student_t = hf_alpha.plugins.student_t_obs:StudentTObsEnergy',
        ],
    },
)
```

**Usage (automatic discovery):**
```python
from infodynamics_jax.plugins import discover_plugins, load_plugin

# Auto-discover all installed plugins
discover_plugins(entry_points=True)

# Use proprietary plugin with fallback to public
StreamClass = load_plugin(
    'data_stream',
    'lob',                    # Try proprietary first
    fallback='public_csv'     # Fall back to public implementation
)
```

---

### Pattern 3: Private Package on Private PyPI/GitHub

**When to use:**
- You want a shared private library across multiple projects
- Example: `infodynamics-jax-proprietary` used by multiple hedge fund strategies

**Structure:**
```
infodynamics-jax/              (public GitHub)
infodynamics-jax-proprietary/  (private GitHub or private PyPI)
  ├── infodynamics_jax_proprietary/
  │   ├── plugins/
  │   └── ...
  └── setup.py  ← Depends on: infodynamics-jax (public)

hedge-fund-alpha/              (private repo)
  └── setup.py  ← Depends on:
                   - infodynamics-jax (public)
                   - infodynamics-jax-proprietary (private)
```

**Installation via GitHub SSH:**
```bash
pip install infodynamics-jax
pip install git+ssh://git@github.com/YourOrg/infodynamics-jax-proprietary.git@main
pip install -e .  # Your project
```

**Or via private PyPI:**
```bash
pip install infodynamics-jax
pip install infodynamics-jax-proprietary --extra-index-url https://pypi.yourcompany.com/simple
pip install -e .
```

---

## Plugin Interface Reference

### DataStreamPlugin

```python
from infodynamics_jax.plugins import DataStreamPlugin
import jax.numpy as jnp

class ProprietaryLOBStream(DataStreamPlugin):
    """Load limit order book data from proprietary database."""

    def __init__(self, symbol: str, start_date: str, end_date: str):
        # Your proprietary data loading logic
        self.data = load_lob_from_db(symbol, start_date, end_date)
        self.current_idx = 0

    def next(self):
        if self.current_idx >= len(self.data):
            raise StopIteration

        row = self.data[self.current_idx]
        t = row.timestamp

        # Extract features (bid-ask spread, depth, etc.)
        y_t = jnp.array([
            row.bid_ask_spread,
            row.depth_imbalance,
            row.trade_volume,
        ])

        meta = {'market_regime': row.regime, 'volatility': row.volatility}

        self.current_idx += 1
        return t, y_t, meta

    def reset(self):
        self.current_idx = 0

    @property
    def observation_dim(self):
        return 3  # 3 features
```

### ObservationEnergyPlugin

```python
from infodynamics_jax.plugins import ObservationEnergyPlugin
import jax.numpy as jnp
from jax.scipy.special import gammaln

class StudentTObsEnergy(ObservationEnergyPlugin):
    """
    Student-t observation model for heavy-tailed noise.

    Better for financial data with outliers.
    """

    def energy(self, y_t, x_t, graph, params):
        nu = params.get('degrees_of_freedom', 5.0)  # df
        scale = params.get('observation_noise', 1.0)

        # Predicted observation (assuming direct observation)
        y_pred = x_t  # Or use a more complex observation function

        # Student-t negative log-likelihood
        residual = y_t - y_pred
        d = len(residual)

        log_prob = (
            gammaln((nu + d) / 2)
            - gammaln(nu / 2)
            - (d / 2) * jnp.log(nu * jnp.pi * scale**2)
            - ((nu + d) / 2) * jnp.log(1 + jnp.sum(residual**2) / (nu * scale**2))
        )

        return -log_prob  # Energy = negative log-likelihood

    def grad_x(self, y_t, x_t, graph, params):
        from jax import grad
        grad_fn = grad(lambda x: self.energy(y_t, x, graph, params))
        return grad_fn(x_t)
```

### RiskEvaluatorPlugin

```python
from infodynamics_jax.plugins import RiskEvaluatorPlugin
import jax.numpy as jnp

class TradingRiskEvaluator(RiskEvaluatorPlugin):
    """
    Proprietary trading risk metrics.

    Evaluates predictions in terms of:
    - P&L if used for trading
    - Sharpe ratio
    - Max drawdown
    - VaR/CVaR
    """

    def evaluate(self, predictions, actuals, metadata=None):
        metrics = {}

        # Simulated trading P&L
        # (assumes prediction['mean'] is price forecast)
        positions = jnp.sign(predictions['mean'][:-1])  # Long if predict up
        returns = jnp.diff(actuals, axis=0)
        pnl = positions * returns

        metrics['total_pnl'] = float(jnp.sum(pnl))
        metrics['sharpe_ratio'] = float(
            jnp.mean(pnl) / (jnp.std(pnl) + 1e-8) * jnp.sqrt(252)
        )

        # Max drawdown
        cumulative_pnl = jnp.cumsum(pnl)
        running_max = jnp.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        metrics['max_drawdown'] = float(jnp.min(drawdown))

        # VaR and CVaR (from posterior samples)
        if 'samples' in predictions:
            samples = predictions['samples']  # [num_samples, T, D]

            # 95% VaR
            var_95 = jnp.percentile(samples, 5, axis=0)
            exceedances = actuals < var_95
            metrics['var_95_violations'] = float(jnp.mean(exceedances))

            # CVaR (expected shortfall)
            cvar = jnp.mean(jnp.where(exceedances, actuals - var_95, 0))
            metrics['cvar_95'] = float(cvar)

        return metrics
```

---

## Example: Building a Private Project

### Step 1: Create Private Project Structure

```bash
mkdir hedge-fund-alpha
cd hedge-fund-alpha

# Create package structure
mkdir -p hf_alpha/plugins
touch hf_alpha/__init__.py
touch hf_alpha/plugins/__init__.py
```

### Step 2: Setup Dependencies

**`setup.py`:**
```python
from setuptools import setup, find_packages

setup(
    name="hedge-fund-alpha",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "infodynamics-jax>=0.1.0",  # Public core
        # Your other dependencies
        "pandas>=1.5.0",
        "sqlalchemy>=2.0.0",  # For database access
    ],
    # Register plugins via entry points
    entry_points={
        'infodynamics_jax.plugins': [
            'data_stream.lob = hf_alpha.plugins.lob_data:LOBDataStream',
            'obs_energy.student_t = hf_alpha.plugins.obs_energy:StudentTObsEnergy',
            'risk_evaluator.trading = hf_alpha.plugins.risk_eval:TradingRiskEvaluator',
        ],
    },
)
```

### Step 3: Implement Plugins

**`hf_alpha/plugins/lob_data.py`:**
```python
from infodynamics_jax.plugins import DataStreamPlugin
import jax.numpy as jnp

class LOBDataStream(DataStreamPlugin):
    """Proprietary LOB data loader."""

    def __init__(self, symbol: str, **kwargs):
        # Your proprietary data loading
        from .db_client import load_lob_data
        self.data = load_lob_data(symbol, **kwargs)
        self.idx = 0

    def next(self):
        if self.idx >= len(self.data):
            raise StopIteration
        row = self.data[self.idx]
        self.idx += 1
        return row.time, jnp.array(row.features), row.metadata

    def reset(self):
        self.idx = 0

    @property
    def observation_dim(self):
        return len(self.data[0].features)
```

### Step 4: Use in Your Application

**`hf_alpha/main.py`:**
```python
from infodynamics_jax.plugins import load_plugin, discover_plugins
from infodynamics_jax.inference import oEGPF

# Discover all plugins (including proprietary ones)
discover_plugins(entry_points=True)

# Load proprietary data stream
StreamClass = load_plugin('data_stream', 'lob')
data = StreamClass(symbol='AAPL', start_date='2024-01-01')

# Load proprietary observation energy
EnergyClass = load_plugin('obs_energy', 'student_t')
obs_energy = EnergyClass()

# Run inference with proprietary components
result = oEGPF.run(
    data_stream=data,
    obs_energy=obs_energy,
    # ... other config ...
)

# Evaluate with proprietary metrics
EvalClass = load_plugin('risk_evaluator', 'trading')
evaluator = EvalClass()
metrics = evaluator.evaluate(result.predictions, result.actuals)

print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max drawdown: {metrics['max_drawdown']:.2f}")
```

---

## Gradual Migration: Private → Public

As your private projects mature, you may want to promote some features to the public core.

### Example: Student-t Likelihood

1. **Initially in private repo:**
   ```
   hedge-fund-alpha/
     └── hf_alpha/plugins/student_t_obs.py  (proprietary)
   ```

2. **After testing, promote to public:**
   ```python
   # Move to: infodynamics-jax/infodynamics_jax/energy/student_t.py

   class StudentTObsEnergy(ObservationEnergyPlugin):
       """Now part of public library."""
       # ... (same implementation, but now open-source)
   ```

3. **Update private repo:**
   ```python
   # hf_alpha/plugins/__init__.py

   # No longer need our own implementation
   # from .student_t_obs import StudentTObsEnergy  ❌ Remove

   # Use public version instead
   from infodynamics_jax.energy import StudentTObsEnergy  ✅

   # Still register it with a proprietary name if you want
   from infodynamics_jax.plugins import register_plugin
   register_plugin('obs_energy', 'heavy_tail', StudentTObsEnergy)
   ```

---

## Testing Strategy

### Public Core Tests
Test plugin system with mock implementations:

```python
# tests/test_plugin_registry.py

def test_plugin_registration():
    from infodynamics_jax.plugins import register_plugin, load_plugin

    class MockDataStream:
        pass

    register_plugin('data_stream', 'mock', MockDataStream)

    loaded = load_plugin('data_stream', 'mock')
    assert loaded is MockDataStream

def test_entry_point_discovery():
    from infodynamics_jax.plugins import discover_plugins

    results = discover_plugins(entry_points=True)
    assert results['entry_points'] >= 0  # May find plugins if installed
```

### Private Project Tests
Test with real proprietary data:

```python
# hf_alpha/tests/test_lob_plugin.py

def test_lob_data_stream():
    from hf_alpha.plugins.lob_data import LOBDataStream

    stream = LOBDataStream('AAPL', start_date='2024-01-01')
    t, y_t, meta = stream.next()

    assert y_t.shape[0] == stream.observation_dim
    assert 'market_regime' in meta
```

---

## Security Considerations

### ⚠️ NEVER Put Secrets in Plugin Code

**Bad:**
```python
class ProprietaryDataStream(DataStreamPlugin):
    API_KEY = "sk-1234567890abcdef"  # ❌ NEVER DO THIS
```

**Good:**
```python
import os

class ProprietaryDataStream(DataStreamPlugin):
    def __init__(self):
        self.api_key = os.getenv('DATA_API_KEY')  # ✅ Use environment variables
        if not self.api_key:
            raise ValueError("DATA_API_KEY environment variable not set")
```

### Access Control

- Public repo: Anyone can see
- Private repo: Control via GitHub org permissions
- Private PyPI: Require authentication for installation

---

## Summary: What Should Go Where?

### ✅ Public Core (`infodynamics-jax`)
- Abstract interfaces
- Standard algorithms (oEGPF, SMC)
- Common implementations (Gaussian, CSV loader)
- Plugin registry system
- Documentation and examples

### 🔒 Private Projects
- Proprietary data sources
- Custom likelihoods/dynamics
- Domain-specific evaluators
- Secrets and credentials
- Trading strategies
- Specialized features

### 📈 Migration Path
```
Private experimental → Private stable → Public contribution
```

---

## Next Steps

1. **Now:** Plugin system is ready in public core
2. **Create your first private project:**
   ```bash
   mkdir my-private-project
   cd my-private-project
   # Follow "Building a Private Project" example above
   ```
3. **Start with simple plugins** (e.g., custom data loader)
4. **Gradually add sophisticated models** (custom likelihoods, risk evaluators)
5. **Promote mature features** back to public core when ready

Questions? Check `examples/plugins/` for working code samples.
