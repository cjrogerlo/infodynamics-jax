# Rejuvenation Kernels 設計

## 當前狀況

### 已實現的 Rejuvenation Kernels

目前只有 **HMC** 作為 rejuvenation kernel：

1. **`annealed.py`**:
   - `_hmc_kernel()`: 內部實現的 HMC kernel
   - 目標：tempered energy `U_beta = beta * E(phi)`
   - 配置：`rejuvenation: str = "hmc"`

2. **`ibis.py`**:
   - `_hmc_kernel()`: 內部實現的 HMC kernel
   - 目標：full posterior `p(phi | y_{1:t})`
   - 配置：`rejuvenation: str = "hmc"` 或 `None`

### 問題

1. **代碼重複**：
   - `annealed.py` 和 `ibis.py` 都有各自的 `_hmc_kernel` 實現
   - 兩個實現幾乎相同，只是 energy 的處理略有不同

2. **缺少其他 kernels**：
   - 沒有 MALA, NUTS, Slice 等作為 rejuvenation kernels
   - 但 `inference/sampling/` 中已經有這些實現

3. **無法重用**：
   - `inference/sampling/` 中的 kernels 是作為 `InferenceMethod` 實現的
   - 不是直接可重用的 kernel 函數

## 設計選項

### 選項 1: 保持現狀（簡單但重複）

**優點**：
- ✅ 簡單，不需要重構
- ✅ 每個方法可以獨立調整 kernel

**缺點**：
- ❌ 代碼重複
- ❌ 無法重用其他 kernels（MALA, NUTS, Slice）

### 選項 2: 提取共享的 Rejuvenation Kernel 模組（推薦）

創建 `inference/particle/rejuvenation.py`：

```python
# inference/particle/rejuvenation.py
"""
Rejuvenation kernels for particle-based methods.

These kernels are used to refresh particles after resampling in SMC methods.
They target either:
  - Tempered distribution: π_β(φ) ∝ p(φ) p(y|φ)^β (for Annealed SMC)
  - Full posterior: p(φ | y_{1:t}) (for IBIS)
"""

def hmc_rejuvenate(key, particles, energy_fn, step_size, n_leapfrog, n_steps):
    """HMC rejuvenation kernel."""
    ...

def mala_rejuvenate(key, particles, energy_fn, step_size, n_steps):
    """MALA rejuvenation kernel."""
    ...

def nuts_rejuvenate(key, particles, energy_fn, step_size, n_steps):
    """NUTS rejuvenation kernel."""
    ...
```

**優點**：
- ✅ 消除代碼重複
- ✅ 可以添加多種 rejuvenation kernels
- ✅ 可以重用 `inference/sampling/` 中的邏輯

**缺點**：
- ⚠️ 需要重構現有代碼

### 選項 3: 重用 `inference/sampling/` 中的 Kernels

**問題**：
- `inference/sampling/` 中的 kernels 是 `InferenceMethod`，不是函數
- 它們的接口是 `run(energy, phi_init, ...)`，不是 `kernel(key, particles, ...)`

**解決方案**：
- 提取 `inference/sampling/` 中的核心 kernel 邏輯為函數
- 或創建 adapter 將 `InferenceMethod` 轉換為 rejuvenation kernel

**優點**：
- ✅ 最大化代碼重用
- ✅ 統一所有 MCMC kernels

**缺點**：
- ⚠️ 需要較大的重構
- ⚠️ 可能過度設計

## 推薦方案

**選項 2（提取共享模組）**，原因：

1. **適度的抽象**：
   - 不需要重構整個 `inference/sampling/`
   - 只需要提取 rejuvenation 需要的部分

2. **清晰的職責**：
   - `inference/sampling/`: 獨立的 MCMC 方法
   - `inference/particle/rejuvenation.py`: SMC 中的 rejuvenation kernels

3. **易於擴展**：
   - 可以輕鬆添加新的 rejuvenation kernels
   - 不需要修改 `inference/sampling/`

## 實現建議

### 1. 創建 `rejuvenation.py`

```python
# inference/particle/rejuvenation.py
"""
Rejuvenation kernels for particle-based methods.
"""

def hmc_rejuvenate(
    key,
    particles,  # pytree stacked [P, ...]
    energy_fn,  # function(phi) -> scalar
    step_size: float = 1e-2,
    n_leapfrog: int = 4,
    n_steps: int = 1,
) -> Any:
    """
    HMC rejuvenation kernel.
    
    Args:
        key: PRNG key
        particles: Stacked particles pytree [P, ...]
        energy_fn: Energy function (phi) -> scalar
        step_size: HMC step size
        n_leapfrog: Number of leapfrog steps
        n_steps: Number of HMC steps per particle
    
    Returns:
        rejuvenated_particles: pytree stacked [P, ...]
    """
    ...
```

### 2. 更新 `annealed.py` 和 `ibis.py`

```python
# annealed.py
from .rejuvenation import hmc_rejuvenate

# 在 run() 中：
if rejuvenation == "hmc":
    def energy_fn(phi):
        return beta * energy(phi, *energy_args, **energy_kwargs)
    particles = hmc_rejuvenate(
        key_rejuv, particles, energy_fn,
        step_size=self.cfg.step_size,
        n_leapfrog=self.cfg.n_leapfrog,
        n_steps=self.cfg.rejuvenation_steps,
    )
```

### 3. 添加其他 Kernels（可選）

```python
# 未來可以添加：
if rejuvenation == "mala":
    particles = mala_rejuvenate(...)
elif rejuvenation == "nuts":
    particles = nuts_rejuvenate(...)
```

## 配置更新

```python
@dataclass(frozen=True)
class AnnealedSMCCFG:
    ...
    rejuvenation: str = "hmc"  # "hmc", "mala", "nuts", None
    rejuvenation_steps: int = 1
    step_size: float = 1e-2  # For HMC/MALA
    n_leapfrog: int = 4  # For HMC
```

## 結論

**當前狀況**：
- ✅ 只有 HMC 作為 rejuvenation kernel
- ❌ 代碼重複（`annealed.py` 和 `ibis.py` 都有 `_hmc_kernel`）

**建議**：
- 創建 `inference/particle/rejuvenation.py` 提取共享邏輯
- 未來可以添加 MALA, NUTS 等作為選項
- 保持與 `inference/sampling/` 的獨立性（不同的使用場景）
