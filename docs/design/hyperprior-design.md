# Hyperprior 設計

## 概述

Hyperprior 是對結構超參數 φ 的 prior：
- `kernel_params`: kernel 超參數
- `Z`: inducing point 位置
- `likelihood_params`: likelihood 超參數

**重要**：Hyperprior **不是** EnergyTerm，因為它不符合 energy 的定義（見 `docs/design/hyperprior-not-energy.md`）。

## 設計原則

1. **Hyperprior 不是 Energy**：
   - Energy = `E[-log p(y|f,phi)]`（數據相關）
   - Hyperprior = `-log p(phi)`（數據無關）

2. **通用工具函數**：
   - 提供工具函數（不是 EnergyTerm）
   - 可以在所有 inference 方法中使用

3. **選擇性應用**：
   - 不同 kernels/likelihoods 使用不同的參數子集
   - 通過 `fields` 和 `keys` 參數選擇性應用 priors

## 位置

Hyperprior 工具函數位於：
- `infodynamics_jax/infodynamics/hyperprior.py`

這是在 `infodynamics/` 層次，因為：
- 所有 inference 方法都需要（MAP2, HMC, NUTS, SMC, IBIS）
- 不是 energy layer（不符合 energy 定義）
- 不是單個 inference method（通用工具）

## API

### 工具函數

```python
from infodynamics_jax.infodynamics import make_hyperprior

# 創建 hyperprior 函數
hyperprior = make_hyperprior(
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance"],  # RBF kernel
    likelihood_log_lambda=0.1,
    likelihood_keys=["noise_var"],  # Gaussian likelihood
)
```

### 使用方式

#### 方式 1: 在 runner 層次添加（推薦）

```python
from infodynamics_jax.infodynamics import run, make_hyperprior
from infodynamics_jax.inference.sampling import HMC, HMCCFG

hyperprior = make_hyperprior(
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance"],
)

out = run(
    key=key,
    method=HMC(cfg=HMCCFG(...)),
    energy=target_energy,
    phi_init=phi_init,
    energy_args=(X, Y),
    hyperprior=hyperprior,  # ✅ 所有方法都支持
)
```

#### 方式 2: 通過 TargetEnergy.extra 添加

```python
from infodynamics_jax.energy import TargetEnergy
from infodynamics_jax.infodynamics import make_hyperprior

hyperprior = make_hyperprior(...)

target = TargetEnergy(
    inertial=inertial_energy,
    extra=[hyperprior],  # 作為 extra term
)
```

## 處理不同 Kernels/Likelihoods

### 問題

不同 kernels/likelihoods 使用不同的參數：
- **RBF kernel**: `lengthscale`, `variance`
- **Matern kernel**: `lengthscale`, `variance`, `nu`
- **Periodic kernel**: `lengthscale`, `variance`, `period`
- **Gaussian likelihood**: `noise_var`
- **Bernoulli likelihood**: (可能沒有 hyperparameters)

### 解決方案：選擇性字段/鍵

通過 `fields` 和 `keys` 參數選擇性地應用 priors：

```python
# RBF kernel: 只對 lengthscale 和 variance 施加 prior
hyperprior = make_hyperprior(
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance"],  # 只選擇這兩個
)

# Matern kernel: 對 lengthscale, variance, nu 施加 prior
hyperprior = make_hyperprior(
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance", "nu"],  # 包含 nu
)

# Gaussian likelihood: 對 noise_var 施加 prior
hyperprior = make_hyperprior(
    likelihood_log_lambda=0.1,
    likelihood_keys=["noise_var"],
)

# Bernoulli likelihood: 可能不需要 hyperprior（沒有 hyperparameters）
# 或者可以對其他參數施加 prior（如果有的話）
```

### 設計原則

1. **選擇性應用**：通過 `fields`/`keys` 選擇參數子集
2. **向後兼容**：如果 `fields`/`keys` 為 None，使用默認值
3. **類型安全**：使用 `hasattr` 和 `in` 檢查參數是否存在
4. **模組化**：不同的 prior 類型（L2, log-L2）可以組合

## 支持的 Prior 類型

### L2 Prior

```python
kernel_l2_hyperprior(phi, fields=["lengthscale"], lam=1.0)
# = 0.5 * lam * sum(phi.kernel_params.lengthscale ** 2)
```

### Log-L2 Prior (Log-Normal)

```python
kernel_log_l2_hyperprior(phi, fields=["lengthscale"], lam=1.0, mu={"lengthscale": 0.0})
# = 0.5 * lam * sum((log(lengthscale) - mu) ** 2)
```

適合正參數（如 `lengthscale`, `variance`, `noise_var`）。

### Z Prior

```python
z_l2_hyperprior(phi, lam=1.0)
# = 0.5 * lam * sum(phi.Z ** 2)
```

## 使用範例

### 完整範例

```python
from infodynamics_jax.infodynamics import run, make_hyperprior
from infodynamics_jax.inference.optimisation import MAP2, MAP2CFG
from infodynamics_jax.energy import TargetEnergy, InertialEnergy

# 創建 hyperprior（根據 kernel/likelihood 選擇參數）
hyperprior = make_hyperprior(
    # RBF kernel priors
    kernel_log_lambda=0.1,
    kernel_fields=["lengthscale", "variance"],
    
    # Gaussian likelihood prior
    likelihood_log_lambda=0.1,
    likelihood_keys=["noise_var"],
    
    # Inducing points prior
    z_lambda=0.01,
)

# 創建 energy
inertial = InertialEnergy(...)
target = TargetEnergy(inertial=inertial)

# MAP2 優化（hyperprior 在 runner 層次添加）
method = MAP2(cfg=MAP2CFG(steps=200, lr=1e-2))
out = run(
    key=key,
    method=method,
    energy=target,
    phi_init=phi_init,
    energy_args=(X, Y),
    hyperprior=hyperprior,  # ✅ 所有方法都支持
)
```

## 擴展性

如果需要新的 prior 類型，可以：
1. 在 `infodynamics/hyperprior.py` 中添加新的 atomic 函數（如 `kernel_l1_hyperprior`）
2. 在 `make_hyperprior` 中添加對應的參數
3. 在 `make_hyperprior` 的實現中添加處理邏輯

這樣設計保持了：
- ✅ 通用性（適用於所有 kernels/likelihoods）
- ✅ 靈活性（選擇性應用）
- ✅ 可擴展性（易於添加新的 prior 類型）

## 與舊設計的區別

### 舊設計（錯誤）

```python
# ❌ HyperpriorEnergy 繼承 EnergyTerm
class HyperpriorEnergy(EnergyTerm):
    ...
```

**問題**：
- Hyperprior 不是 energy
- 違反了 energy layer 的設計原則

### 新設計（正確）

```python
# ✅ 工具函數（不是 EnergyTerm）
def make_hyperprior(...) -> Callable[[Phi], jnp.ndarray]:
    ...
```

**優點**：
- 符合設計原則（hyperprior 不是 energy）
- 通用（所有 inference 方法都可以使用）
- 靈活（可以通過 runner 或 TargetEnergy.extra 添加）
