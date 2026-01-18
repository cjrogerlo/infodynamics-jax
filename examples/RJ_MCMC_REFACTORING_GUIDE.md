# RJ-MCMC 重構指南：從自定義實現到 infodynamics-jax

本指南說明如何將自定義的 RJ-MCMC 實現（`sbgp_rj_sgpr_rank1_scanjit_fixed_v3_traceinit.py`）重構為使用 `infodynamics-jax` 庫的組件。

## 重構對照表

### 1. 核心組件替換

| 原代碼 | infodynamics-jax 替代 | 說明 |
|--------|----------------------|------|
| 自定義 `State` 類 | `Phi` + 自定義 `RJState` | 使用 `Phi` 存儲結構參數，保留緩存矩陣 |
| 自定義 `rbf_kernel` | `infodynamics_jax.gp.kernels.rbf` | 使用庫的 RBF 核函數 |
| 自定義 VFE 計算 | `vfe_objective()` | 使用庫的 VFE 計算（可選，或用於驗證） |
| `KernelParams` | `infodynamics_jax.gp.kernels.params.KernelParams` | 使用庫的參數結構 |

### 2. 主要重構步驟

#### Step 1: 替換 Kernel 函數

**原代碼：**
```python
def rbf_kernel(X1, X2, log_ls, log_sf):
    ls = jnp.exp(log_ls).reshape(1, 1, -1)
    sf2 = jnp.exp(2.0 * log_sf)
    d2 = jnp.sum(((X1[:, None, :] - X2[None, :, :]) / ls) ** 2, axis=-1)
    return sf2 * jnp.exp(-0.5 * d2)
```

**重構為：**
```python
from infodynamics_jax.gp.kernels import rbf as rbf_kernel
from infodynamics_jax.gp.kernels.params import KernelParams

# 使用 infodynamics-jax 的 kernel
kernel_params = KernelParams(
    lengthscale=jnp.exp(log_ls),
    variance=jnp.exp(2.0 * log_sf)
)
K = rbf_kernel(X1, X2, kernel_params)
```

#### Step 2: 使用 Phi 結構

**原代碼：**
```python
@dataclass
class State:
    theta: jnp.ndarray      # (D+2,)
    Z_buf: jnp.ndarray      # (M_max,) int32
    M: jnp.ndarray          # () int32
    # ... cached matrices ...
```

**重構為：**
```python
from infodynamics_jax.core import Phi

@dataclass
class RJState:
    phi: Phi                    # 使用庫的 Phi 結構
    M: jnp.ndarray
    Z_buf: jnp.ndarray
    # ... 保留緩存矩陣 ...
```

#### Step 3: 構建 Phi 對象

**原代碼：**
```python
log_ls, log_sf, log_sn = theta[:D], theta[D], theta[D+1]
# 直接使用 log_ls, log_sf, log_sn
```

**重構為：**
```python
kernel_params = KernelParams(
    lengthscale=jnp.exp(log_ls),
    variance=jnp.exp(2.0 * log_sf)
)
phi = Phi(
    kernel_params=kernel_params,
    Z=X[Z_buf[:M]],
    likelihood_params={"noise_var": jnp.exp(2.0 * log_sn)},
    jitter=1e-6
)
```

#### Step 4: 驗證 VFE 計算（可選）

可以使用庫的 `vfe_objective()` 來驗證自定義的 ELBO 計算：

```python
from infodynamics_jax.inference.optimisation.vfe import vfe_objective

# 計算 VFE（用於驗證）
elbo_library = -vfe_objective(phi, X, Y, kernel_fn=rbf_kernel, residual="fitc")

# 與自定義計算比較
assert jnp.allclose(elbo_custom, elbo_library, atol=1e-4)
```

### 3. 保留的部分

以下部分**不需要重構**，可以保留：

1. **Rank-1 Updates**：`birth_rank1_update()`, `death_drop_last()`
   - 這些是高效的實現細節，與庫的核心組件無關

2. **RJ Moves**：`rj_step()`, `birth_pool_choose()`
   - 這些是 MCMC 特定的邏輯，庫不提供

3. **HMC Implementation**：可以使用 `blackjax` 或 `infodynamics_jax.inference.sampling.hmc`
   - 參考代碼使用 `blackjax`，這是合理的選擇
   - 也可以使用庫的 HMC，但需要適配

4. **Prior Functions**：`log_prior_theta()`, `log_prior_M_trunc_geom()`
   - 這些是模型特定的，不屬於庫的核心功能

### 4. 完整重構示例

參見 `notebook_08_rjmcmc_advanced.ipynb` 中的示例代碼。

### 5. 優勢

使用 `infodynamics-jax` 的重構版本有以下優勢：

1. **一致性**：與其他 notebook 使用相同的組件
2. **可維護性**：kernel 和參數結構由庫統一管理
3. **可擴展性**：容易切換不同的 kernel（RBF, Matérn, Periodic 等）
4. **驗證**：可以使用庫的 VFE 計算來驗證自定義實現

### 6. 注意事項

1. **性能**：如果原代碼已經高度優化，重構時需要注意性能
2. **緩存矩陣**：庫的 `Phi` 不包含緩存矩陣，需要自定義 `RJState` 來保留
3. **API 兼容性**：確保重構後的代碼與庫的 API 兼容

## 總結

重構的主要目標是：
- ✅ 使用庫的 kernel 函數和參數結構
- ✅ 使用 `Phi` 來管理結構參數
- ✅ 保留高效的 rank-1 updates 邏輯
- ✅ 保留 MCMC 特定的實現細節

這樣既利用了庫的優勢，又保留了高性能的自定義實現。
