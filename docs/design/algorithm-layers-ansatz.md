# Algorithm 層次與 Ansatz 使用

## 問題核心

不同 algorithm 層次需要不同的抽象：

1. **`sampling/` (HMC, NUTS, MALA)**
   - **層次**: 純 sampler
   - **需要**: `energy(phi, ...) -> scalar`
   - **Ansatz**: ✅ 已通過 `InertialEnergy` 內部處理
   - **不關心**: energy 內部如何計算

2. **`optimisation/map2.py`**
   - **層次**: 優化 energy
   - **需要**: `energy(phi, ...) -> scalar`
   - **Ansatz**: ✅ 已通過 `InertialEnergy` 內部處理
   - **不關心**: energy 內部如何計算

3. **`particle/annealed.py`**
   - **層次**: 使用 energy 進行 β-annealing
   - **需要**: `energy(phi, ...) -> scalar`
   - **Ansatz**: ✅ 已通過 `InertialEnergy` 內部處理
   - **不關心**: energy 內部如何計算

4. **`particle/ibis.py`** ⚠️
   - **層次**: 需要 `log p(y|φ)`，不只是 energy
   - **問題**: `E[-log p(y|f,φ)] ≠ -log p(y|φ)` (non-conjugate)
   - **Ansatz**: ❌ 需要**重新調用** ansatz 來計算 `log p(y|φ)`

## 關鍵問題

### IBIS 需要什麼？

IBIS weight update:
```
logw += log p(y_t | φ)
```

但 `InertialEnergy` 提供:
```
E(φ) = E_{q(f|φ)}[-log p(y|f,φ)]
```

對於 non-conjugate:
- `E[-log p(y|f,φ)]` ≠ `-log E[p(y|f,φ)]` (Jensen's inequality)
- 需要計算 `log ∫ p(y|f,φ) q(f|φ) df`

### Ansatz 已經做了什麼？

`InertialEnergy` 內部調用 ansatz 計算:
```
E[-log p(y|f,φ)] = Σ_i E_{q(f_i|φ)}[-log p(y_i|f_i,φ)]
```

但 IBIS 需要:
```
log p(y|φ) = log ∫ p(y|f,φ) q(f|φ) df
```

這是**不同的計算**！

## 解決方案

### 選項 1: IBIS 直接調用 ansatz（推薦）

讓 IBIS 能夠訪問 `InertialEnergy` 的內部組件來計算 `log p(y|φ)`：

```python
# IBIS 需要能夠：
# 1. 獲取 q(f_i|φ) marginals（ansatz 已經計算）
# 2. 計算 log ∫ p(y_i|f_i,φ) q(f_i|φ) df_i（需要新的 ansatz 函數）
# 3. 對所有 i 求和
```

**問題**: 這需要暴露 ansatz 內部，違反封裝

### 選項 2: 在 `InertialEnergy` 中添加 `log_likelihood` 方法

```python
class InertialEnergy(EnergyTerm):
    def __call__(self, phi, X, Y, key=None):
        # 現有的 energy 計算
        ...
    
    def log_likelihood(self, phi, X, Y, key=None):
        # 計算 log p(y|φ) 使用 ansatz
        # 對於 Gaussian: 使用 analytic
        # 對於 non-conjugate: 使用 ansatz
        ...
```

**問題**: 違反 energy layer 設計原則（不提供 marginal likelihood）

### 選項 3: 創建專門的 `LogLikelihoodTerm`（推薦）

```python
class LogLikelihoodTerm:
    """專門用於 IBIS 的 log likelihood 計算"""
    def __init__(self, inertial_energy: InertialEnergy):
        # 重用 InertialEnergy 的配置（kernel, likelihood, estimator）
        ...
    
    def __call__(self, phi, X, Y, key=None):
        # 使用 ansatz 計算 log p(y|φ)
        # 重用 InertialEnergy 的內部邏輯
        ...
```

**優點**:
- ✅ 不違反 energy layer 設計（這是新的 term，不是 energy）
- ✅ 重用 `InertialEnergy` 的配置和 ansatz 邏輯
- ✅ IBIS 可以接受 `LogLikelihoodTerm` 或 `EnergyTerm`

### 選項 4: 讓 IBIS 接受 `InertialEnergy` 並內部調用 ansatz

```python
class IBIS:
    def _compute_log_likelihood_from_inertial(
        self, 
        inertial: InertialEnergy,
        phi, X, Y, key
    ):
        # 檢查 estimator
        if inertial.cfg.estimator == "analytic":
            # Gaussian: 使用 -energy
            return -inertial(phi, X, Y, key=key)
        else:
            # Non-conjugate: 需要調用 ansatz
            # 重用 InertialEnergy 的內部邏輯
            state = inertial._solve_inner(phi, X, Y)
            # 計算 log p(y|φ) 而不是 E[-log p(y|f,φ)]
            ...
```

**問題**: 需要訪問 `InertialEnergy` 的私有方法

## 推薦方案

**選項 3 + 選項 4 的混合**:

1. 創建 `LogLikelihoodTerm` 作為新的 protocol/interface
2. 讓 `InertialEnergy` 可以轉換為 `LogLikelihoodTerm`
3. IBIS 接受 `LogLikelihoodTerm` 或 `EnergyTerm`（向後兼容）

這樣：
- ✅ 保持 energy layer 的設計原則
- ✅ IBIS 可以正確處理 non-conjugate
- ✅ 重用現有的 ansatz 邏輯
- ✅ 清晰的抽象層次
