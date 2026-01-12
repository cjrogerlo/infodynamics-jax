# Non-Conjugate Support in Particle Methods

## 問題分析

### 當前狀態

1. **`annealed.py`**: ✅ **已支援 non-conjugate**
   - 直接使用 `energy(phi, *args, **kwargs)`
   - 如果 `energy` 是 `InertialEnergy`，它已經處理了 non-conjugate（通過 GH/MC estimators）
   - 不需要額外處理

2. **`ibis.py`**: ⚠️ **部分支援，但有問題**
   - 使用 `-energy` 作為 log likelihood 的近似
   - 對於 Gaussian: `log p(y|φ) ≈ -E[-log p(y|f,φ)]` ✅
   - 對於 non-conjugate: 這是**近似**，不準確 ❌

3. **`sampling/` 方法**: ✅ **已支援 non-conjugate**
   - 只需要 energy function
   - 不關心 energy 內部如何計算（analytic/GH/MC）
   - 這是正確的設計：inference layer 不應該關心 energy 的內部實現

### 核心問題

**IBIS 需要 `log p(y_t | φ)`，但 energy layer 只提供 `E[-log p(y|f,φ)]`**

對於 non-conjugate：
- `E[-log p(y|f,φ)] ≠ -log p(y|φ)`
- 因為 Jensen's inequality: `E[-log p(y|f,φ)] ≥ -log E[p(y|f,φ)]`

---

## 設計層次分析

### Algorithm 層次差異

1. **`sampling/` (HMC, NUTS, MALA)**
   - **層次**: 純 sampler，只關心 energy landscape
   - **輸入**: `energy(phi, ...) -> scalar`
   - **不關心**: energy 如何計算（analytic/GH/MC）
   - **支援 non-conjugate**: ✅ 自動支援（通過 energy layer）

2. **`particle/annealed.py`**
   - **層次**: 使用 energy 進行 β-annealing
   - **輸入**: `energy(phi, ...) -> scalar`
   - **不關心**: energy 如何計算
   - **支援 non-conjugate**: ✅ 自動支援（通過 energy layer）

3. **`particle/ibis.py`**
   - **層次**: 需要 log likelihood，不只是 energy
   - **輸入**: `energy(phi, ...) -> scalar`，但需要 `log p(y|φ)`
   - **問題**: energy layer 不提供 `log p(y|φ)`
   - **支援 non-conjugate**: ⚠️ 目前是近似

---

## 解決方案

### 選項 1: 在 Energy Layer 添加 log_likelihood 方法（不推薦）

**問題**: 違反設計原則
- Energy layer 明確不提供 `marginal_likelihood` 或 `log_evidence`
- 這會破壞 "inference-as-dynamics" 的設計哲學

### 選項 2: 在 IBIS 中接受 log_likelihood_fn（推薦）

**設計**: IBIS 接受可選的 `log_likelihood_fn`，如果提供則使用，否則回退到 `-energy` 近似

```python
def run(
    self,
    energy: EnergyTerm,
    init_particles_fn: Callable,
    data_stream: ...,
    *,
    key: jax.random.PRNGKey,
    log_likelihood_fn: Optional[Callable] = None,  # 新增
    energy_kwargs: Optional[dict] = None,
) -> IBISRun:
```

**優點**:
- 不違反 energy layer 的設計原則
- 對於 Gaussian，可以通過 `vfe_objective` 或 analytic 計算
- 對於 non-conjugate，用戶可以提供自己的實現
- 向後兼容（默認使用 `-energy` 近似）

### 選項 3: 使用 TargetEnergy 組合（推薦）

**設計**: IBIS 接受 `TargetEnergy`，它已經組合了 inertial + prior

```python
# 用戶組合
target_energy = TargetEnergy(
    inertial=InertialEnergy(...),  # 處理 non-conjugate
    prior=PriorEnergy(...)
)

# IBIS 使用
ibis.run(
    energy=target_energy,  # 已經處理了 non-conjugate
    ...
)
```

**但問題**: IBIS 仍然需要 `log p(y|φ)`，不只是 `E[-log p(y|f,φ)]`

---

## 最終建議

### 對於 `annealed.py`: ✅ 無需修改
- 已經正確使用 energy
- 自動支援 non-conjugate（通過 energy layer）

### 對於 `ibis.py`: 添加 `log_likelihood_fn` 參數

**實現策略**:
1. 如果提供 `log_likelihood_fn`，使用它
2. 否則，使用 `-energy` 作為近似（向後兼容）
3. 在文檔中明確說明：
   - Gaussian: `-energy` 是準確的
   - Non-conjugate: 需要提供 `log_likelihood_fn` 或接受近似

**設計理由**:
- 不違反 energy layer 的設計原則
- 保持 inference layer 的抽象（不關心 energy 內部）
- 給用戶靈活性來提供準確的 log likelihood（如果需要）

---

## 總結

1. **`annealed.py`**: ✅ 已支援 non-conjugate（通過 energy layer）
2. **`ibis.py`**: ⚠️ 需要改進（添加 `log_likelihood_fn` 參數）
3. **`sampling/`**: ✅ 已支援 non-conjugate（通過 energy layer）

**關鍵洞察**: 
- `sampling/` 和 `annealed.py` 只需要 energy，所以自動支援 non-conjugate
- `ibis.py` 需要 log likelihood，這超出了 energy layer 的範圍
- 解決方案：讓 IBIS 接受可選的 `log_likelihood_fn`，保持設計層次清晰
