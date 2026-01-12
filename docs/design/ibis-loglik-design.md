# IBIS log_likelihood_fn 設計分析

## 問題

IBIS 需要 `log p(y_t | φ)` 來更新 weights：
```
logw += log p(y_t | φ)
```

但 energy layer 只提供：
```
E(φ) = E_{q(f|φ)}[-log p(y|f,φ)]
```

對於 non-conjugate，由於 Jensen's inequality：
```
E[-log p(y|f,φ)] ≠ -log p(y|φ)
```

## 當前設計：`log_likelihood_fn` 參數

### 優點
- ✅ 不違反 energy layer 設計原則（不要求 energy 提供 marginal likelihood）
- ✅ 給用戶靈活性
- ✅ 向後兼容（默認使用 `-energy` 近似）

### 缺點
- ⚠️ 需要用戶額外提供函數
- ⚠️ 對於 Gaussian，理論上不需要（但實際上 `-energy` 已經足夠）

## 替代方案

### 方案 1: 完全移除 `log_likelihood_fn`，只使用 `-energy`

**理由**：
- 對於 Gaussian: `-energy` 是準確的（up to constant）
- 對於 non-conjugate: 使用近似也是合理的（IBIS 本身是近似方法）

**問題**：
- 對於需要準確 log likelihood 的場景，無法滿足

### 方案 2: 讓 `InertialEnergy` 提供可選的 `log_likelihood` 方法

**問題**：
- ❌ 違反 energy layer 設計原則
- ❌ energy layer 明確不提供 `marginal_likelihood`

### 方案 3: 使用 `vfe_objective` 的負值（僅 Gaussian）

**問題**：
- ❌ `vfe_objective` 是 optimization objective，不是 model energy
- ❌ 只適用於 Gaussian
- ❌ 混合了 optimization 和 inference 層次

### 方案 4: 保持當前設計，但改進文檔

**建議**：
- 明確說明何時需要 `log_likelihood_fn`
- 提供使用範例
- 說明對於 Gaussian，`-energy` 已經足夠

## 推薦方案

**保持當前設計（方案 4）**，但：

1. **改進默認行為**：
   - 對於 Gaussian likelihood，自動使用 `-energy`（準確）
   - 對於 non-conjugate，使用 `-energy` 作為近似（除非提供 `log_likelihood_fn`）

2. **文檔改進**：
   - 明確說明：Gaussian 不需要 `log_likelihood_fn`
   - 說明：non-conjugate 可以使用近似，或提供準確函數

3. **可選增強**：
   - 檢測 `InertialEnergy` 的 `estimator`，如果是 "analytic"（Gaussian），自動使用 `-energy`
   - 但這需要檢查 energy 內部結構，可能違反抽象

## 結論

當前設計是合理的，但可以：
1. 改進文檔，明確說明使用場景
2. 考慮自動檢測 Gaussian 情況（如果可行且不違反抽象）
3. 提供使用範例

**關鍵點**：`log_likelihood_fn` 是**可選的**，默認使用 `-energy` 近似已經足夠大多數場景。
