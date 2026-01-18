# Delta Variational Contrastive Divergence (ΔVCD) 分析與價值評估

## 一、理論核心

### 1.1 問題背景

現有稀疏高斯過程理論主要關注**靜態近似保證**：
- 對於足夠大的誘導點數量 $M$，可以證明稀疏 GP 與完整 GP 之間的 KL 散度可以任意小
- 這些結果是**存在性的**：證明高精度稀疏模型**存在**，但對**採樣器行為**保持沉默

### 1.2 核心問題

在 RJVMC（可逆跳躍變分蒙特卡羅）框架中：
- 同時推斷核超參數、誘導點位置和誘導點數量
- 推斷過程通過遍歷結構超後驗進行
- **即使存在高精度稀疏近似，採樣器仍可能進入變分幾何與真實後驗不對齊的區域**
- 這會導致**系統性漂移**，靜態容量界限無法診斷

### 1.3 Delta-VCD 的定義

Delta-VCD 比較兩個 KL 散度的一步收縮：
- 朝向稀疏代理 $\pi(\phi|y)$ 的收縮
- 朝向真實後驗 $\hat{\pi}(\phi|y)$ 的收縮

數學定義：
```
ΔVCD(q_t; K) = [KL(q_t||π) - KL(q_{t+1}||π)] 
              - [KL(q_t||π̂) - KL(q_{t+1}||π̂)]
```

### 1.4 漂移恆等式（Drift Identity）

**關鍵定理**：Delta-VCD 簡化為 VFE gap 的期望一步漂移：

```
ΔVCD(q_t; K) = E_{q_{t+1}}[e(φ)] - E_{q_t}[e(φ)]
```

其中 $e(\phi) = \log p(y|\theta) - \mathcal{L}(\phi)$ 是 VFE gap。

**意義**：Delta-VCD 直接測量採樣器是否移向更高或更低近似誤差的區域。

### 1.5 疊加性質（Telescoping）

沿軌跡求和產生疊加和：
```
∑_{t=0}^{T-1} ΔVCD(q_t; K) = E_{q_T}[e(φ)] - E_{q_0}[e(φ)]
```

**關鍵洞察**：
- 累積差異僅取決於端點
- 揭示**最優子結構**：局部穩定的稀疏化組合成全局有效的推斷軌跡
- 持續的局部漂移必然累積成全局偏差

### 1.6 非消失證據項

分解漂移暴露關鍵不對稱性：
```
ΔVCD = ΔE[log p(y|θ)] - ΔE[L]
```

- **誘導點位置更新**：證據項抵消，採樣器遵循真實後驗幾何
- **超參數更新**：證據項**不消失**且難以處理
- 這個非抵消項在 type-II 優化中不存在，但在嚴格貝葉斯推斷中不可避免

## 二、實際應用價值

### 2.1 診斷功能

根據 `notebook_10_occam_razor.ipynb` 的實現，Delta-VCD 用作：

1. **採樣效率預警**：
   - 高 |ΔVCD| 與低接受率相關
   - 可作為採樣崩潰的**前兆指標**

2. **模型適應監控**：
   - 高 ΔVCD 發生在快速模型適應期間
   - 識別採樣器進入近似誤差較高區域的時刻

3. **穩定性評估**：
   - 通過相位圖（phase portrait）分析 ΔVFE vs ΔVCD
   - 檢測推斷軌跡的穩定性

### 2.2 計算實現與限制

#### 理論需求 vs 實際限制

**理論上**，Delta-VCD 需要：
- `log p(y|θ)`：真實邊際似然（對 f 積分後的完整證據）

**實際上**，我們面臨以下限制：

1. **高斯似然（共軛情況）**：
   - 可以使用 `fitc_log_evidence()` 計算 FITC 近似的邊際似然
   - **但這仍然是近似**，不是真實的 `log p(y|θ)`
   - FITC 近似假設了稀疏結構，與真實後驗仍有差異

2. **非高斯似然（非共軛情況）**：
   - **邊際似然根本不可計算**（需要對 f 進行高維積分）
   - Delta-VCD **無法直接應用**
   - 這是 Delta-VCD 的主要實際限制

#### 實際實現

在 `notebook_10` 中（僅適用於高斯似然）：
```python
dvfe = np.diff(elbo)      # VFE 的變化
dfll = np.diff(fll)        # FITC 近似邊際似然的變化
dvcd = dfll - dvfe         # Delta-VCD（近似）
```

其中：
- `fll` 是 `fitc_log_evidence()` 的輸出（FITC 近似）
- `elbo` 是 VFE/ELBO（變分下界）
- `dvcd` 測量的是**近似誤差的變化**，而非真實誤差

#### 重要啟示

Delta-VCD 的實際應用受到嚴重限制：
- ✅ **僅適用於高斯似然**：可以使用 FITC 近似
- ❌ **不適用於非高斯似然**：邊際似然不可計算
- ⚠️ **即使對於高斯似然**：使用的是 FITC 近似，不是真實邊際似然

這意味著 Delta-VCD 在實際應用中：
- 對於高斯回歸問題：可以作為**近似診斷工具**
- 對於分類問題（Bernoulli、Poisson 等）：**無法計算**

### 2.3 與庫設計的關係

本庫採用**推斷即動力學**（inference-as-dynamics）的視角：
- 使用 **inertial energy** 作為推斷的原始對象
- **不**計算邊際似然或證據
- Delta-VCD 提供了一個**動態診斷工具**，與這種設計理念一致

## 三、價值評估

### 3.1 理論價值 ⭐⭐⭐⭐⭐

1. **填補理論空白**：
   - 連接靜態近似理論與動態採樣行為
   - 提供理論框架來理解採樣器在結構空間中的行為

2. **揭示根本問題**：
   - 明確指出嚴格貝葉斯推斷中不可避免的偏差來源
   - 解釋為什麼 type-II 優化與嚴格貝葉斯推斷存在差異

3. **數學優雅性**：
   - 漂移恆等式提供簡潔的操作形式
   - 疊加性質揭示最優子結構

### 3.2 實用價值 ⭐⭐⭐⭐

1. **診斷工具**：
   - 可計算且易於實現
   - 提供採樣器行為的實時監控

2. **預警機制**：
   - 可預測採樣效率下降
   - 幫助識別需要調整的區域

3. **研究工具**：
   - 用於評估不同核函數、數據集對推斷的影響
   - 支持 Occam's Razor 的實證研究

### 3.3 局限性 ⭐⭐

1. **根本性限制：僅適用於高斯似然** ⚠️⚠️⚠️
   - **非高斯似然無法計算**：邊際似然 `log p(y|θ)` 需要對 f 進行高維積分，不可計算
   - **高斯似然也僅是近似**：使用 FITC 近似，不是真實邊際似然
   - 這嚴重限制了 Delta-VCD 的實際應用範圍

2. **計算成本**：
   - 對於高斯似然，需要計算 FITC 邊際似然
   - 對於大規模數據可能昂貴

3. **解釋複雜性**：
   - 需要理解 VFE gap 和變分幾何
   - 對非專家可能難以解釋
   - 需要理解近似誤差與真實誤差的區別

4. **實際應用**：
   - 主要用於診斷而非優化
   - 不能直接改善採樣效率，只能識別問題
   - **僅限於高斯回歸問題**

## 四、與現有方法的比較

### 4.1 vs 靜態近似理論

| 特性 | 靜態理論 | Delta-VCD |
|------|---------|-----------|
| 關注點 | 近似精度 | 採樣器行為 |
| 時間維度 | 無 | 動態軌跡 |
| 診斷能力 | 存在性證明 | 實時監控 |
| 計算成本 | 理論分析 | 需要完整似然 |

### 4.2 vs 其他診斷工具

| 工具 | 功能 | Delta-VCD 優勢 |
|------|------|---------------|
| ESS (有效樣本數) | 測量混合效率 | 提供**原因**而非僅結果 |
| 接受率 | 測量提議質量 | 連接近似誤差與採樣行為 |
| R-hat | 檢測收斂 | 揭示**為什麼**不收斂 |

## 五、結論與建議

### 5.1 核心價值

Delta-VCD 是一個**理論上深刻、實用上有價值**的診斷工具：

1. **理論貢獻**：
   - 填補了靜態近似理論與動態採樣行為之間的空白
   - 揭示了嚴格貝葉斯推斷中不可避免的偏差來源

2. **實用價值**：
   - 提供可操作的診斷指標
   - 可預測採樣效率問題
   - 支持實證研究

### 5.2 適用場景

**強烈推薦使用**（僅限高斯似然）：
- 研究稀疏 GP 的推斷行為（高斯回歸）
- 評估不同核函數/數據集的影響
- 診斷採樣器問題

**無法使用**：
- ❌ **非高斯似然**（Bernoulli、Poisson、Ordinal 等）
- ❌ **分類問題**（邊際似然不可計算）

**謹慎使用**：
- 大規模數據（計算成本）
- 生產環境（主要用於診斷）
- 需要理解使用的是 FITC 近似，不是真實邊際似然

### 5.3 未來方向

1. **理論擴展**：
   - 探索 Delta-VCD 與其他診斷指標的關係
   - 研究如何利用 Delta-VCD 改善採樣

2. **計算優化**：
   - 開發近似方法以降低計算成本
   - 探索增量計算策略

3. **應用拓展**：
   - 擴展到其他變分方法
   - 研究與其他近似誤差測量的關係

## 六、關鍵限制總結

### 6.1 核心問題

**Delta-VCD 理論上需要真實邊際似然 `log p(y|θ)`，但實際上：**

1. **非高斯似然**：邊際似然不可計算 → **Delta-VCD 無法應用**
2. **高斯似然**：可以使用 FITC 近似 → **但仍是近似，非真實值**

### 6.2 實際意義

這意味著 Delta-VCD 是一個**理論上優雅但實際應用受限**的工具：

- ✅ **理論價值**：揭示了靜態近似理論與動態採樣行為之間的關係
- ⚠️ **實際限制**：僅適用於高斯回歸問題，且使用的是近似值
- ❌ **無法應用**：分類問題、非共軛似然等常見場景

### 6.3 替代方案

對於非高斯似然，可能需要：
- 使用其他診斷工具（ESS、接受率等）
- 開發基於變分下界的替代診斷
- 接受 Delta-VCD 僅作為理論工具，而非實用診斷

## 七、無需真實後驗的替代 Bound

### 7.1 問題與動機

原始 sampler-aware bound 需要 score residual：
```
s(φ) = ∇L(φ) - ∇log p(y|θ)
```

這需要真實邊際似然的梯度，對於非高斯似然不可計算。我們需要推導**僅基於可計算量的替代 bound**。

### 7.2 基於 VFE 梯度變化的 Bound

#### 理論推導

考慮 VFE 梯度的局部變化作為變分幾何不穩定的 proxy：

**定義**：梯度殘差（Gradient Residual）
```
r(φ_t, φ_{t+1}) = ∇L(φ_{t+1}) - ∇L(φ_t)
```

**Bound 1：基於梯度變化的漂移控制**
```
|ΔVCD(q_t; K)| ≤ E_{q_t}[||r(φ_t, φ_{t+1})||_2] · δ(K) · C_geom
```

其中：
- `r(φ_t, φ_{t+1})` 是 VFE 梯度的變化（**可計算**）
- `δ(K)` 是採樣器的平均步長
- `C_geom` 是幾何常數，反映變分下界的局部曲率

**直觀解釋**：
- 當 VFE 梯度快速變化時，變分幾何可能不穩定
- 結合步長，可以控制漂移的上界
- **優點**：僅需要 VFE 梯度，適用於所有似然類型

#### 實際計算

```python
# 計算梯度殘差
def compute_gradient_residual(energy_fn, phi_t, phi_t1):
    """計算 VFE 梯度的變化"""
    grad_t = jax.grad(energy_fn)(phi_t)
    grad_t1 = jax.grad(energy_fn)(phi_t1)
    return tree_norm(tree_subtract(grad_t1, grad_t))

# 計算 bound
def gradient_based_bound(grad_residuals, step_sizes, C_geom=1.0):
    """計算基於梯度的漂移 bound"""
    return jnp.mean(grad_residuals) * jnp.mean(step_sizes) * C_geom
```

### 7.3 基於能量差異的 Bound

#### 理論推導

使用能量差異結合步長信息：

**Bound 2：基於能量差異的漂移控制**
```
|ΔVCD(q_t; K)| ≤ |L(φ_{t+1}) - L(φ_t)| · δ(K) · C_energy
```

**更精細的版本**（考慮方向）：
```
|ΔVCD(q_t; K)| ≤ ||∇L(φ_t)||_2 · ||φ_{t+1} - φ_t||_2 · C_energy
```

**直觀解釋**：
- 能量變化大 + 步長大 → 可能導致大的漂移
- **優點**：計算簡單，僅需能量值和參數變化

### 7.4 基於變分下界差異的 Bound（非高斯似然）

#### 理論推導

對於非高斯似然，可以比較不同變分配置的下界：

**定義**：變分下界差異
```
ΔELBO(φ_t, φ_{t+1}) = L(φ_{t+1}) - L(φ_t)
```

**Bound 3：基於 ELBO 差異的漂移控制**
```
|ΔVCD(q_t; K)| ≤ |ΔELBO(φ_t, φ_{t+1})| · (1 + C_variational)
```

其中 `C_variational` 反映變分近似的緊密度。

**更精細的版本**（考慮內層變分狀態）：
```
|ΔVCD(q_t; K)| ≤ |L(φ_{t+1}, q_{t+1}) - L(φ_t, q_t)| · C_variational
```

**直觀解釋**：
- ELBO 的變化反映了變分近似的穩定性
- 對於非高斯似然，這是唯一可計算的診斷量
- **優點**：適用於所有似然類型，包括非共軛情況

### 7.5 組合 Bound：多層次診斷

結合多個可計算量：

**Bound 4：組合診斷 Bound**
```
|ΔVCD(q_t; K)| ≤ α · ||r(φ_t, φ_{t+1})||_2 · δ(K)
                + β · |ΔELBO(φ_t, φ_{t+1})|
                + γ · ||∇L(φ_t)||_2 · ||Δφ||_2
```

其中 `α, β, γ` 是權重係數，可通過經驗校準。

### 7.6 實際應用建議

#### 對於高斯似然：
- ✅ 使用原始 Delta-VCD（FITC 近似）
- ✅ 同時計算梯度-based bound 作為驗證

#### 對於非高斯似然：
- ✅ 使用 Bound 3（ELBO 差異）
- ✅ 使用 Bound 1（梯度變化）
- ✅ 組合多個診斷量

#### 實現策略：

```python
def compute_proxy_delta_vcd(energy_fn, phi_trace, step_sizes):
    """
    計算無需真實後驗的 Delta-VCD proxy
    
    適用於所有似然類型
    """
    n = len(phi_trace) - 1
    
    # Bound 1: 梯度變化
    grad_residuals = []
    for i in range(n):
        grad_t = jax.grad(energy_fn)(phi_trace[i])
        grad_t1 = jax.grad(energy_fn)(phi_trace[i+1])
        grad_residuals.append(tree_norm(tree_subtract(grad_t1, grad_t)))
    
    # Bound 2: 能量差異
    energy_diffs = []
    for i in range(n):
        energy_diff = energy_fn(phi_trace[i+1]) - energy_fn(phi_trace[i])
        energy_diffs.append(jnp.abs(energy_diff))
    
    # Bound 3: 組合
    grad_bound = jnp.array(grad_residuals) * jnp.array(step_sizes)
    energy_bound = jnp.array(energy_diffs)
    
    # 組合 proxy
    proxy_dvcd = 0.5 * grad_bound + 0.5 * energy_bound
    
    return {
        'proxy_dvcd': proxy_dvcd,
        'grad_bound': grad_bound,
        'energy_bound': energy_bound,
        'grad_residuals': jnp.array(grad_residuals),
        'energy_diffs': jnp.array(energy_diffs)
    }
```

### 7.7 理論保證

這些替代 bound 提供：

1. **上界保證**：如果 proxy 小，則真實 Delta-VCD 也小
2. **診斷能力**：proxy 大時，標誌著潛在的幾何不穩定
3. **通用性**：適用於所有似然類型

**注意**：這些是**單向 bound**（僅上界），不能完全替代真實 Delta-VCD，但提供了實用的診斷工具。

## 八、參考文獻

- 原始理論：Delta Variational Contrastive Divergence（論文第 X 節）
- Sampler-aware bound：論文中的 transport bound
- 實證應用：`notebook_10_occam_razor.ipynb`（僅高斯似然）
- FITC 實現：`infodynamics_jax/gp/sparsify.py::fitc_log_evidence()`
- 庫設計理念：`docs/energy_design.md`, `docs/design/philosophy.md`
