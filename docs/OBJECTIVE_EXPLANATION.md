# MAP-II 優化的目標函數（Objective）

## 核心目標函數

在 `infodynamics-jax` 中，MAP-II 優化使用的目標函數是 **Inertial Energy**：

```
E(φ) = E_{q(f | φ)}[-log p(y | f, φ)]
```

其中：
- `φ` 是結構參數（kernel hyperparameters, inducing points, noise variance 等）
- `q(f | φ)` 是在 sparsified kernel `S_ff = Q + R` 下的後驗分布
- 期望是對 `q(f | φ)` 取的，但對數在期望**內部**

## 與傳統 Marginal Likelihood 的區別

### 傳統方法（標準 GP）
```
log p(y | φ) = log ∫ p(y | f) p(f | φ) df
```
這是 **marginal likelihood**（邊際似然），對數在積分**外部**。

### infodynamics-jax 的方法
```
E(φ) = E_{q(f | φ)}[-log p(y | f, φ)]
```
這是 **inertial energy**，對數在期望**內部**。

## 對於 Gaussian Likelihood 的具體形式

當使用 Gaussian likelihood 時，對於每個數據點 i：

```
E_i = 0.5 * [log(2πσ²) + ((y_i - μ_i)² + var_i) / σ²]
```

其中：
- `μ_i = E_{q*(f_i | φ)}[f_i]` 是後驗均值
- `var_i = Var_{q*(f_i | φ)}[f_i]` 是後驗方差
- `σ²` 是觀測 noise variance

總能量是對所有數據點求和：
```
E(φ) = Σ_i E_i
```

## 計算方式

### 1. 計算 Sparsified Kernel
```
S_ff = Q + R
```
其中：
- `Q = K_xz @ K_zz^{-1} @ K_xz^T` (Nyström 近似)
- `R = diag(K_xx - diag(Q))` (FITC residual)

### 2. 計算後驗分布
在 sparsified kernel 下，後驗是：
```
μ(φ) = S_ff (S_ff + σ²I)^{-1} y
Σ(φ) = S_ff - S_ff (S_ff + σ²I)^{-1} S_ff
```

### 3. 計算 Energy
```
E(φ) = Σ_i 0.5 * [log(2πσ²) + ((y_i - μ_i)² + var_i) / σ²]
```

## 為什麼使用這個目標函數？

根據 `docs/energy_design.md`，這個設計是基於 **inference-as-dynamics** 的哲學：

1. **不進行 marginal likelihood 估計**
2. **不進行 free-energy 估計**
3. **不進行 evidence-based 模型比較**

這個 energy 是為了：
- 支持非共軛 likelihood（通過 Rao-Blackwellisation）
- 保持 inference abstraction 的一致性
- 支持動態推理框架

## 優化過程

MAP-II 優化器最小化這個 energy：
```
φ* = argmin_φ E(φ)
```

使用梯度下降（Adam/SGD）來優化：
- `lengthscale`
- `variance`
- `noise_var`
- `Z` (inducing points)

## 與傳統 GP 的關係

雖然目標函數不同，但對於 Gaussian likelihood：
- 當使用完整的 kernel（非 sparse）時，結果應該接近傳統的 log marginal likelihood
- 當使用 sparse approximation 時，這是對傳統方法的近似

## 參考

- `infodynamics_jax/energy/inertial.py`: InertialEnergy 實現
- `docs/energy_design.md`: Energy 層設計哲學
- `infodynamics_jax/energy/inertial.py:_gaussian_collapsed_energy()`: Gaussian 情況的具體計算
