# Utils 設計哲學

## 當前 Utils

### 1. `inference/particle/resampling.py`
- `multinomial_resample`: SMC 核心需要的重採樣函數
- `effective_sample_size`: SMC 核心需要的 ESS 計算

**這些是 SMC 算法專用的核心工具，不是 problem-specific。**

### 2. `gp/kernels/utils.py`
- `scaled_sqdist`: Kernel 計算的內部工具函數

**這是模型層次需要的，不是 problem-specific。**

## 不應該添加的 Utils

### Plotting / Visualization
**理由**：
- ❌ 非常 problem-specific（不同問題需要不同的圖表）
- ❌ 增加依賴（matplotlib, seaborn 等）
- ❌ 違反「庫應該輕量」的原則
- ✅ 用戶可以在 `examples/` 或自己的代碼中使用外部庫

**建議**：
- 在 `examples/` 中展示如何使用 matplotlib/seaborn 進行可視化
- 不應該在核心庫中提供 plotting 工具

### Metrics / Evaluation
**理由**：
- ❌ 非常 problem-specific（不同問題需要不同的 metrics）
- ❌ 增加依賴（scikit-learn 等）
- ❌ 違反「庫應該專注於 inference」的原則
- ✅ 用戶可以在自己的代碼中計算 metrics

**建議**：
- 在 `examples/` 中展示如何計算常見 metrics（如 RMSE, accuracy 等）
- 不應該在核心庫中提供 metrics 工具

### Diagnostics 擴展
**當前設計**：
- `runner.py` 已經提供基本的 diagnostics（`accept_rate`, `energy_trace`, `grad_norm_trace`）
- 這些是**通用的、算法層次的** diagnostics

**不應該添加**：
- ❌ Problem-specific diagnostics（如 regression metrics, classification metrics）
- ❌ 複雜的診斷工具（這些應該在用戶代碼中）

## 設計原則

### 應該包含的 Utils
1. **算法核心需要的**（如 `multinomial_resample`, `effective_sample_size`）
2. **模型計算需要的**（如 `scaled_sqdist`）
3. **通用的、非 problem-specific 的**

### 不應該包含的 Utils
1. **Plotting / Visualization**（problem-specific）
2. **Metrics / Evaluation**（problem-specific）
3. **複雜的診斷工具**（應該在用戶代碼中）

## 建議的結構

```
infodynamics_jax/
  ├── inference/particle/resampling.py  ✅ SMC 核心工具
  ├── gp/kernels/utils.py               ✅ 模型計算工具
  └── (不應該有)
      ├── utils/plotting.py        ❌ 不應該
      ├── utils/metrics.py         ❌ 不應該
      └── utils/diagnostics.py     ❌ 不應該（已有 runner.py 的基本 diagnostics）

examples/
  ├── ibis_annealed_smc.py         ✅ 可以包含簡單的 plotting 示例
  └── (可以添加)
      ├── plot_results.py          ✅ 展示如何使用 matplotlib
      └── compute_metrics.py       ✅ 展示如何計算 metrics
```

## 結論

**當前設計是正確的**：
- ✅ 只包含算法/模型核心需要的 utils
- ✅ 保持庫的輕量和通用性
- ✅ 不包含 problem-specific 的工具

**如果用戶需要 plotting/metrics**：
- 在 `examples/` 中提供示例
- 或在自己的代碼中使用外部庫（matplotlib, seaborn, scikit-learn 等）

**這符合庫的設計原則**：
- 專注於 inference dynamics
- 不假設特定的問題類型
- 保持輕量和可組合
