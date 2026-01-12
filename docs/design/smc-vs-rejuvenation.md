# SMC vs Rejuvenation: 是否應該合併？

## 當前狀況

### `resampling.py` (53 行)
- `multinomial_resample`: SMC 核心操作（重採樣）
- `effective_sample_size`: SMC 核心操作（ESS 計算）
- **用途**：SMC 算法**必需**的操作

### `rejuvenation.py` (448 行)
- `hmc_rejuvenate`, `mala_rejuvenate`, `nuts_rejuvenate`: Rejuvenation kernels
- `kinetic_energy`: 輔助函數
- **用途**：SMC 的**可選**步驟（rejuvenation）

## 合併的考慮

### 優點

1. **統一管理**：
   - 所有 particle 相關的工具都在一個地方
   - 減少文件數量

2. **相關性**：
   - 兩者都屬於 SMC/particle methods
   - 都在 `inference/particle/` 目錄下

### 缺點

1. **文件大小**：
   - 合併後會超過 500 行（53 + 448 = 501 行）
   - 可能過長，不利於維護

2. **概念差異**：
   - `resampling.py`: SMC **核心**操作（重採樣、ESS）
   - `rejuvenation.py`: SMC **可選**步驟（rejuvenation）
   - 概念上略有不同

3. **可重用性**：
   - Rejuvenation kernels 理論上可以被其他方法使用
   - 雖然目前只在 particle methods 中使用

4. **職責分離**：
   - `resampling.py`: 純 SMC 操作（不依賴 energy）
   - `rejuvenation.py`: 依賴 energy（需要 energy_fn）

## 建議

### 選項 1: 保持分離（推薦）

**理由**：
- ✅ 概念清晰：核心 SMC 操作 vs 可選的 rejuvenation
- ✅ 文件大小合理：`resampling.py` (53 行) 和 `rejuvenation.py` (448 行)
- ✅ 職責分離：`resampling.py` 不依賴 energy，`rejuvenation.py` 依賴 energy
- ✅ 易於維護：每個文件專注於一個概念

**當前結構**：
```
inference/particle/
  ├── resampling.py   # 核心 SMC 操作（重採樣、ESS）
  ├── rejuvenation.py # 可選的 rejuvenation kernels
  ├── annealed.py     # β-annealed SMC
  └── ibis.py         # IBIS
```

### 選項 2: 合併為 `resampling.py`

**如果合併**：
- 文件會超過 500 行
- 混合了核心操作和可選步驟
- 但所有 particle 工具都在一個地方

## 結論

**建議保持分離**，因為：

1. **概念清晰**：
   - `resampling.py`: SMC 核心操作（必需）
   - `rejuvenation.py`: Rejuvenation kernels（可選）

2. **文件大小**：
   - 當前分離更易於維護
   - 合併後會超過 500 行

3. **職責分離**：
   - `resampling.py` 不依賴 energy
   - `rejuvenation.py` 依賴 energy

4. **未來擴展**：
   - 如果未來有其他方法需要 rejuvenation，可以重用
   - 保持分離更靈活

**當前設計是合理的，不需要合併。**
