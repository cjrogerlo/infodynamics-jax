# Particle Inference API 檢查報告

## 結論

✅ **所有檔案乾淨，權責清晰分離，API 設計一致**

---

## 權責分離

### 1. `annealed.py` - β-annealed SMC
**權責：**
- 實現 β-annealing（thermodynamic path）
- 固定數據集上的溫度退火
- Weight update: `Δlogw = -Δβ * E(φ)`

**不負責：**
- ❌ Data streaming
- ❌ IBIS logic
- ❌ SVGP-specific inference

### 2. `ibis.py` - IBIS (Iterated Batch Importance Sampling)
**權責：**
- 實現 data streaming（Bayesian filtering path）
- 處理數據流，更新 `p(φ | y_{1:t})`
- Weight update: `logw += log p(y_t | φ)`

**不負責：**
- ❌ β-annealing
- ❌ 固定數據集處理

### 3. `resampling.py` - SMC 核心工具函數
**權責：**
- `multinomial_resample`: 多項式重採樣
- `effective_sample_size`: ESS 計算

**設計原則：**
- 只包含純函數，無狀態
- 可被多個 particle 方法重用

---

## API 設計

### `InferenceMethod` Protocol 符合性

兩者都實現 `InferenceMethod` protocol，但 `run()` 簽名不同（這是允許的）：

#### `AnnealedSMC.run()`
```python
def run(
    self, 
    energy: EnergyTerm, 
    init_particles_fn: Callable[[jax.random.PRNGKey, int], Any], 
    *, 
    key, 
    energy_args=(), 
    energy_kwargs=None
) -> SMCRun
```

**特點：**
- 接受 `energy_args`（固定數據集）
- 不需要 `data_stream`
- 返回 `SMCRun`（包含 `betas`）

#### `IBIS.run()`
```python
def run(
    self,
    energy: EnergyTerm,
    init_particles_fn: Callable[[jax.random.PRNGKey, int], Any],
    data_stream: Union[Iterator[SupervisedData], list[SupervisedData]],
    *,
    key: jax.random.PRNGKey,
    energy_kwargs: Optional[dict] = None,
) -> IBISRun
```

**特點：**
- 接受 `data_stream`（數據流）
- 不需要 `energy_args`（數據在 stream 中）
- 返回 `IBISRun`（包含 `logZ_trace`, `time_steps`）

**設計合理性：**
- ✅ 簽名差異反映方法本質差異
- ✅ 都遵循 `InferenceMethod` protocol 的精神
- ✅ 都接受 `energy: EnergyTerm`（核心契約）

---

## 代碼重複處理

### ✅ 已提取到 `resampling.py`
- `multinomial_resample()`
- `effective_sample_size()`

### ✅ 保留差異（合理）
- `_hmc_kernel()` 在兩個文件中都有，但：
  - `annealed.py`: targets `β * E(φ)` (tempered)
  - `ibis.py`: targets `E(φ; y_{1:t})` (full posterior)
  - 實現差異是必要的，不應合併

---

## 導出檢查

### `inference/particle/__init__.py`
```python
from .annealed import AnnealedSMC, AnnealedSMCCFG, SMCRun
from .ibis import IBIS, IBISCFG, IBISRun

__all__ = [
    "AnnealedSMC", "AnnealedSMCCFG", "SMCRun",
    "IBIS", "IBISCFG", "IBISRun",
]
```
✅ 正確導出所有公共 API

### `inference/__init__.py`
```python
from .particle import (
    AnnealedSMC, AnnealedSMCCFG, SMCRun,
    IBIS, IBISCFG, IBISRun
)

__all__ = [
    ...
    "AnnealedSMC", "AnnealedSMCCFG", "SMCRun",
    "IBIS", "IBISCFG", "IBISRun",
]
```
✅ 正確導出到頂層

---

## 文檔完整性

### ✅ 模組級 docstring
- `annealed.py`: 明確說明不是 IBIS，不是 SVGP
- `ibis.py`: 明確說明與 β-annealing 的區別

### ✅ 類級 docstring
- 兩者都清楚說明：
  - 目標分布
  - Weight update 公式
  - 演化軸（thermodynamic vs data streaming）

### ✅ 方法級 docstring
- `run()` 方法都有完整的參數說明
- 關鍵差異都有註釋標註

---

## 潛在改進（可選）

### 1. HMC Kernel 共享（低優先級）
目前兩個 `_hmc_kernel` 實現相似但不同。可以考慮：
- 提取基礎 HMC 邏輯到共享函數
- 但保留不同的 energy function wrapper
- **建議：保持現狀**（差異是必要的）

### 2. 類型提示增強（可選）
- `data_stream` 可以更明確的類型
- 但當前 `Union[Iterator, list]` 已經足夠

---

## 總結

### ✅ 權責分離
- 兩者權責清晰，互不重疊
- 理論基礎不同（thermodynamic vs Bayesian filtering）

### ✅ API 設計
- 符合 `InferenceMethod` protocol
- 簽名差異反映方法本質
- 導出完整且一致

### ✅ 代碼質量
- 重複代碼已提取到 `resampling.py`
- 必要的差異保留（HMC kernel）
- 文檔完整清晰

### ✅ 架構一致性
- 都使用 `EnergyTerm` 作為輸入
- 都遵循相同的設計原則
- 與整體架構（energy/, core/, infodynamics/）一致

**結論：所有檔案乾淨，權責清晰，API 設計一致且合理。**
