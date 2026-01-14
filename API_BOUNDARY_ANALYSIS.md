# infodynamics-jax API 邊界分析：公開安全 vs 建議保留

## 分析原則

根據你提供的策略：
- **公開安全** = 數學上明確的 primitives（energy, kernel, likelihood, inference dynamics）
- **建議保留** = system-level configuration, auto-tuning, domain-specific recipes, end-to-end pipelines

---

## ✅ 公開安全（Primitives - 可公開）

### 1. Core 層（資料結構）
- ✅ `core/data.py`: `SupervisedData`, `LatentData` - 純資料容器
- ✅ `core/phi.py`: `Phi` - 超參數結構
- ✅ `core/typing.py`: 型別定義

**理由**：這些是數學上明確的資料結構，沒有系統設計邏輯。

---

### 2. Energy 層（能量函數）
- ✅ `energy/base.py`: `EnergyTerm` protocol
- ✅ `energy/inertial.py`: `InertialEnergy` - 核心能量定義
- ✅ `energy/prior.py`: `PriorEnergy`
- ✅ `energy/compose.py`: `TargetEnergy`, `SumEnergy`, `WeightedEnergy` - 組合邏輯
- ✅ `energy/vfe.py`: `VFEEnergy` - VFE 目標函數

**理由**：這些是數學上明確的能量函數，符合「inference as infodynamics」的理論框架。

---

### 3. GP 組件（Kernels, Likelihoods, Ansatz）
- ✅ `gp/kernels/`: 所有 kernel 函數（RBF, Matérn, Periodic, etc.）
- ✅ `gp/kernels/params.py`: `KernelParams` - 參數結構
- ✅ `gp/likelihoods/`: 所有 likelihood 函數（Gaussian, Bernoulli, Poisson, etc.）
- ✅ `gp/ansatz/`: 估計器（GH, MC, Expected） - 這些是數學方法
- ✅ `gp/predict.py`: `predict_typeii` - 預測函數（標準 GP 預測）

**理由**：這些是 textbook-level 的 GP 組件，沒有系統級配置。

---

### 4. Inference 方法（Dynamics）
- ✅ `inference/base.py`: `InferenceMethod` protocol
- ✅ `inference/optimisation/`: `MAP2`, `VGA`, `VFE` - 最佳化方法
- ✅ `inference/sampling/`: `HMC`, `NUTS`, `MALA`, `SliceSampler` - MCMC 方法
- ✅ `inference/particle/`: `AnnealedSMC`, `IBIS` - Particle 方法
- ✅ `inference/particle/resampling.py`: Resampling 演算法
- ✅ `inference/particle/rejuvenation.py`: Rejuvenation kernels
- ✅ `inference/particle/schedules.py`: Annealing schedules（數學定義）
- ✅ `inference/rj/`: `RJMCMC`, `RJVMC` - Reversible Jump 方法

**理由**：這些是標準的 inference dynamics，沒有 domain-specific 配置。

---

### 5. Orchestration 層（Algorithm-agnostic）
- ✅ `infodynamics/runner.py`: `run()` 函數 - **這是 algorithm-agnostic 的組合器**
- ✅ `infodynamics/hyperprior.py`: Hyperprior 工具函數（L2, log-L2 priors）

**理由**：
- `run()` 只是把 `energy + method` 組合起來，沒有假設特定演算法
- Hyperprior 是標準的正則化工具，不是系統配置

---

### 6. GP 工具（標準工具）
- ✅ `gp/sparsify.py`: `SparsifiedKernel` - FITC 稀疏化（標準方法）
- ✅ `gp/utils.py`: 通用 GP 工具函數

**理由**：這些是標準的 GP 工具，沒有系統級優化。

**註**：`compute_metrics()` 等評估工具實際在 `examples/utils/plotting_utils.py` 中，不在核心 library 中。

---

## ❌ 建議保留（System-level - 不應公開）

### 1. ~~`phase_diagram_pf.py`（根目錄）~~ ✅ 已刪除
**內容**：實驗腳本，包含 phase diagram 分析

**理由**：
- 這是 end-to-end 的實驗腳本
- 可能包含 domain-specific 的配置和最佳化
- 不是「primitives」

**狀態**：✅ **已刪除**

---

### 2. Examples 中的「最佳設定」說明
**檢查點**：
- ❌ 不要在 README 或 examples 中寫「最佳設定」
- ❌ 不要提供「auto-tuning」的範例
- ❌ 不要提供「latency benchmark」或「performance comparison」

**目前狀態**：需要檢查 `examples/` 下的 notebooks 是否有這些內容。

---

## 📊 總結：公開 vs 保留邊界圖

```
┌─────────────────────────────────────────────────────────┐
│                    PUBLIC API (安全)                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ✅ Core: Phi, SupervisedData, LatentData                 │
│  ✅ Energy: InertialEnergy, PriorEnergy, TargetEnergy     │
│  ✅ GP: Kernels, Likelihoods, Ansatz (GH, MC)            │
│  ✅ Inference: HMC, NUTS, MALA, SMC, IBIS, RJMCMC        │
│  ✅ Orchestration: run() (algorithm-agnostic)             │
│  ✅ Hyperprior: L2, log-L2 priors                        │
│  ✅ GP Tools: predict_typeii, sparsify                   │
│                                                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              PRIVATE (建議保留)                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ❌ phase_diagram_pf.py (實驗腳本)                        │
│  ❌ End-to-end pipelines (未來)                            │
│  ❌ Auto-tuning 邏輯 (未來)                                │
│  ❌ Domain-specific recipes (未來)                        │
│  ❌ Performance benchmarks (未來)                          │
│  ❌ 「最佳設定」說明 (未來)                                 │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 下一步建議

### 1. 立即行動
- [x] **刪除 `models/` 目錄**（已確認未被使用）
- [x] **刪除 `phase_diagram_pf.py`**（實驗腳本）
- [ ] **檢查 `examples/` notebooks** 是否有「最佳設定」或「auto-tuning」內容

### 2. 未來防護
- [ ] 建立明確的 **Public vs Private API 文件**（可選）
- [ ] 在 CI 中檢查是否有「不應該公開」的關鍵字（如 "best config", "auto-tune"）
- [ ] 考慮在 `infodynamics_jax/__init__.py` 中只 export primitives，不 export system-level 工具

### 3. 專利考量（未來）
根據你的策略，以下可能是「值得申請專利」的部分（如果有的話）：
- **System-level composition**：如何把 primitives 組合成高效系統
- **Auto-tuning 策略**：如何自動選擇 inference method + hyperparameters
- **Domain-specific recipes**：特定領域的 inference 配方

但這些都應該在 **private repo** 中，不在公開的 `infodynamics-jax`。

---

## 📝 標準回答模板

如果有人問：「為什麼不直接給一個完整 system？」

**回答**：
> Because system-level inference design is inherently domain-specific.
> infodynamics-jax focuses on providing explicit and reusable probabilistic primitives rather than prescribing a single system configuration.

---

## ✅ 結論

**目前的 `infodynamics-jax` 結構已經非常符合你的策略**：
- ✅ 所有 primitives 都是公開安全的
- ✅ `run()` 是 algorithm-agnostic 的，可以公開
- ⚠️ 只需要處理 `phase_diagram_pf.py` 這個實驗腳本
- ✅ 整體符合「trade secret by design」策略

**你現在的位置 = Secondmind 的開源策略**：
- ✅ 發 paper：大家可以用
- ✅ 開源 primitives：大家可以用
- ✅ 你保留的是「怎麼把它組成系統」→ 這在 private repo 中
