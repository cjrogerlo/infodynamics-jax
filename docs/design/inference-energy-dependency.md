# Inference 層對 Energy 層的依賴

## 當前狀況

### Inference 層知道 Energy 嗎？

**是的，但這是設計的一部分，而且是合理的。**

### 依賴關係

1. **Protocol 層次**：
   - `inference/base.py` 定義了 `InferenceMethod` protocol
   - 所有 inference 方法都必須接受 `energy: EnergyTerm`
   - 這是**接口依賴**，不是實現依賴

2. **具體實現**：
   - 所有 inference 方法都導入 `EnergyTerm`：
     - `sampling/hmc.py`, `mala.py`, `nuts.py`, `slice.py`
     - `optimisation/map2.py`, `vga.py`
     - `particle/annealed.py`, `ibis.py`
   - 但**只使用 `EnergyTerm` protocol**，不依賴具體實現

### 設計原則（從 `base.py`）

```python
class InferenceMethod(Protocol):
    """
    - An InferenceMethod consumes an EnergyTerm and performs inference.
    - It MUST treat EnergyTerm as a black box (no inspection of internal structure).
    - It MUST NOT assume any specific energy signature beyond EnergyTerm contract.
    """
```

## 這是合理的設計嗎？

### ✅ 是的，這是合理的

**理由**：

1. **依賴注入模式**：
   - Inference 需要知道如何調用 energy（通過 protocol）
   - 但不應該知道 energy 的內部實現
   - 這是標準的依賴注入模式

2. **分層清晰**：
   - `energy/`: 定義 energy landscape
   - `inference/`: 定義 dynamics（需要知道如何調用 energy）
   - `infodynamics/`: 組合兩者

3. **符合設計原則**：
   - Inference 只依賴 `EnergyTerm` protocol（接口）
   - 不依賴具體實現（如 `InertialEnergy`, `TargetEnergy`）
   - 這是**接口隔離原則**的正確應用

### ❌ 不應該有的依賴

檢查結果顯示，inference 層**沒有**以下不當依賴：
- ❌ 沒有直接導入 `InertialEnergy`
- ❌ 沒有直接導入 `TargetEnergy`
- ❌ 沒有直接導入 `PriorEnergy`
- ❌ 沒有檢查 energy 的內部結構

**所有 inference 方法都只使用 `EnergyTerm` protocol**。

## 對比：如果 Inference 不知道 Energy

### 選項 1: Inference 不知道 Energy（不推薦）

```python
# 如果 inference 不知道 energy
class InferenceMethod(Protocol):
    def run(self, objective_fn: Callable, *args, **kwargs) -> Any:
        # 使用通用的 Callable，而不是 EnergyTerm
        ...
```

**問題**：
- ❌ 失去類型安全
- ❌ 無法表達「這是 energy」的語義
- ❌ 無法在 protocol 層次定義契約

### 選項 2: Inference 知道 Energy Protocol（當前設計，推薦）

```python
# 當前設計
class InferenceMethod(Protocol):
    def run(self, energy: EnergyTerm, *args, **kwargs) -> Any:
        # 明確表達：這是 energy，不是任意函數
        ...
```

**優點**：
- ✅ 類型安全
- ✅ 語義清晰（這是 energy，不是任意函數）
- ✅ 可以在 protocol 層次定義契約
- ✅ 仍然保持抽象（不依賴具體實現）

## 結論

### 當前設計是正確的

1. **Inference 知道 Energy Protocol**：
   - ✅ 這是必要的（需要知道如何調用）
   - ✅ 這是接口依賴，不是實現依賴
   - ✅ 符合依賴注入模式

2. **Inference 不知道 Energy 實現**：
   - ✅ 不依賴 `InertialEnergy`, `TargetEnergy` 等具體類
   - ✅ 只使用 `EnergyTerm` protocol
   - ✅ 將 energy 視為黑盒

3. **分層清晰**：
   - `energy/`: 定義 energy landscape（實現）
   - `inference/`: 定義 dynamics（使用 energy protocol）
   - `infodynamics/`: 組合兩者（orchestration）

### 設計原則總結

```
energy/          → 定義 EnergyTerm protocol + 實現
     ↓ (protocol dependency)
inference/       → 使用 EnergyTerm protocol（黑盒）
     ↓ (composition)
infodynamics/    → 組合 energy + inference
```

這是**正確的分層架構**：
- 上層可以依賴下層的**接口**（protocol）
- 上層不應該依賴下層的**實現**（具體類）

## 檢查結果

所有 inference 方法都：
- ✅ 只導入 `EnergyTerm`（protocol）
- ✅ 不導入具體實現（如 `InertialEnergy`, `TargetEnergy`）
- ✅ 將 energy 視為黑盒（只調用，不檢查內部）

**結論：當前設計是正確的，不需要修改。**
