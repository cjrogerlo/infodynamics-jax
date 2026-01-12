# Hyperprior 不是 Energy

## 問題

Hyperprior 不應該是 `EnergyTerm`，因為它不符合 energy 的定義。

## Energy 的定義

根據 `docs/energy_design.md`：

```
E(phi) = E_{q(f | phi)}[ -log p(y | f, phi) ]
```

這是 **inertial energy**，是數據相關的。

## Hyperprior 是什麼？

Hyperprior 是 `-log p(phi)`，即對 hyperparameters 的 prior。

這**不是** energy，因為：
- Energy 是數據相關的：`E[-log p(y|f,phi)]`
- Hyperprior 是數據無關的：`-log p(phi)`

## 設計問題

### 當前設計（錯誤）

```python
class HyperpriorEnergy(EnergyTerm):  # ❌ 不應該繼承 EnergyTerm
    ...
```

**問題**：
- Hyperprior 不是 energy
- 違反了 energy layer 的設計原則
- 混淆了概念層次

### 正確的設計

Hyperprior 應該：
1. **不是 EnergyTerm**（不符合 energy 定義）
2. **可以作為 term 添加到 TargetEnergy**（通過組合）
3. **或者應該在 inference layer 處理**（MAP-II 可以添加 hyperprior）

## 解決方案

### 選項 1: 作為獨立的 Term（推薦）

```python
# 不繼承 EnergyTerm，但可以作為 term 使用
class HyperpriorTerm:
    """Hyperprior on φ (not an energy, but a regularization term)."""
    def __call__(self, phi, X=None, Y=None, key=None):
        return -log p(phi)  # 或 log p(phi) 的負值
```

然後在 `TargetEnergy` 中：
```python
@dataclass(frozen=True)
class TargetEnergy(EnergyTerm):
    inertial: EnergyTerm
    prior: Optional[EnergyTerm] = None  # Prior on X
    hyperprior: Optional[Callable] = None  # -log p(phi)，不是 EnergyTerm
    extra: Optional[Sequence[EnergyTerm]] = None
```

### 選項 2: 在 Inference Layer 處理

```python
# map2.py 可以接受 hyperprior 作為額外參數
def run(
    self,
    energy: EnergyTerm,
    phi_init,
    *,
    hyperprior: Optional[Callable] = None,  # -log p(phi)
    ...
):
    def objective(phi):
        E = energy(phi, *energy_args, **energy_kwargs)
        if hyperprior is not None:
            E = E + hyperprior(phi)
        return E
    ...
```

### 選項 3: 通過 `extra` 參數（當前可行）

```python
# 不叫 HyperpriorEnergy，而是作為普通函數
def hyperprior_term(phi, X, Y, key=None):
    return -log p(phi)

target = TargetEnergy(
    inertial=inertial_energy,
    extra=[hyperprior_term],  # 作為 extra term
)
```

## 推薦方案

**選項 1 + 選項 3 的混合**：

1. **不創建 `HyperpriorEnergy` 類**（違反設計原則）
2. **提供工具函數**（不是 EnergyTerm）
3. **通過 `TargetEnergy.extra` 或 `hyperprior` 參數添加**

```python
# 工具函數（不是 EnergyTerm）
def hyperprior_l2(phi, fields=None, lam=1.0):
    """L2 hyperprior on kernel params (not an energy)."""
    ...

# 使用
target = TargetEnergy(
    inertial=inertial_energy,
    extra=[lambda phi, X, Y, key=None: hyperprior_l2(phi)],
)
```

或者：

```python
# 在 inference layer 處理
map2.run(
    energy=target_energy,
    phi_init=phi_init,
    hyperprior=lambda phi: hyperprior_l2(phi),  # 在 inference layer 添加
    ...
)
```

## 關鍵區別

| 概念 | 定義 | 是否 Energy |
|------|------|------------|
| Inertial Energy | `E[-log p(y|f,phi)]` | ✅ Yes |
| Prior on X | `-log p(X)` | ⚠️ 可以是 EnergyTerm（但只依賴 X） |
| Hyperprior | `-log p(phi)` | ❌ No（數據無關，不是 energy） |

## 結論

Hyperprior **不應該是 EnergyTerm**，因為：
1. 它不符合 energy 的定義（不是 `E[-log p(y|f,phi)]`）
2. 它是數據無關的（energy 是數據相關的）
3. 它應該在 inference layer 或作為組合 term 處理

**建議**：
- 移除 `HyperpriorEnergy` 類
- 提供工具函數（不是 EnergyTerm）
- 通過 `TargetEnergy.extra` 或 inference layer 添加
