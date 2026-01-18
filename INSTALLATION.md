# 安裝指南 (Installation Guide)

## 問題說明

目前 `infodynamics_jax` 包可以從源代碼目錄直接導入，但沒有以開發模式安裝。這會導致以下問題：

1. **Notebook 需要手動設置路徑**：需要在 notebook 第一個 cell 手動添加 `sys.path`
2. **依賴項可能缺失**：某些依賴項可能沒有安裝
3. **Jupyter Kernel 可能找不到包**：不同的 kernel 可能使用不同的 Python 環境

## 解決方案

### 方法 1: 開發模式安裝（推薦）

這是最推薦的方法，包會以「可編輯」模式安裝，修改源代碼後無需重新安裝：

```bash
# 進入項目目錄
cd /Users/cjrogerlo/infodynamics-jax

# 以開發模式安裝
pip install -e .

# 如果需要 notebook 相關依賴
pip install -e ".[examples]"
```

安裝後，包會被安裝到當前 Python 環境中，所有 notebook 都可以直接 `import infodynamics_jax`。

### 方法 2: 檢查當前環境

確認當前 Python 環境和 Jupyter kernel 是否一致：

```bash
# 檢查 Python 位置
which python
python --version

# 檢查 Jupyter kernel 使用的 Python
jupyter kernelspec list

# 如果 kernel 和 Python 環境不一致，安裝 kernel
python -m ipykernel install --user --name infodynamics --display-name "Python (infodynamics)"
```

### 方法 3: 如果方法 1 不可用，保持當前手動設置方式

如果無法安裝（例如權限問題），可以保持當前方式，但確保 notebook 的第一個 cell 正確設置：

```python
import sys
from pathlib import Path

cwd = Path.cwd().resolve()
candidates = [cwd, *cwd.parents]

for p in candidates:
    if (p / 'infodynamics_jax').is_dir():
        sys.path.insert(0, str(p))
        sys.path.insert(0, str(p / 'examples'))
        sys.path.insert(0, str(p / 'examples' / 'utils'))
        break
```

## 檢查安裝狀態

安裝後，可以用以下命令檢查：

```python
import infodynamics_jax
print(infodynamics_jax.__file__)  # 應該顯示安裝路徑，而不是源代碼路徑

# 測試導入主要模塊
from infodynamics_jax.core import Phi
from infodynamics_jax.gp.kernels import rbf
print("安裝成功！")
```

## 故障排除

### 問題：pip install 權限錯誤

**解決方案**：
```bash
# 使用 --user 安裝到用戶目錄
pip install --user -e .

# 或使用虛擬環境（推薦）
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows
pip install -e .
```

### 問題：Jupyter 仍找不到包

**解決方案**：
1. 確認 Jupyter 和 Python 使用相同的環境：
   ```bash
   # 檢查 Jupyter 的 Python
   jupyter --paths
   
   # 在當前環境安裝 Jupyter
   pip install jupyter ipykernel
   ```

2. 重新啟動 Jupyter：
   ```bash
   jupyter notebook
   # 或
   jupyter lab
   ```

### 問題：導入錯誤

如果出現導入錯誤，檢查：
1. 包是否正確安裝：`python -c "import infodynamics_jax; print(infodynamics_jax.__file__)"`
2. 依賴項是否安裝：`pip list | grep jax`
3. Python 版本是否 >= 3.10：`python --version`

## 當前狀態

根據檢查：
- **Python 環境**: `/Users/cjrogerlo/miniforge3/bin/python`
- **包位置**: `/Users/cjrogerlo/infodynamics-jax/infodynamics_jax/` (源代碼目錄)
- **狀態**: 可以導入，但未正式安裝

**建議操作**：
```bash
pip install -e .
```