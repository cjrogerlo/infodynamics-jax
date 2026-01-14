# infodynamics-jax 檔案樹結構

```
infodynamics-jax/
├── LICENSE                          # Apache 2.0
├── README.md                        # 專案說明（含定位宣言 + IP Notice）
├── QUICKSTART.md                    # 快速入門
├── requirements.txt                 # 依賴套件
├── setup.py                         # 安裝設定
│
├── docs/                            # 文件
│   ├── contributing_energy.md
│   ├── energy_design.md
│   ├── OBJECTIVE_EXPLANATION.md
│   ├── design/                      # 設計文件
│   └── theory/                      # 理論文件
│
├── examples/                        # 範例 notebooks 與腳本
│   ├── notebook_01_basic_regression.ipynb
│   ├── notebook_02_different_kernels.ipynb
│   ├── notebook_03_classification.ipynb
│   ├── notebook_04_annealed_smc.ipynb
│   ├── notebook_05_rjmcmc.ipynb
│   ├── notebook_06_rjmcmc_minimal.py
│   ├── notebook_07_smc_vfe_phase_diagram.ipynb
│   ├── particleFeynman-0.ipynb
│   ├── particleFeynman-0.py
│   ├── particleFeynman.ipynb
│   └── utils/                       # 範例工具
│       ├── __init__.py
│       ├── plotting_style.py
│       ├── plotting_utils.py
│       ├── smc_array_only.py
│       └── synthetic_functions.py
│
├── infodynamics_jax/                # 核心 library
│   ├── __init__.py
│   │
│   ├── core/                        # 核心資料結構
│   │   ├── __init__.py
│   │   ├── data.py                  # SupervisedData, LatentData
│   │   ├── phi.py                   # Phi (hyperparameters)
│   │   └── typing.py                # 型別定義
│   │
│   ├── energy/                      # Energy 層（primitives）
│   │   ├── __init__.py
│   │   ├── base.py                  # EnergyTerm protocol
│   │   ├── compose.py               # Energy 組合（Sum, Weighted, Target）
│   │   ├── inertial.py              # InertialEnergy
│   │   ├── prior.py                 # PriorEnergy
│   │   └── vfe.py                   # VFEEnergy
│   │
│   ├── gp/                          # GP 組件（primitives）
│   │   ├── __init__.py
│   │   ├── ansatz/                  # 估計器（GH, MC, Expected）
│   │   │   ├── __init__.py
│   │   │   ├── expected.py
│   │   │   ├── gh.py                # Gauss-Hermite
│   │   │   ├── mc.py                # Monte Carlo
│   │   │   ├── object.py
│   │   │   └── state.py
│   │   ├── kernels/                 # Kernel 函數（primitives）
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── params.py            # KernelParams
│   │   │   ├── rbf.py
│   │   │   ├── matern12.py
│   │   │   ├── matern32.py
│   │   │   ├── matern52.py
│   │   │   ├── periodic.py
│   │   │   ├── linear.py
│   │   │   ├── polynomial.py
│   │   │   ├── rational_quadratic.py
│   │   │   ├── white.py
│   │   │   ├── composite.py
│   │   │   └── utils.py
│   │   ├── likelihoods/             # Likelihood 函數（primitives）
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── gaussian.py
│   │   │   ├── bernoulli.py
│   │   │   ├── poisson.py
│   │   │   ├── negative_binomial.py
│   │   │   └── ordinal.py
│   │   ├── predict.py               # 預測函數
│   │   ├── sparsify.py              # 稀疏化工具
│   │   └── utils.py                 # GP 工具函數
│   │
│   ├── inference/                   # Inference 方法（primitives）
│   │   ├── __init__.py
│   │   ├── base.py                  # InferenceMethod protocol
│   │   │
│   │   ├── optimisation/             # 最佳化方法
│   │   │   ├── __init__.py
│   │   │   ├── typeii.py            # MAP-II (MAP2)
│   │   │   ├── vfe.py               # VFE 最佳化
│   │   │   └── vga.py               # VGA
│   │   │
│   │   ├── particle/                 # Particle 方法
│   │   │   ├── __init__.py
│   │   │   ├── annealed.py          # Annealed SMC
│   │   │   ├── ibis.py              # IBIS
│   │   │   ├── rejuvenation.py      # Rejuvenation kernels
│   │   │   ├── resampling.py        # Resampling
│   │   │   └── schedules.py         # Annealing schedules
│   │   │
│   │   ├── rj/                       # Reversible Jump
│   │   │   ├── __init__.py
│   │   │   ├── rjmcmc.py            # RJMCMC
│   │   │   ├── rjvmc.py             # RJVMC
│   │   │   └── state.py
│   │   │
│   │   └── sampling/                 # MCMC 採樣
│   │       ├── __init__.py
│   │       ├── hmc.py               # HMC
│   │       ├── nuts.py              # NUTS
│   │       ├── mala.py              # MALA
│   │       └── slice.py             # Slice Sampling
│   │
│   └── infodynamics/                # Orchestration 層
│       ├── __init__.py
│       ├── runner.py                # run() 函數（algorithm-agnostic）
│       └── hyperprior.py            # Hyperprior 工具
│
├── tests/                           # 測試
└── infodynamics_jax.egg-info/       # 安裝資訊
```
