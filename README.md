# PRC Data Challenge 2026 — Taxi-Out Time Prediction

> 团队:**jubilant-puzzle**
> 比赛:[PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/)
> 截止:**2026-10-11 23:59:59 CET**(还剩 23 天)
> 评分:RMSE(以秒计)

## 项目结构

```
prc_data_challenge/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖
├── .gitignore                # 忽略 data/、models/、mlruns/ 等大文件
│
├── data/
│   ├── raw/                  # 从 OSN bucket 下载的 parquet(不进 git)
│   └── .gitkeep
│
├── src/                      # 所有源代码
│   ├── data_loader.py        # 读取月度 parquet,合并,过滤 DEP
│   ├── features.py           # 基础特征工程(baseline 用)
│   ├── validation.py         # 时序切分 + 评估指标
│   ├── train_lgb.py          # LightGBM baseline 训练
│   └── predict.py            # 推理 + 生成提交 parquet
│
├── models/                   # 训练好的模型(不进 git)
│
├── submissions/              # 提交文件(不进 git,.gitkeep 占位)
│   └── .gitkeep
│
├── mlruns/                   # MLflow 实验记录(本地后端)
│
├── notebooks/                # EDA、临时分析
│
└── docs/
    └── superpowers/specs/    # 设计文档、计划
```

## 快速开始

### 1. 环境准备

需要 Python 3.10+。建议用虚拟环境:

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# macOS 上 LightGBM 需要 libomp(如果还没装)
brew install libomp

# 安装依赖
pip install -r requirements.txt
```

> **Smoke test**:在临时工作目录的 `smoke_test.py` 已用 sklearn 验证整个数据流能跑通。等装好 libomp 后,直接 `python -m src.train_lgb --data-dir data/raw` 即可。

### 2. 获取数据

报名组队后,OSN 团队会审核并发 access key。拿到后:

1. 登录 OSN 账号,在 Dashboard 生成 MinIO access key
2. 用 MinIO 客户端(或直接网页)下载:
   - `training_2025-*.parquet`(12 个,约 280 MB)
   - `ranking.parquet`(约 27 MB)
   - `submitting.parquet`(约 1.1 MB)
3. 放到 `data/raw/` 目录

**完整下载教程**(命令行 + GUI + Python 三种):[docs/MINIO_DOWNLOAD_GUIDE.md](docs/MINIO_DOWNLOAD_GUIDE.md)

详见数据页:https://prc-data-challenge-2026.netlify.app/data.html

### 3. 跑 Baseline

```bash
# 一键 EDA(探索数据,拿到数据后第一件事)
python -m scripts.eda --data-dir data/raw --output-dir notebooks/eda

# 训练 baseline(基础特征,XGBoost)
python -m src.train_xgb --data-dir data/raw --feature-set basic

# 训练 baseline(全特征:基础+拥堵+lead+domain+TE,XGBoost)
python -m src.train_xgb --data-dir data/raw --feature-set all

# 三模型融合(XGB + LGB + CatBoost,自动权重优化) - 推荐第一轮提交用
python -m src.train_ensemble --data-dir data/raw --feature-set all --models xgb,lgb,cat --n-seeds 5

# Optuna 自动调参(XGBoost)
python -m src.tune --data-dir data/raw --feature-set all --model-type xgb --n-trials 50

# 长跑用鲁棒脚本(自动重启,崩了接着跑)
python -m scripts.run_robust --train --feature-set all --models xgb,lgb,cat --n-seeds 5

# 推理 + 生成提交
python -m src.predict_xgb --model models/baseline_xgb.json --output submissions/v1.parquet
python -m src.predict_ensemble --output submissions/ensemble_v1.parquet
```

### 4. 查看实验记录(MLflow UI)

```bash
# 启动 MLflow UI
mlflow ui --backend-store-uri mlruns

# 然后浏览器打开 http://localhost:5000
```

可以看到:
- 每次实验的参数(learning_rate, num_leaves 等)
- 多切分验证的 RMSE / MAE / P95
- K 折时序 CV 的均值和方差
- 特征重要性 CSV
- 不同实验之间的对比

## 当前进度

| 阶段 | 状态 | 备注 |
|---|---|---|
| 数据接入 | ⏳ 等 OSN 审核 | 提交组队表单中 |
| Baseline (基础特征,XGBoost) | ✅ 代码就绪 | 拿到数据即可跑 |
| 拥堵聚合特征 (lag) | ✅ 代码就绪 | `src/features_aggregate.py`,18 个聚合特征 |
| Lead 特征 (未来 30 分钟) | ✅ 代码就绪 | `src/features_laglead.py`,18 个 lead 特征 |
| Domain 特征(登机口区/跑道/航司×机场) | ✅ 代码就绪 | `src/features_domain.py`,8 个 TE + 3 个结构特征 |
| EDA 脚本 | ✅ 代码就绪 | `scripts/eda.py` 一键生成所有图表 |
| 三模型融合 (XGB+LGB+CatBoost) | ✅ 代码就绪 | `src/train_ensemble.py`,多 seed + median/mean 自动选最优 |
| Optuna 自动调参 | ✅ 代码就绪 | `src/tune.py`,**50 trials 默认**(v2) |
| Median voting 融合 | ✅ 代码就绪 | `train_ensemble.py` 内置(从 2024 第 4 名学到) |
| 目标重缩放(按机场) | ✅ 代码就绪 | `src/features.py`,可选(从 2024 第 1 名学到) |
| 鲁棒性自动重启 | ✅ 代码就绪 | `scripts/run_robust.py`(从 2025 第 2 名学到) |
| 分段模型(长尾处理) | 📝 待开发 | 拿到 baseline 数字后再做 |
| 论文 draft | 📝 可选 | |

详细设计见 [设计文档](docs/superpowers/specs/2026-09-18-prc-data-challenge-2026-design.md)。

## 关键决策(为什么这样设计)

| 决策 | 选择 | 原因 |
|---|---|---|
| 硬件 | 笔记本为主,可租 GPU | 个人参赛,无固定服务器 |
| 建模方案 | 全局回归 + 分段处理长尾 | RMSE 指标对长尾放大,必须处理 |
| 特征层级 | 基础特征 + 拥堵聚合 | 拥堵聚合是最大提分点,聚合 lag 模拟排队 |
| 验证策略 | 多切分(winter + summer + kfold) | 时序预测不能随机切,贴近比赛真实场景 |
| 实验管理 | MLflow | 自动记录、可视化、便于发论文时复现 |

## 提交到 OSN

比赛每天 5 次提交机会,每 bucket 限 1 GB。

1. 用 MinIO 客户端把生成的 `submissions/baseline_v1.parquet` 上传到团队 bucket
2. 文件命名必须符合 `<team-name>_v<integer>.parquet`(如 `jubilant-puzzle_v1.parquet`)
3. 排名脚本会自动比对 `MVT_ID_mvt`,计算 RMSE

详细提交规则:https://prc-data-challenge-2026.netlify.app/ranking.html

## 参考资料

- 比赛主页:https://ansperformance.eu/study/data-challenge/dc2026/
- 数据字典:https://prc-data-challenge-2026.netlify.app/data.html
- 评分规则:https://prc-data-challenge-2026.netlify.app/ranking.html
- 往年方案(GitHub):https://github.com/prc-data-challenge-2025
- LightGBM 文档:https://lightgbm.readthedocs.io/
- MLflow 文档:https://mlflow.org/docs/latest/index.html

## License

本仓库代码以 MIT 协议开源(配合比赛要求)。
