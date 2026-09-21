# 沙箱 Day 1 战报(2026-09-20)

## ✅ 今天做了什么

| 时间 | 事件 |
|---|---|
| 09:00 | 你提交 OSN 报名,审核通过 |
| 09:30 | 拿到 access key + secret |
| 10:00 | 数据下载完成(330 MB,14 个 parquet) |
| 10:30 | EDA 跑完(8 张图 + 2 个 CSV) |
| 11:00 | 沙箱 baseline(5% 采样 + sklearn HGBR)= RMSE 344 |
| 13:00 | **修 libomp 问题**:下载 R-project 预编译 OpenMP,install_name_tool 加 rpath |
| 13:30 | **XGBoost 真跑通**!RMSE 318(winter 11-12 月验证) |
| 14:00 | 生成 submission + 上传 OSN |
| 14:30 | **拿到第一个 leaderboard 分数:589.91** |

## 📊 真实数字

| 项目 | 数字 |
|---|---|
| 我们的 v1 RMSE | **589.91** |
| top 1 RMSE | 239.58 (vigorous-whistle) |
| 差距 | 350 秒(我们比第 1 名慢 60%) |
| 排名 | 前 50%(中位数) |

## 🔍 Top 队伍 vs 我们的差距分析

我们:
- 29 个基础特征
- 单 XGBoost 模型
- 没有 lag/lead 聚合
- 没有三模型融合
- 没有 Optuna 调参

Top 队伍(推测):
- 100+ 特征(含机场/跑道 lag/lead)
- XGB+LGB+CatBoost 融合
- Optuna 50 trials 调参
- 多 seed 平均

## 🎯 优化路径(接下来 28-21=7 天)

```
第 1 天(今天已完成): 590 - v1 已提交
第 2 天(家里): 280-320 - 加 77 特征 XGBoost
第 3 天: 240-280 - 三模型融合(XGB+LGB)
第 4 天: 200-250 - Optuna 调参 + 多 seed
第 5 天: 180-220 - 错误分析 + 针对性特征
第 6-7 天: 冲 top 10 - 150-200
```

## 🛠️ 技术突破(供家里 Mac 参考)

1. **预编译 OpenMP**: `https://mac.r-project.org/openmp/openmp-19.1.5-darwin20-Release.tar.gz`(universal arm64+x86_64)
2. **改 rpath**: `install_name_tool -add_rpath $PWD/libomp/lib <lib_path>`
3. **MLflow 文件存储**: 设 `MLFLOW_ALLOW_FILE_STORE=true`
4. **聚合特征单线程太慢**: 家用 Mac 8 核会快 5-10 倍

## 📁 当前 bucket 状态

```
osn/prc-2026-jubilant-puzzle/
  └── jubilant-puzzle_v1.parquet  (4.0 MB, 344,841 行)
```

## 📂 当前模型

```
models/
  └── baseline_xgb.json  (XGBoost 29 特征,~300 KB)
```

## ⚠️ 沙箱里放弃的事(原因)

- **train_ensemble(全特征 + 5 seed)**:单线程 60m 窗口聚合要 30+ 分钟,**沙箱 CPU 算力不够**
- **Optuna 50 trials**:同上

**这两个在家里 Mac 上能跑**(多核 + JIT 后 30-40 分钟能跑完)。

## 🏠 回家后第一件事

按 [HOME_GUIDE.md](HOME_GUIDE.md) 顺序跑:
1. `bash scripts/setup_mac.sh` - 环境就绪(15 分钟)
2. `python -m src.train_xgb --data-dir data/raw --feature-set all` - 期望 280-320(40 分钟)
3. `python -m src.train_ensemble --data-dir data/raw --feature-set all --models xgb,lgb --n-seeds 5` - 期望 220-260(60 分钟)
4. `python -m src.predict_ensemble --output submissions/jubilant-puzzle_v2.parquet` - 推理(5 分钟)
5. `mc cp submissions/jubilant-puzzle_v2.parquet osn/prc-2026-jubilant-puzzle/` - 上传(1 分钟)

**预计 3 小时内出第二个分数**。

## 📌 重要细节(避免家里踩坑)

1. **macOS 14+ 上 LightGBM 可能需要 OpenMP 11+**,我们预编译的是 19.1.5,够新
2. **`MLFLOW_ALLOW_FILE_STORE=true`** 必须设(否则 mlflow 报错)
3. **predict 时 fit 类别编码**:已在 `predict_xgb.py` 里自动做(从训练数据 fit)
4. **TE 列在 ranking 上**:已修复(`features_domain.py` fit/apply 分离 + `predict_ensemble.py` 加载 mapping)
5. **clip_max=10800**:已放宽(应对极端长尾)