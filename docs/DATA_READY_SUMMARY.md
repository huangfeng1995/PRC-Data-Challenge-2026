# 数据到位总结(2026-09-20)

**今天发生的:**
1. OSN 审核通过,队名确认:`jubilant-puzzle`
2. 数据下载完成(330 MB,14 个 parquet)
3. 跑完 EDA,发现数据真实特征
4. 修了 2 个数据驱动的 bug(过滤条件、endpoint)

---

## 📊 数据特征速览

| 项目 | 实际值 |
|---|---|
| 总行数 | 4,167,797 |
| **DEP 行数** | **2,085,047**(50%,符合预期) |
| 机场数 | **10 个**(不是 11) |
| 时间跨度 | 2025-01-01 ~ 2025-12-31 |
| 字段数 | 26(去重后) |

### 各机场 taxi-out 分布

| 机场 | 均值 | P99 | 数据量 |
|---|---|---|---|
| EGLL(伦敦) | 1364s | 2701s | 24万 |
| LIRF(罗马) | 1195s | 4019s(长尾最严重) | 16万 |
| LTFM(伊斯坦布尔) | 1053s | 2587s | 27万(最多) |
| LFPG(巴黎) | 1020s | 2409s | 24万 |
| LEMD(马德里) | 1015s | 1932s | 21万 |
| LEBL(巴塞罗那) | 957s | 1962s | 18万 |
| EDDF(法兰克福) | 864s | 1819s | 23万 |
| EDDM(慕尼黑) | 812s | 1921s | 17万 |
| EHAM(阿姆斯特丹) | 785s | 1775s | 25万 |
| LSZH(苏黎世) | **741s(最简单)** | 1707s | 13万 |

**关键洞察**:
- **EGLL 均值最高(22.7 分钟)** —— 希思罗是欧洲最拥堵机场之一,预测最难
- **LIRF 长尾最严重**(P99 是均值的 3.4 倍)—— 罗马有"长队堆积"现象
- 各机场差异巨大(均值 741 ~ 1364s,差近 2 倍)→ **"按机场分组训练"在 v2 提到的方案应该能进一步加分**

---

## 🐛 数据驱动的 bug 修复

| Bug | 原因 | 修复 |
|---|---|---|
| `TAXITIME_SEC_mvt < 0`(最小 -12s) | 数据 bug | 过滤改为 `> 0 and < 10800` |
| `TAXITIME_SEC_mvt` 最大 131167s(36 小时) | 数据 bug / 长尾 | 同样过滤掉 |
| EDA log 直方图 NaN | clip 后用 log1p 正常 | 已修 |
| `datacomp.opensky-network.org` endpoint 404 | 实际是 `s3.opensky-network.org` | 改默认 endpoint |

---

## 🆕 真实数据上的新发现

| 发现 | 影响 |
|---|---|
| 实际是 **10 个机场**(不是 11) | 进度表/文档可能要改;影响 rank/ranking 加载逻辑的健壮性 |
| `AIRCRAFT_OPERATOR_flt` 缺失率 1.08% | 我们的 TE 处理已 fallback 到 global_mean,不影响 |
| 同一航班 ID 有时多条 movement | 训练时要考虑时序顺序,数据加载已经按月排序 |
| TAXITIME_SEC_mvt 有负数和极端值 | 训练过滤已加强 |

---

## 🚀 回家后立刻能跑的命令

环境假设:你回家在自己的 Mac 上,数据已经下好(330 MB 在 `~/Documents/PRC_data_challenge/data/raw/`)。

```bash
cd ~/Documents/PRC_data_challenge

# 1. 最简 baseline(基础特征,XGBoost)—— 看起点 RMSE
python -m src.train_xgb --data-dir data/raw --feature-set basic

# 2. 全特征 baseline(基础+拥堵+lead+domain+TE,XGBoost)
python -m src.train_xgb --data-dir data/raw --feature-set all

# 3. 三模型融合(XGB + LGB + CatBoost,5 seed)
python -m src.train_ensemble --data-dir data/raw --feature-set all \
    --models xgb,lgb,cat --n-seeds 5

# 4. 调参(25% 数据,约 30 分钟)
python -m src.tune --data-dir data/raw --feature-set all --n-trials 50

# 5. 推理 + 生成提交
python -m src.predict_ensemble --output submissions/jubilant-puzzle_v1.parquet

# 6. 上传到 OSN
mc cp submissions/jubilant-puzzle_v1.parquet \
    osn/prc-2026-jubilant-puzzle/
```

---

## 📋 Mac 环境要求

我们的代码依赖:`xgboost`, `lightgbm`, `catboost`, `scipy`, `mlflow`, `optuna`, `pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`, `minio`

**macOS 上需要 libomp**(否则 XGBoost/LightGBM 会 crash):
```bash
brew install libomp
```

装好后跑 baseline。

---

## 📦 Bucket 信息(给 mc 用)

| 项 | 值 |
|---|---|
| Endpoint | `https://s3.opensky-network.org` |
| **数据 bucket** | `prc-2026-datasets`(所有团队共享)|
| **你的提交 bucket** | `prc-2026-jubilant-puzzle` |

**配置 mc**:
```bash
mc alias set osn https://s3.opensky-network.org <ACCESS_KEY> <SECRET_KEY>
```

---

## 🎯 接下来 21 天的执行路径

| Day | 任务 | 预计 RMSE |
|---|---|---|
| 4 | 跑 baseline (basic → all) | 250-300 |
| 5 | 跑三模型 5-seed 融合 | 200-250 |
| 6 | **第一次提交 jubilant-puzzle_v1** | 进入 leaderboard |
| 7-8 | Optuna 调参 | 180-220 |
| 11-12 | 看误差分析,加 lag/lead 优化 | 150-180 |
| 13-15 | **第二次/第三次提交** | 冲 top 10 |
| 16-22 | 鲁棒性 + 论文 | 持续优化 |

---

## ✅ 项目状态

- ✅ 数据下载完成(330 MB,14 个文件)
- ✅ EDA 完成(8 张图 + 2 个 CSV 在 `notebooks/eda/`)
- ✅ 修 2 个数据驱动的 bug(过滤条件、endpoint)
- ✅ 6 个高/中优先级 bug 修复(TE mapping、final best_iter、Optuna subsample、clip_max、median、try/except、blend_mode)
- ✅ 14 个 Python 文件语法全过
- ✅ 全部文档与代码队名统一为 `jubilant-puzzle`
- ⏳ 等你回家在 Mac 上跑 baseline

**回家后第一件事**:跑 baseline,看 RMSE 数字。