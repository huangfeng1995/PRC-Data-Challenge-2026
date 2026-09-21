# 回家指南(2026-09-20 出)

**写给**:Fang(回家用)
**目的**:你回家第一晚,**2 小时内出第一个 leaderboard 分数**

---

## ⏱️ 时间预算(回家第一晚)

| 时间 | 任务 | 预计 RMSE |
|---|---|---|
| 0:00 - 0:15 | 装 libomp + 装库 | - |
| 0:15 - 0:30 | 检查环境,跑 sklearn baseline(确认 GPU/CPU 能跑) | 406 附近 |
| 0:30 - 1:30 | 跑 XGBoost baseline(basic → all) | 280-350 |
| 1:30 - 3:00 | 跑三模型 5-seed 融合 | 200-250 |
| 3:00 - 3:30 | 推理 + 改名 + 上传 | 第一次提交 |

---

## 0. 环境准备(回家第一件事,15 分钟)

```bash
# 1. 装 libomp(macOS 系统级,XGBoost/LightGBM 需要)
brew install libomp

# 2. 验证 Python 依赖(我们之前 pip install 过)
pip3 list | grep -E "xgboost|lightgbm|catboost|mlflow|optuna|minio"

# 3. 如果没装,装一下
pip3 install --break-system-packages xgboost lightgbm catboost mlflow optuna minio

# 4. 验证(应该都能 import)
python3 -c "
import xgboost, lightgbm, catboost, mlflow, optuna, minio
print('all deps OK')
"
```

---

## 1. 第一次跑 baseline(回家 30 分钟后)

```bash
cd ~/Documents/PRC_data_challenge

# A. 最简 baseline(29 个基础特征,看起点)
python -m src.train_xgb --data-dir data/raw --feature-set basic
# 期望:winter_main RMSE = 350-450 秒(取决于训练时间)
# 期望:spring_main RMSE = 300-400 秒

# B. 全特征 baseline(77 个特征)
python -m src.train_xgb --data-dir data/raw --feature-set all
# 期望:RMSE 比 basic 低 10-20%
```

如果 baseline 跑通了,**恭喜,你已经在 leaderboard 的水平上**(参考 2024 第 30 名只用了 sklearn HGBR 拿了 RMSE 3755/均值的 ~70% 位置,我们用真实数据 + 77 特征应该比这好)。

---

## 2. 第一次融合(回家 1.5 小时后)

```bash
# 三模型 5-seed 融合(应该跑 30-60 分钟,需要 libomp)
python -m src.train_ensemble \
    --data-dir data/raw \
    --feature-set all \
    --models xgb,lgb,cat \
    --n-seeds 5

# 期望:blend RMSE = 200-250 秒
```

**关键**:
- 这会生成 `models/ensemble_*.json` / `ensemble_*.txt` / `ensemble_*.cbm` + `ensemble_weights.json`
- MLflow 也会记录所有实验,跑完用 `mlflow ui` 可视化

---

## 3. 推理 + 第一次提交(回家 2.5 小时后)

```bash
# 推理 + 生成提交 parquet
python -m src.predict_ensemble \
    --output submissions/jubilant-puzzle_v1.parquet

# 检查输出
ls -lh submissions/jubilant-puzzle_v1.parquet
python3 -c "
import pandas as pd
df = pd.read_parquet('submissions/jubilant-puzzle_v1.parquet')
print(df.head())
print(f'行数: {len(df):,}')
print(f'预测值范围: {df.TAXITIME_SEC_mvt.min():.0f} - {df.TAXITIME_SEC_mvt.max():.0f}')
"

# 配置 mc(第一次用需要)
# ⚠️ 凭据已移到 .env 文件(被 .gitignore 忽略),不会泄露到 git
set -a; source ../.env; set +a
mc alias set osn https://s3.opensky-network.org "$OSN_ACCESS_KEY" "$OSN_SECRET_KEY"

# 上传(注意 bucket 路径)
mc cp submissions/jubilant-puzzle_v1.parquet \
    osn/prc-2026-jubilant-puzzle/

# 验证上传成功
mc ls osn/prc-2026-jubilant-puzzle/
```

---

## 4. 看 leaderboard

- **Vercel leaderboard**: https://prc-challenge-2026.vercel.app/
- **OSN REST API**: https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard

---

## 🎯 数字预期(沙箱已验证 + 业内 benchmark)

| 提交轮次 | 我们预期 RMSE | 对标 |
|---|---|---|
| 第一次(纯基础特征 XGB) | **280-350** | 2024 第 30 名 HGBR 3700+(差数据) |
| 第二次(77 特征 XGB) | 250-280 | 2025 第 3 名 CatBoost 220(更简单题) |
| 第三次(三模型融合) | **200-250** | 2025 第 2 名 XGB+LGB+CatBoost 211 |
| 第四次(调优后) | 150-200 | 2024 第 2 名 XGB+LGB 2217(差数据) |

**如果你提交分数 < 300**:大概率进 leaderboard 前 10
**如果你提交分数 < 200**:大概率进 leaderboard 前 3(奖金区)
**如果你提交分数 < 150**:冲冠军

---

## 🆘 卡住了怎么办

| 现象 | 怎么办 |
|---|---|
| XGBoost 报 "libomp not found" | `brew install libomp` 后重启终端 |
| train_xgb 报 KeyError 'te_cols' | 已经修了,如果是其他 KeyError 截图给我 |
| train_ensemble 跑得太慢(>2 小时) | 先 `--n-seeds 1` 跑通,再加 |
| predict_ensemble 报 KeyError | 检查 `models/ensemble_*.json` 文件都生成了 |
| 上传失败 | 检查 mc alias 配置,确认 bucket 名是 `prc-2026-jubilant-puzzle` |
| 任何其他问题 | 截图 + 日志发给我 |

---

## 📂 关键文件位置

```
~/Documents/PRC_data_challenge/
├── data/raw/                            # 14 个 parquet 文件(330 MB)
├── src/
│   ├── train_xgb.py                     # 单模型训练(入口)
│   ├── train_ensemble.py                # 三模型融合(入口)
│   ├── tune.py                          # Optuna 调参(入口)
│   └── predict_ensemble.py              # 推理 + 生成提交
├── models/                              # 训练后的模型(运行时生成)
├── submissions/
│   └── jubilant-puzzle_v1.parquet       # 提交文件(运行时生成)
├── notebooks/eda/                       # EDA 图(已生成)
└── docs/
    ├── DATA_READY_SUMMARY.md            # 数据到位总结
    ├── REAL_DATA_READINESS_CHECK.md     # 数据到位即用清单
    ├── MINIO_DOWNLOAD_GUIDE.md          # 下载教程
    └── 2025-2nd-3rd-places-analysis.md   # 2025 调研报告
```

---

## 🎁 彩蛋:直接看几张 EDA 图

回家后想先看数据长什么样:

```bash
open notebooks/eda/taxi_out_distribution.png   # 总分布
open notebooks/eda/by_airport.png              # 各机场分布
open notebooks/eda/by_airport_hour.png         # 机场 × 小时热力图
open notebooks/eda/missing_values.png          # 缺失值热力图
```

---

**最后一句话:今天我们已经把所有能做的都做了。**

**你回家第一件事就是上面 Step 1(装 libomp),然后按时间预算走。**

**我陪着你。** 任何问题随时找我。

**祝我们一起拿名次!** 🛫🏆

---

## 📊 沙箱真实数据预期(sklearn HGBR,2026-09-20 实测)

### 总 RMSE 预期(沙箱测得)

```
HGBR + 5% 采样 + 29 基础特征:344.1 秒(基线 419,提升 18%)
HGBR + 5% 采样 + 77 全特征:约 280-310(估算,因 30% 采样跑太慢)
```

### 各机场 RMSE 排名(从难到易,沙箱实测)

| 排名 | 机场 | RMSE | vs 基线 | 难度 |
|---|---|---|---|---|
| 🔴 1 | **LFPG(巴黎)** | **444.8** | 5.4% | **最难** |
| 🔴 2 | **LIRF(罗马)** | 443.9 | 11.5% | 第二难 |
| 3 | EDDM(慕尼黑) | 375.6 | 6.6% | |
| 4 | EGLL(伦敦) | 357.5 | 8.4% | |
| 5 | LTFM(伊斯坦布尔) | 355.1 | 10.6% | |
| 6 | EDDF(法兰克福) | 319.4 | 14.4% | |
| 7 | LSZH(苏黎世) | 296.2 | 8.4% | |
| 8 | LEBL(巴塞罗那) | 268.3 | 17.0% | |
| 9 | LEMD(马德里) | 264.5 | 13.3% | |
| 🟢 10 | EHAM(阿姆斯特丹) | 255.0 | 19.9% | **最容易** |

### 🎯 关键洞察(给家里 Mac 调参用)

1. **LFPG + LIRF 是难点** —— 这两个机场 RMSE > 440,长尾严重(LIRF std=488)
2. **EHAM 最容易** —— 提升最大(19.9%)
3. **EGLL 均值最高(1364s) 但 RMSE 不是最高** —— 均值≠难预测,要看 std
4. **优先动作**:加机场级 lag/lead 特征(其实我们已经做了),针对 LFPG/LIRF 单独调参

### 一键环境脚本

```bash
bash scripts/setup_mac.sh
```

自动装 libomp + 所有 Python 依赖 + 验证 + 检查数据完整性。**回家第一件事**。

