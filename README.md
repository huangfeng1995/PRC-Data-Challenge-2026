# PRC Data Challenge 2026 — Taxi-Out Time Prediction

> **队伍**:`jubilant-puzzle`
> **目的**:参赛工程仓库。**代码 + 小样本数据 + 复现脚本**,全量数据留在本地/从官方 MinIO 拉取。
> **比赛**:[PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/) — Taxi-Out Time Prediction
> **评分**:RMSE(以秒计)

**最终提交**:`submissions/jubilant-puzzle_final.parquet`(670,790 行)— 官方回执 `Succeeded`
| 口径 | 数值 | 说明 |
|---|---:|---|
| holdout 四月 RMSE | **324.76** | 2025 年1/2/6/7 月内部验证,较基线 493.05 降 **34.1%** |
| 榜单试错最佳 | **360.86** | v22,truthing.parquet 口径(官方即时出分) |
| final 官方分 | 待赛后公布 | final 槽位换用final_submitting.parquet,赛后统一评分 |

> holdout 与榜单口径不同,不可直接比较(v22: holdout 342.40 ↔ 真榜 360.86)。
> 方法与完整实验记录见 `docs/method.md`,战绩台账见 `docs/LEADERBOARD_LEDGER.md`。

---

## 1. 仓库结构

```
prc_data_challenge/
├── README.md                 本文件 (云端入口)
├── requirements.txt          Python 依赖 (锁定关键版本)
├── .env.example              凭据模板 (不含真实密钥)
├── .gitignore                排除大文件/凭据/日志
├──
├── src/                      ★ 统一特征模块 (云端 clone 后可直接 import)
│   ├── data_loader.py            读月度 parquet + DEP 过滤 + 时间解析
│   ├── features.py               30 个基础特征 + 目标重缩放
│   ├── features_aggregate.py     18 个 lag 拥堵聚合
│   ├── features_laglead.py       18 个 lead 未来拥堵聚合
│   ├── features_domain.py        3 结构 + 8 个目标编码
│   ├── features_queue.py         拥堵队列特征
│   ├── validation.py             时序切分 + RMSE/MAE/P95
│   ├── train_xgb.py / train_lgb.py / train_ensemble.py
│   ├── predict_xgb.py / predict_ensemble.py
│   └── tune.py                   Optuna 调参
│
├── scripts/                  ★ 一键脚本 (云端可调用)
│   ├── eda.py                    EDA + 8 张图
│   ├── build_sample.py           ★ 从全量生成 data/sample/ 小样本
│   ├── download_data.py          ★ 从 OSN MinIO 拉全量 (凭 .env)
│   ├── upload_submission.py      上传 submission (本地用)
│   ├── run_robust.py             长跑自动重启 (本地用)
│   ├── setup_mac.sh              macOS 环境 (本地用)
│   └── gate_check.py             D0 一键门禁 (云端可用)
│
├── data/
│   ├── raw/                    ⚠️ 全量 parquet (不入库, 需下载)
│   ├── sample/                 ★ 小样本 (已入库, ~2 MB, 云端开箱即用)
│   │   ├── training_2025-{01..12}.parquet   12 个月, 每机场每月 ~70 行 (40 DEP + 30 ARR)
│   │   ├── ranking.parquet                   5,000 行
│   │   ├── submitting.parquet                5,000 行
│   │   └── sample_meta.json                  采样参数
│   └── external/               第三方数据 (apt.dat 衍生 CSV/JSON 已入, .dat 排除)
│
├── models/                    模型产物 (权重不入库, 特征清单入)
│   ├── feature_lists/          ★ 各类特征的列名 + cat 映射 (小, 必入)
│   ├── ensemble_weights.json
│   ├── *.json / *.txt / *.cbm  ⚠️ 模型权重 (~1.4 GB, 不入库, 需重训)
│
├── docs/                      设计文档 / 报告
│   ├── method.md                ★ 最终方法说明(含负面结果)
│   ├── LEADERBOARD_LEDGER.md    ★ 逐版本真榜战绩台账
│   └── ...                      历史设计与诊断报告
│
├── supplementary/           ★ repo 补充产物 (不进评分)
│   └── final_uncertainty.parquet   final 四个月的 σ 与共形区间(8 列)
│
├── notebooks/                 一次性诊断脚本 (本地用, 不在云端跑)
│   ├── diag/                  历史诊断, 多数含本机绝对路径 (仅本地执行)
│   ├── eda/                   EDA 图表 (PNG/CSV)
│   └── holdout/               holdout 验证
│
└── submissions/               提交文件 (.gitkeep 占位)
```

★ = 云端 clone 后立即可用

---

## 2. 快速开始 (云端 Agent)

### 2.1 克隆与安装

```bash
git clone https://github.com/huangfeng1995/PRC-Data-Challenge-2026.git
cd PRC-Data-Challenge-2026
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# macOS: LightGBM 需要 libomp
brew install libomp
```

### 2.2 用小样本跑通 (默认,推荐)

仓库**已含 `data/sample/` 小样本 (~2 MB)**,直接可跑:

```bash
# 验证小样本 + 冻结入口 + 特征管线
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data_loader import load_month, load_ranking, filter_departures, parse_time_columns
from src.features import build_features
df = load_month('data/sample/training_2025-01-01_2025-02-01.parquet')
df = filter_departures(parse_time_columns(df, cols=['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','IOBT_flt','EOBT_1_flt','AOBT_3_flt','ARVT_3_flt','LOBT_flt']))
print('DEP rows:', len(df))
print('features shape:', build_features(df).shape)
"

# EDA on sample
python3 scripts/eda.py --data-dir data/sample --output-dir notebooks/eda

# 跑小规模 baseline (几分钟内出结果)
python3 -m src.train_xgb --data-dir data/sample --feature-set basic
```

### 2.3 拉全量数据 (云端需要时, 本机不需要)

```bash
# 1. 把 .env 复制到项目根目录, 填入真实密钥
cp .env.example .env
# 编辑 .env, 填入 OSN_ACCESS_KEY / OSN_SECRET_KEY

# 2. 拉全量 (约 330 MB)
python3 scripts/download_data.py --bucket prc-2026-datasets --dest data/raw

# 3. 全量 EDA / 训练 (需要更多内存, 推荐本机或租 GPU)
python3 -m src.train_ensemble --data-dir data/raw --feature-set all --models xgb,lgb,cat --n-seeds 5
```

---

## 3. 安全红线

⚠️ **以下内容绝不能 push 到 git**:

- `.env`(OSN 凭据) — 已在 `.gitignore` 中
- `data/raw/`(全量比赛数据, 330 MB) — 已在 `.gitignore` 中
- `models/*.json` / `*.txt` / `*.cbm` / `*.bin`(模型权重, ~1.4 GB) — 已在 `.gitignore` 中
- `submissions/*.parquet`(比赛提交文件) — 已在 `.gitignore` 中
- 任何 API token / 私钥 / 密码

**提交前检查清单**:

```bash
# 1. 确认 .env 没被加进去
git status | grep -E '\.env$|OSN_'
# 应为空

# 2. 确认 .gitignore 覆盖了所有大文件
git check-ignore data/raw/*.parquet models/*.json submissions/*.parquet
# 应全部输出 "<filename>" (被忽略)
```

---

## 4. 数据说明

| 字段 | 来源 | 入库? | 大小 |
|---|---|---|---|
| `data/raw/*.parquet` | OSN MinIO 比赛数据 | ❌ 需下载 | ~330 MB |
| `data/sample/*.parquet` | 本机全量采样 (random_state=42) | ✅ 已入库 | ~2 MB |
| `data/external/xplane/*.csv` | X-Plane apt.dat 几何 | ✅ 已入库 | ~370 KB |
| `data/external/xplane/*.dat` | X-Plane 原始 apt.dat | ❌ 排除 | ~10 MB, 云端可重新下载 |
| `data/external/xplane/stand_rwy_distance.csv` | 全量 DEP 距离 | ❌ 排除 | 246 MB, 跑全量时重生 |
| `models/*.json` | XGBoost/LGB/Cat 模型 | ❌ 排除 | ~1.4 GB |

**全量数据如何获得**:必须凭 OSN MinIO Access Key/Secret Key 从官方桶拉取。详见 `docs/MINIO_DOWNLOAD_GUIDE.md`。

---

## 5. 复现脚本

| 脚本 | 作用 |
|---|---|
| `python3 scripts/build_sample.py` | 从 `data/raw/` 生成 `data/sample/` 小样本(需全量数据) |
| `python3 scripts/download_data.py` | 从 OSN MinIO 拉全量数据(需 `.env`) |
| `python3 scripts/eda.py` | EDA (单 XGB 可视化) |
| `python3 -m src.train_xgb` | 单 XGBoost 训练 |
| `python3 -m src.train_ensemble` | XGB+LGB+Cat 三模型融合 + 多 seed |
| `python3 -m src.tune` | Optuna 调参 (50 trials) |
| `python3 -m src.predict_ensemble` | 推理 + 生成提交 |
| `python3 scripts/gate_check.py` | D0 一键门禁 (5 项 PASS/FAIL) |

---

## 6. 当前进度

| 阶段 | 状态 | 备注 |
|---|---|---|
| 数据接入 | ✅ 已下载全量 | OSN 凭据见 .env(不入库) |
| 77 特征 baseline | ✅ | XGB+LGB+Cat 融合 |
| 榜单迭代 | ✅ | v1 589.91 → v22 **360.86** (试错口径最佳) |
| LIRF 标签损坏取证 | ✅ | 发现 BLOCK 被 SCHED 顶替,修复 1484 → 655 |
| 最终提交 | ✅ 已上传 | noclip E2,holdout **324.76**,官方回执 Succeeded |
| 不确定性量化 | ✅ | 物理约束二阶矩混合 + 损坏感知分层共形,见 `supplementary/` |

**核心结论**:493.05 → 324.76 的降幅**几乎全部来自发现并修复 LIRF 标签数据损坏**,而非新算法。
非 LIRF 长尾行(>1800s极端滑行)属真实物理现象(拥堵/除冰/天气),在无外部数据条件下已确认不可压缩。

详见 `docs/method.md`(方法与负面结果)与 `docs/LEADERBOARD_LEDGER.md`(逐版本战绩)。

---

## 7. 本机 vs 云端的边界

**本机(全量训练)**:
- `data/raw/` 全量数据 (~330 MB)
- `models/` 大模型权重 (~1.4 GB)
- `submissions/` 提交文件
- 跑全量 `train_ensemble` (1+ 小时)

**云端(原型/小样本)**:
- `data/sample/` 小样本 (~2 MB, 已入库)
- `data/external/xplane/*.csv` 衍生几何 (已入库)
- 跑 `build_features` + 验证管线 + 小规模 `train_xgb`
- **不跑全量训练**(耗时且本机有完整数据)

**两个端共享**: `src/` + `scripts/eda.py` + `scripts/gate_check.py` + `docs/` + `requirements.txt`

---

## 8. License

MIT (配合比赛要求)。