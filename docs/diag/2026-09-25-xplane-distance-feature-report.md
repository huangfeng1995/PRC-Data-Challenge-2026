# 盘点报告 + X-Plane 距离特征报告(2026-09-25)

> **范围**:在沙箱内、**不依赖 OSN 凭据** 的两件事——(1) 代码主线盘点,(2) X-Plane apt.dat 几何 → 距离 → 训练管线。
> **不修改**:v0~v10 / holdout_v1_queue_ab 冻结入口;不擅自合并模糊匹配;不发版、不上传。

---

## 一、仓库主线盘点(只报告)

### 1.1 顶层结构(除 mlruns / .git)

```
.
├── README.md / requirements.txt / .env.example / .gitignore
├── data/
│   ├── sample/                  12 月 training + ranking + submitting (入档)
│   └── external/xplane/         衍生几何 (runways.csv, stands_geo.csv, taxi_network.json)
│                                + _reports/  本次产出
├── docs/
│   ├── DATA_ANALYSIS_AND_MODEL_SELECTION.md
│   ├── HOME_GUIDE.md / MINIO_DOWNLOAD_GUIDE.md / SANDBOX_*
│   ├── analysis/    2024/2025 获奖方案复盘
│   └── diag/        holdout/ranking/encoding 各阶段诊断报告
├── notebooks/
│   ├── diag/                   A/B + 校验 + gate
│   │   ├── holdout_v1_queue_ab.py      ← **G0=357.14 冻结入口**(在)
│   │   ├── holdout_v1_queue_ab_result.json (在)
│   │   ├── _gate.py / gate_check.py  ← RMSE ≈ 357.14 ± 2 门禁
│   │   ├── build_feat_lists.py       ← 衍生 feat_basic30.json / feat_cnt36.json
│   │   └── ... ab_v2/...
│   ├── eda/                    EDA 图与 summary
│   └── holdout/                v0/v8/same_month holdout 脚本
├── scripts/
│   ├── build_sample.py / download_data.py / upload_submission.py
│   ├── eda.py / run_robust.py / gate_check.py / setup_mac.sh
│   └── build_xplane_features.py      ← 本次新增
├── src/
│   ├── data_loader.py / validation.py
│   ├── features.py / features_aggregate.py / features_laglead.py
│   ├── features_domain.py / features_queue.py
│   ├── predict*.py / train_lgb.py / train_xgb.py / train_ensemble.py / tune.py
└── models/feature_lists/*.json  37 个,横跨 v0~v10,特征数 30~39
```

### 1.2 v15 与 71 特征的关系(关键)

| 文档/口径 | 数值 | 在仓库中是否落地 |
|---|---|---|
| README L182: "v15 holdout RMSE 281.87,71 特征 + 9 模型集成" | 71 列 | **❌ 没有 v15 主线源码/模型/特征清单** |
| 现存最大特征清单 | 39 列(`xgb_v9_Gd_ABD_runway_seed0_val` = basic30 + 拥堵 + 跑道衍生) | ✅ 在 |
| G0 冻结 baseline RMSE | **357.14 ± 2**(由 `scripts/gate_check.py` / `notebooks/diag/_gate.py` 守护) | ✅ 在 |
| 距离特征 add-one 已做(README L183) | X-Plane apt.dat **88.26% 覆盖**, **−0.31s 边际增益** | README 自报,代码已 commit 但**没看到 `stand_to_rwy_m` 在 `src/features*.py` 的现网入口**;本次已独立验证管道并入 |

### 1.3 特征集层次

```
BASELINE_FEATURE_COLS (src/features.py)        : 30 列  ← G0
feat_basic30.json  (notebooks/diag/build_*)    : 30 列  ← G0 same
feat_cnt36.json                                : 36 列  ← G4 ≈ cnt+queue_heavy
xgb_v9_G5_ABCE_seed0_val_features.json         : 38 列  ← G5 (含 dep_mvt / queue / queue_heavy / arr)
xgb_v9_Gd_ABD_runway_seed0_val_features.json   : 39 列  ← Gd (basic30 + dep_mvt + 跑道衍生)
xgb_v10_G3queue_seed0_final_features.json      : 35 列  ← v10 终版
```

**与 README "71 特征" 差距 32 列**:这 32 列应当在 v11~v15 之间引入(可能含 lead / aggregate / 距离 / 目标重缩放 等),**仓库里没有该阶段的源码或特征清单**。

### 1.4 add-one 验证应基于什么

按你"v15 = 71 特征"的语境,严格的 add-one 应:

1. **找到或重建 v15 主线**(特征清单 + 集成入口 + 验证脚本)—— **当前仓库缺失**。
2. 在 v15 之上加 1 个候选特征跑 holdout,再跑 gate(G0=357.14)对比。
3. 当前可用的"半成品"基线只有 v10_G3queue_seed0_final (35 列) 或 v9_Gd_ABD_runway (39 列)。

⚠️ **不建议**用 `BASELINE_FEATURE_COLS`(30 列)或 `feat_cnt36`(36 列)做 add-one 然后外推 v15 增益——这两套是阶段 1 的中间产物,量级不对,容易出假阴性/假阳性。

### 1.5 已确认在 / 缺什么

| 在仓库里 | 缺什么 |
|---|---|
| 阶段 0/1 全套:basic30 / queue / lag-lead / runway / encoding / gate | v11~v15 的源码 |
| holdout_v1_queue_ab.py 冻结入口、G0=357.14 门禁 | v15 集成入口(9 模型 / 71 列) |
| EDA 图、`docs/diag/*.md` 7 份诊断 | `data/external/xplane/*.dat` 原文件(.gitignore) |
| X-Plane 衍生几何三件套(runways/stands/taxi) | `taxi_network.json` 的节点坐标(只有 edges) |
| `data/sample/` 12 月训练 + ranking + submitting | `data/raw/` 全量(需 OSN key) |

---

## 二、X-Plane apt.dat → 几何 → 距离

### 2.1 抓取与解析:状态

- **沙箱到 `gateway.x-plane.com` 出站不通**:直接 `curl` SSL_ERROR_SYSCALL,走 `HTTPS_PROXY` 走 CONNECT 后 TLS 仍 EOF;`WebFetch` 只能取少量 JSON。
- 但 `data/external/xplane/{runways.csv, stands_geo.csv, taxi_network.json}` **已入档**(前一轮工作产物,README L151)。
- 解析覆盖 10 机场:跑道 36 / 机位 2838 / 滑行边 6286。
- **缺节点坐标**:`taxi_network.json` 节点列表为空,意味着"图搜索最短滑行路径"不可用,只能用端点 haversine 直线距离(已在脚本里支持 haversine,留好接口给最短路径)。

### 2.2 匹配:STAND_mvt ↔ apt.dat 机位

> 数据:`data/sample/training_2025-01-01_2025-02-01.parquet`, 400 行 DEP, 全 10 机场。
> 规则:精确匹配 + difflib 模糊候选(cutoff=0.6, n=3),**模糊候选只报告,不擅自合并**。

| 机场 | DEP | 精确 | +模糊候选 | 精确率 | 失配代表 |
|---|---|---|---|---|---|
| EDDF | 40 | 40 | 0 | 100% | — |
| LEBL | 40 | 40 | 0 | 100% | — |
| LEMD | 40 | 40 | 0 | 100% | — |
| LIRF | 40 | 40 | 0 | 100% | — |
| LFPG | 40 | 37 | 3 | 92.5% | `E27`→`S27/L27/I27`(同号家族) |
| EGLL | 40 | 39 | 1 | 97.5% | `237R`→`357R/253R/247R`(同号家族) |
| EDDM | 40 | 26 | 12 | 65.0% | `113`→`113X/113B/113A`(脱后缀) |
| EHAM | 40 | 25 | 13 | 62.5% | `C07`↔`C7`、`H05`↔`H5`(前导 0 缺省) |
| LSZH | 40 | 26 | 13 | 65.0% | `E43`→`43/B43`、`GA6`→`G06` |
| LTFM | 40 | 0 | 31 | **0%** | `A11-(E)`/`F7L-(C)`/`G10R-(C)`(括号机型后缀) |
| **合计** | **400** | **313** | **73** | **78.25%** | |

**关键观察**:
- 失配**全部可解释**,没有"乱码";三种主要模式都集中在 4 个机场。
- LTFM 0/40 是最大问题——**所有 LTFM 机位在 apt.dat 里都带括号机型后缀 `(C)/(E)/(F)`**, 数据侧是裸名。简单规则(剥括号)即可把覆盖率推到接近 100%,但**按你的口径我不擅自合并**,作为改进项列在 §2.5。

### 2.3 距离:stand → runway

> 口径:`haversine(stand, runway_endpoint)`, 跑道端点取两端 min。精确匹配上的才计算,未匹配置 NaN,不兜底。

按机场(2025-01, 400 DEP):

| 机场 | 非空 | 均值 (m) | 中位 (m) |
|---|---|---|---|
| EDDF | 40/40 | 2351 | 2408 |
| LEMD | 40/40 | 2742 | 2833 |
| LFPG | 37/40 | 1741 | 1938 |
| EDDM | 26/40 | 1717 | 2014 |
| LIRF | 40/40 | 1631 | 1672 |
| EGLL | 39/40 | 1222 | 1113 |
| EHAM | 25/40 | 1237 | 1004 |
| LEBL | 40/40 | 1341 | 1134 |
| LSZH | 26/40 | 1029 | 1011 |
| LTFM | 0/40 | NaN | NaN |

**合理性自检**:LEMD 中位 2.8 km / LSZH 中位 1.0 km,与"马德里 T4–卫星楼群跨度大 / 苏黎世机场小且集中"一致。

### 2.4 训练管线并入验证

> 12 个月样本拼接 + 距离 join + 时间切分(前 9 月训,后 3 月验)+ 最小 XGBoost(`BASELINE_FEATURE_COLS` + 1 列)。
> 跑两遍:含/不含 `stand_to_rwy_m`。

```
加载 DEP: 4800 (12 × 400)
距离 join: 313/4800 = 6.5% 非空(受样本稀)
时间切分: train=3600 / val=1200

>>> 含 stand_to_rwy_m    RMSE = 354.16
>>> 基线 (无 distance)   RMSE = 352.90
>>> delta = +1.26 (negative = better)
stand_to_rwy_m gain = 35,015 (在 top10 外,属辅助项)
```

- **管道并入**:✅ 成功;不修改 `src/features.py`,只用 feat_cols_override 模式叠加。
- **覆盖率受样本限制**(6.5%);**全量预期 88.26%**(README 自报),全量训练时增益方向和量级才会有统计意义。
- 小样本 +1.26s 不代表"距离特征没用"——README 全量报告 −0.31s,**两者在小样本下不可比**。

### 2.5 改进项(只列,本轮不做)

1. **LTFM 后缀规则**:剥 `-(C)/(E)/(F)` 后再精确匹配,预期覆盖率 → ≥ 95%。口径:"建议作为 v11+ 的 v15 候补特征工程"。
2. **同号家族归并**(EDDM `113→113X/A/B`, LSZH `E43→43`, EHAM `C07→C7`):做规则化映射表,但要决定"映射是否入训练特征"(语义不同——`113A` 通常是远机位,`113` 大概率是同一机位家族)。
3. **图搜索最短滑行路径**:需要补 `taxi_network.json` 的 nodes 坐标(从 apt.dat 的 1100/1200 行解析)。当前沙箱无 apt.dat,无法补;建议下个 agent 或本地脚本完成。
4. **把 `stand_to_rwy_m` 接入 v15 主线**:如 §1.4 所述,需先重建 v15 源码与 71 列特征清单。
5. **把 `scripts/build_xplane_features.py` 跑全量 `data/raw/`**:生成 `stand_rwy_distance_full.csv`(README L153 提到这个文件,已在 `.gitignore` 排除,全量时重生)。

---

## 三、产物清单(本次)

| 路径 | 用途 | 是否入 git |
|---|---|---|
| `scripts/build_xplane_features.py` | 匹配报告 + 距离计算,带参数 | ✅ 新增,建议提交 |
| `data/external/xplane/_reports/match_report_2025-01-01_2025-02-01.csv` | 失配机位 + 模糊候选 | ✅ 跑过可入 |
| `data/external/xplane/_reports/match_summary_2025-01-01_2025-02-01.json` | 每机场精确率 + 跑道交集 | ✅ 可入 |
| `data/external/xplane/_reports/stand_rwy_distance_2025-01-01_2025-02-01.csv` | 313 行距离(MVT_ID 锚) | ✅ 可入 |
| `data/external/xplane/_reports/stand_rwy_distance_*.csv` | 12 个月样本对应 12 份 | 跑全量时再生 |
| `docs/diag/2026-09-25-xplane-distance-feature-report.md` | 本报告 | ✅ 建议入 |

> 报告口径:**不**改 `holdout_v1_queue_ab.py`、`scripts/gate_check.py`、`src/features*.py`;
> 距离特征**未合并**进训练主分支,只在最小验证里临时叠加。

---

## 四、给后续 agent / 你 的接力清单

1. **重建 v15**:从 README L182 的口径反推 71 列 + 9 模型集成的代码;若在另一个分支,请 fetch。
2. **补 apt.dat 抓取脚本**(`scripts/fetch_xplane_apt.py`,走代理或本机跑),输出到 `data/external/xplane/*.dat`,然后跑 `scripts/build_xplane_features.py --full`。
3. **写 LTFM 后缀规则 + 同号家族映射**,作为正式 feature engineering PR(非本轮)。
4. **补全 `taxi_network.json` 的 nodes 坐标**,以便做图搜索最短路径(本次留接口)。
5. **把 `scripts/build_xplane_features.py` 接到 `train_xgb --feat-cols-override`**,作为 v15 add-one 入口。
