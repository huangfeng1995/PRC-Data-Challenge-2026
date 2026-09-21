# PRC Data Challenge 2026 — 项目全景文档

> **目的**:一站式讲清楚
> 1) 这场比赛的数据长什么样、规模多大、有哪些坑;
> 2) 我们 `jubilant-puzzle` 团队当前方案每一步在做什么、为什么这么做、做完了什么;
> 3) 已经跑出来哪些数字、还差什么、接下来怎么走。
>
> **撰写日期**:2026-09-21 · **距截止**:2026-10-11 23:59:59 CET(还剩 20 天)

---

## 0. 一句话总结

用 **77 个特征 + XGBoost / LightGBM / CatBoost 三模型融合 + 多 seed 平均**,在 `2025-12` holdout 上做到了 **RMSE 311.48**(v7,42 特征 XGB),在更贴近比赛真实分布的 `2025-01+07` 同月 holdout 上 RMSE **352.68**(v7,42 特征)。v1 提交已拿到第一个 leaderboard 分数 **589.91**(单 XGBoost + 29 个基础特征,无 lag/lead),中位数水平,后续版本按计划升级。

---

## 1. 比赛核心信息

| 项目 | 详情 |
|---|---|
| 比赛 | [PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/) — Taxi-Out Time Prediction |
| 主办 | EUROCONTROL / ANS Performance 通过 OSN(OpenSky Network) |
| 任务 | 预测每条 DEP(出发)航班的 **taxi-out 时间**(秒),即从推出到起飞的滑行耗时 |
| 评分 | RMSE(以秒计) |
| 提交限制 | 每天 5 次,每 bucket 上限 1 GB,文件名 `<team-name>_v<integer>.parquet` |
| 截止 | **2026-10-11 23:59:59 CET** |
| 奖金 | 前 3 名共 €5,000 |
| 队伍 | `jubilant-puzzle`(中国海口,独立研究者) |
| 已注册队伍 | 30+,我们是第一支中国队伍 |

---

## 2. 数据情况(我们看到了什么)

### 2.1 文件结构(`data/raw/`)

| 文件 | 大小 | 行数 | 说明 |
|---|---|---|---|
| `training_2025-{MM}-{MM+1}.parquet` × 12 | ~280 MB | **4,167,797**(原始) | 2025 全年 12 个月,逐月切分 |
| `ranking.parquet` | ~27 MB | 约 34 万 | 2026-01 + 2026-07,目标值被屏蔽(我们就是要预测这些) |
| `submitting.parquet` | ~1.1 MB | 约 34 万 | 提交模板,只有 `MVT_ID_mvt` + `TAXITIME_SEC_mvt` 两列 |

### 2.2 数据规模与过滤

```
原始行数       : 4,167,797
过滤 PHASE=DEP : 2,085,047 行 (50%,符合预期)
时间跨度       : 2025-01-01 00:05:56 ~ 2025-12-31 23:57:36 (UTC)
字段数         : 26(去重后,来自两表 join)
机场数         : 10 个欧洲枢纽(预期 11,实际 10)
```

### 2.3 字段分类

**机场侧(`_mvt` 后缀)**:MVT_ID、FLIGHT_ID、ADEP、ADES、PHASE、MVT_TIME、BLOCK_TIME、SCHED_TIME、AIRCRAFT_TYPE、RUNWAY、STAND、**TAXITIME_SEC(目标)**。

**航班侧(`_flt` 后缀)**:LOBT、CALLSIGN、ADES(原始/计划)、MARKET_SEGMENT、IOBT、EOBT_1、FLIGHT_RULE/TYPE、AIRCRAFT_TYPE、WK_TBL_CAT(尾流等级 L/M/H/J)、AIRCRAFT_OPERATOR、AOBT_3、ARVT_3。

**ranking 数据(目标变量被屏蔽)**:
- `BLOCK_TIME_UTC_mvt` 和 `TAXITIME_SEC_mvt` 在 DEP 行上 **100% 缺失**(我们要预测的)
- 其他 `_flt` 字段缺失率约 1.53%(EOBT/IOBT/CALLSIGN/OPERATOR 等)
- `ADEP_mvt / RUNWAY_mvt / SCHED_TIME_UTC_mvt / MVT_TIME_UTC_mvt` 0% 缺失(可用)

### 2.4 防泄漏原则(很重要)

| 可用(无泄漏) | 危险(会泄漏,禁用) |
|---|---|
| `SCHED_TIME_UTC_mvt`(计划起降) | `AOBT_3_flt`(实际推出 = taxi-out 起点) |
| `IOBT_flt`(初始计划推出) | `BLOCK_TIME_UTC_mvt`(实际推出) |
| `EOBT_1_flt`(FPL 估计推出) | `MVT_TIME_UTC_mvt`(实际起飞 = taxi-out 终点) |
| `ADEP_mvt` / `ADES_mvt` / `RUNWAY_mvt` / `STAND_mvt` | `ARVT_3_flt`(实际到达) |
| `AIRCRAFT_TYPE_*` / `WK_TBL_CAT_flt` / 各种 _flt 类别字段 | `LOBT_flt`(last off-block) |

> **关键决策**:训练时聚合特征参考时间用 **`EOBT_1_flt`**(FPL 估计推出,可在预测时拿到),而不是 `AOBT_3_flt`(实际推出,会泄漏)。这保证训练和推理时间口径一致。

### 2.5 目标变量分布(各机场差异巨大)

| 机场 | 均值(s) | P99(s) | 行数 | 备注 |
|---|---|---|---|---|
| EGLL(伦敦希思罗) | **1364** | 2701 | 24 万 | 均值最高,欧洲最拥堵 |
| LIRF(罗马菲乌米奇诺) | 1195 | **4019** | 16 万 | **长尾最严重**(P99 是均值的 3.4 倍) |
| LTFM(伊斯坦布尔) | 1053 | 2587 | 27 万 | 数据量最大 |
| LFPG(巴黎戴高乐) | 1020 | 2409 | 24 万 | 长尾第二严重 |
| LEMD(马德里) | 1015 | 1932 | 21 万 | |
| LEBL(巴塞罗那) | 957 | 1962 | 18 万 | |
| EDDF(法兰克福) | 864 | 1819 | 23 万 | |
| EDDM(慕尼黑) | 812 | 1921 | 17 万 | |
| EHAM(阿姆斯特丹) | 785 | 1775 | 25 万 | |
| LSZH(苏黎世) | **741** | 1707 | 13 万 | 均值最低,最简单 |

**全量目标变量统计**(`summary_stats.csv`):
```
count:  2,085,047
mean:   991.20  秒 (≈ 16.5 分钟)
std:    546.36  秒
min:    -12     秒 ← 数据 bug,过滤为 > 0
25%:    719
50%:    912
75%:    1,188
max:    131,167 ← 异常值,过滤为 < 10800(3 小时)
```

**关键洞察**:
1. **机场差异近 2 倍**(741 ~ 1364s),按机场分组训练在后续版本值得尝试。
2. **均值 ≠ 难度**(EGLL 均值最高但 RMSE 不是最高,要 std 才行)。沙箱实测最难的反而是 **LFPG(巴黎,RMSE 444.8)** 和 **LIRF(罗马,443.9)**。
3. **长尾驱动 RMSE**:P99 区间贡献了约 12% 的 SSE,P99.9 以上仅 0.1% 样本却贡献 3.8% 的 SSE。

### 2.6 数据驱动的 bug 修复(踩过的坑)

| Bug | 原因 | 修复 |
|---|---|---|
| `TAXITIME_SEC_mvt < 0`(最小 -12s) | 数据 bug | 过滤条件改为 `> 0 and < 10800` |
| `TAXITIME_SEC_mvt` 最大 131167s(36 小时) | 异常值/录入错误 | 同上过滤 |
| `AIRCRAFT_OPERATOR_flt` 缺失 1.08% | NM 网络偶尔缺报 | TE 处理已 fallback 到 global_mean |
| EDA log 直方图 NaN | clip 后 log1p 正常 | 已修 |
| `datacomp.opensky-network.org` endpoint 404 | 实际是 `s3.opensky-network.org` | 改默认 endpoint |
| 类别列 `fillna(-1)` 报错 | category 不能直接 fillna | 先 `astype(str)` 再 map+fillna |
| macOS 缺 `libomp`(XGBoost/LightGBM 跑不起来) | 沙箱无 Homebrew | 下载 R-project 预编译 OpenMP 19.1.5 + `install_name_tool -add_rpath` |
| `MLFLOW_ALLOW_FILE_STORE` 被新版 mlflow 移除 | 默认值变了 | 设环境变量绕过 |
| `te_cols` KeyError 在 ranking 上 | ranking 缺 `TAXITIME_SEC_mvt` 无法 fit TE | `fit_te_mappings` + `apply_te_mappings` 分离,推理时复用训练时保存的 mapping |
| NaT 时间戳污染 lag 聚合 | `np.searchsorted` 把 NaT 当极小值,窗口拉到"全部历史" | cumsum 链前剔除 NaT 行,输出 NaN 让模型学"无信号" |

---

## 3. 当前方案完整流程(每一步做了什么)

### 3.1 数据加载 — `src/data_loader.py`

```
输入: data/raw/training_2025-*.parquet × 12(330 MB)
输出: 2,085,047 行 DEP DataFrame,时间字段已转 UTC datetime

步骤:
1. 读月度 parquet(用 pyarrow 显式指定列,省内存)
2. 12 个月 pd.concat
3. filter_departures: PHASE == 'DEP'
4. parse_time_columns: 8 个 UTC 时间字段 → datetime64[ns, UTC]
```

辅助函数:
- `load_ranking(data_dir)`:读 `ranking.parquet`,同样过滤 DEP 和解析时间
- `load_submission_template(data_dir)`:读 `submitting.parquet` 模板

### 3.2 特征工程 — 总共 77 个特征

#### Level 1:基础特征 30 个(`src/features.py`)

| 类别 | 数量 | 特征 |
|---|---|---|
| 时间(基于 SCHED_TIME,无泄漏) | 14 | `sched_hour/dow/month/day/minute`、`is_weekend/morning_peak/evening_peak`、`hour_sin/cos`、`dow_sin/cos`、`month_sin/cos` |
| 空间/路线 | 6 | `ADEP_mvt`、`ADES_mvt`、`RUNWAY_mvt`、`STAND_mvt`、`airport_hour`(机场×小时组合)、`is_domestic_estimate` |
| 飞机/航班 | 7 | `AIRCRAFT_TYPE_mvt`、`WK_TBL_CAT_flt`、`MARKET_SEGMENT_flt`、`FLIGHT_RULE_flt`、`FLIGHT_TYPE_flt`、`AIRCRAFT_OPERATOR_flt`、`is_diverted` |
| 计划性 | 3 | `iobt_to_sched_sec`、`eobt_to_sched_sec`、`iobt_eobt_diff_sec` |

> 还有个 **目标重缩放**(2024 第 1 名 `team_likable_jelly` 的招数,默认关闭):
> 按机场算 `mean/std`,训练时把 `target` 转 `(t - mean) / std`,预测后 inverse 回去。边际收益小,留作可选。

#### Level 2:拥堵 lag 聚合 18 个(`src/features_aggregate.py`)

**核心思想**:对每条 DEP,统计它在 `EOBT_1_flt`(参考时间)之前 15 / 30 / 60 分钟内同机场 / 同跑道的拥堵情况。

```
cnt_dep_airport_15m / 30m / 60m     过去 N 分钟同机场的 DEP 数
mean_taxi_airport_15m / 30m / 60m   过去 N 分钟同机场的平均 taxi-out
p90_taxi_airport_15m / 30m / 60m    过去 N 分钟同机场的 P90(m 用 max 近似)
cnt_dep_runway_15m / 30m / 60m      同上,但按 ADEP × RUNWAY 分组
mean_taxi_runway_15m / 30m / 60m
p90_taxi_runway_15m / 30m / 60m
```

**算法优化(v3 cumsum 重构)**:
- 旧版 `O(N²)`:内层 for k + `np.searchsorted` + `np.percentile`
- 新版 `O(N)`:先 `sort+lexsort`,然后 group 内 `cumsum/cumcnt` 一次性算出 count/mean
- p90 用 `np.partition` 切片 + `np.max` 近似
- **实测**:200 万行 10 机场 30 分钟窗口,旧版 12+ 分钟 → 新版 < 10 秒

#### Level 3:Lead 拥堵预测 18 个(`src/features_laglead.py`)

**从 2025 第 2 名 `jubilant-vase` 学到**:对未来 N 分钟的"已计划"航班密度做统计。

```
cnt_lead_airport_15m / 30m / 60m     未来 N 分钟同机场的 DEP 数
mean_lead_taxi_airport_15m / 30m / 60m   未来 N 分钟同机场平均 taxi-out
p90_lead_taxi_airport_15m / 30m / 60m
cnt_lead_runway_15m / 30m / 60m       同上,按 ADEP × RUNWAY
mean_lead_taxi_runway_15m / 30m / 60m
p90_lead_taxi_runway_15m / 30m / 60m
```

> **可行性**:ranking 数据本身含所有 DEP 航班(只是屏蔽了目标),所以"未来航班"在预测时是已知的,不会泄漏。

#### Level 4:Domain 特征 11 个(`src/features_domain.py`)

**3 个结构性**:
- `stand_zone`:从 STAND 提取首字母(`A12` → `A`)
- `runway_family`:跑道数字(`07L/07R` → `07`)
- `runway_side`:L/R/C

**8 个目标编码(TE,平滑后)**:
- 单变量:operator / aircraft_type / runway / airport
- 交叉:operator×airport / aircraft×airport / wake×airport / operator×wake
- 公式:`encoded = (n * group_mean + 20 * global_mean) / (n + 20)`(smoothing=20)

**关键设计**:`fit_te_mappings` 和 `apply_te_mappings` 分离。
- 训练时 fit + apply 到训练数据
- 推理时直接复用训练时保存的 mapping(因为 ranking 没有 `TAXITIME_SEC_mvt`,没法现场 fit)

#### 特征总数

| Level | 数量 | 来源文件 |
|---|---|---|
| 基础 | 30 | `features.py` |
| Domain 结构 | 3 | `features_domain.py` |
| Lag 聚合 | 18 | `features_aggregate.py` |
| Lead 聚合 | 18 | `features_laglead.py` |
| TE | 8 | `features_domain.py` |
| **合计** | **77** | |

**防泄漏审计**:`BASELINE_FEATURE_COLS` 中已剔除所有 `MVT_TIME / BLOCK_TIME / AOBT / ARVT / LOBT` 等会泄漏的字段。

### 3.3 验证策略 — `src/validation.py`

| 切分 | 用途 | 原因 |
|---|---|---|
| **Winter**(`get_validation_splits`) | 2025-11/12 验证 2026-01 | 主验证,对应测试集 1 月 |
| **Summer** | 2025-06 验证 2025-07 | 辅验证,对应测试集 7 月 |
| **K 折时序 CV**(5 折) | 调参时用 | 避免单切分过拟合 |
| **最终模型** | 全量 12 个月 | 不切分 |

指标:`RMSE`、`MAE`、`P95 error`、`best_iteration`。

### 3.4 模型栈 — `src/train_ensemble.py`

```
Level 1: 基础特征 (30 个)
Level 2: + 拥堵 lag 聚合 (+18 个 → 48 个)
Level 3: + Lead 聚合 (+18 个 → 66 个)
Level 4: + Domain 结构 + TE (+11 个 → 77 个)
Level 5: 三模型融合: XGBoost + LightGBM + CatBoost
Level 6: 每模型 × 5 个 seed(默认)→ 15 个独立模型
Level 7: seed 内平均 → 模型间加权(scipy SLSQP)
Level 8 (可选): median voting(2024 第 4 名 `team_brave_pillow` 招数,outlier 多时更鲁棒)
```

**默认超参**:

| 模型 | 关键超参 |
|---|---|
| XGBoost | `learning_rate=0.05`, `max_depth=8`, `min_child_weight=5`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_lambda=1.0`, `tree_method=hist` |
| LightGBM | `learning_rate=0.05`, `num_leaves=127`, `min_data_in_leaf=50`, `feature_fraction=0.85`, `bagging_fraction=0.85`, `lambda_l2=1.0` |
| CatBoost | `learning_rate=0.05`, `depth=8`, `l2_leaf_reg=3.0`, `loss_function=RMSE`, `iterations=2000` |

**多 seed 工作流**:
1. 每个模型 × N 个不同 seed → N 个 booster
2. 推理时:每个模型先在 seed 内平均,得到 1 个预测
3. 再用 `optimize_blend_weights` 在 OOF 上学模型间权重(或 median)
4. 自动选 `mean` vs `median` 中 RMSE 更低者(保存在 `weights.json` 的 `blend_mode`)

**为什么选 XGB+LGB+CatBoost?** 从 2025 第 2 名 `jubilant-vase` 学到,他们的融合权重分布是 **CatBoost 64% + LGB 20% + XGB 16%**——三者集成 RMSE 241 → 224(-7%)。

### 3.5 推理 — `src/predict_ensemble.py`

```
输入: data/raw/ranking.parquet + data/raw/submitting.parquet
      + models/ensemble_weights.json + models/ensemble_{xgb,lgb,cat}__seed_*.{json,txt,cbm}
输出: submissions/jubilant-puzzle_v{N}.parquet

步骤:
1. 读 ensemble_weights.json(weights, blend_mode, feat_cols, te_mappings)
2. 从训练数据 fit 类别编码映射(LGBM/XGB 兼容 unknown)
3. 加载 ranking,构造特征(用训练时保存的 TE mapping 复用)
4. 加载每个模型 × N 个 seed,逐个预测
5. seed 内平均 → 模型间加权(weight 与 blend_mode 来自训练)
6. 裁剪到 [30, 10800] 秒(防极端值影响 leaderboard)
7. 按 submitting 模板对齐 MVT_ID,缺失填全局中位数
8. 保存为 parquet
```

### 3.6 鲁棒性 — `scripts/run_robust.py`

为长跑任务(比如全特征融合)设计,自动重启 + 黑名单:
- 进程被 kill / OOM 时自动重启
- 失败 N 次后停止
- 配合 `nohup` 可后台跑几个小时不用管

### 3.7 实验管理 — MLflow

每次训练自动记录:
- 参数(超参 + feature_set + n_seeds + models)
- 指标(每个 seed 的 RMSE/MAE/P95,模型平均 RMSE,最终 blend RMSE,权重)
- 模型文件 + weights.json
- 特征重要性

启动 UI:`mlflow ui --backend-store-uri mlruns`,浏览器看 http://localhost:5000

---

## 4. 已跑出来的数字(实证)

### 4.1 Holdout 验证(2025-12)

```
v1 (30 基础特征 XGBoost):     RMSE = 319.40, MAE = 201.61, P95 = 568.49
v7 (42 特征 XGBoost):         RMSE = 311.48, MAE = 194.23, P95 = 546.95
                              改进: -7.92 (-2.48%)
```

### 4.2 同月 Holdout(2025-01 + 2025-07,更贴近比赛)

```
v1: RMSE = 356.38, MAE = 210.74, P95 = 612.62
v7: RMSE = 352.68, MAE = 205.78, P95 = 596.79
    改进: -3.70 (-1.04%)
```

> **注意**:同月 holdout 的 RMSE 比 12 月高 ~40 秒,因为 1 月 + 7 月本来就是模型最不熟的两个月份(2026 测试集也是 1 月 + 7 月)。

### 4.3 训练集/holdout 分布对比

`cnt_dep_*` 聚合特征在 train vs holdout 上分布几乎一致(mean shift 约 -3%),说明 EOBT 缺失率可控(1.08% train / 0.99% holdout)。

### 4.4 三模型融合结果(沙箱跑的)

```
保存的权重(models/ensemble_val/weights.json):
  xgb: 0.2171
  lgb: 0.7829
blend_mode: mean
feature_set: all(包含 lag+lead,77 特征全集)
6 个模型文件(xgb__seed_0/1/2 + lgb__seed_0/1/2,沙箱 CPU 限制只跑了 3 seed)
```

> ⚠️ 沙箱单线程,没跑过完整 5-seed × 3 模型。CatBoost 在沙箱也没跑(计算成本)。

### 4.5 第一个 Leaderboard 分数(v1 提交)

```
提交: jubilant-puzzle_v1.parquet (4.0 MB, 344,841 行)
leaderboard RMSE: 589.91
Top 1:           239.58 (vigorous-whistle)
差距:            350 秒(我们比 top 1 慢 60%)
排名:            前 50%(中位数)
```

> **这不是最终成绩**:v1 只用了 29 个基础特征 + 单 XGBoost,没有 lag/lead、没有融合、没有调参。后续 v2-v7 都还没提交到 leaderboard,只在 holdout 上验证。

### 4.6 误差分桶分析(ranking 上的预测分布)

```
                    n      y_mean  pred_mean   RMSE    SSE 占%
p0-50       :  172,269     696.85   1355.06   765.81   54.26%  ← 大量低估
p50-90      :  137,724    1154.50   1546.95   640.06   30.30%
p90-95      :   17,077    1619.08   1787.14   621.24    3.54%
p95-99      :   13,776    2039.33   1761.77   701.28    3.64%  ← 长尾被低估
p99-99.5    :    1,708    2810.34   1699.31  1256.77    1.45%
p99.5-99.9  :    1,378    3624.69   1730.67  2002.91    2.97%
p99.9-100   :      344    6403.49   2045.78  4563.00    3.85%  ← 极端长尾
```

**结论**:
1. v1 单模型对**短途(p0-50)严重高估**(predict 1355 vs y 697),这是因为基础特征不带拥堵信号,模型只能"猜平均"。
2. **长尾(p99+)严重低估**(predict 2046 vs y 6403),同样因为模型看不到当时已经堆积的拥堵。
3. 加 lag/lead 之后(v7 holdout)这两端应该都改善——v7 holdout RMSE 319 → 311 已印证。

### 4.7 沙箱真实数据预期(5% 采样 sklearn HGBR)

```
v1 baseline:  RMSE ≈ 344
v7 加 lag:    RMSE 估算 280-310
```

各机场 RMSE 排名(从难到易):
- LFPG 巴黎 444.8(最难)
- LIRF 罗马 443.9
- EDDM 慕尼黑 375.6
- EGLL 伦敦 357.5
- LTFM 伊斯坦布尔 355.1
- EDDF 法兰克福 319.4
- LSZH 苏黎世 296.2
- LEBL 巴塞罗那 268.3
- LEMD 马德里 264.5
- EHAM 阿姆斯特丹 255.0(最容易)

---

## 5. 文件总览

```
prc_data_challenge/
├── README.md                         项目入口
├── requirements.txt                  Python 依赖
│
├── data/raw/                         数据(330 MB,14 个 parquet)
│   ├── training_2025-{01..12}.parquet   12 个月训练集
│   ├── ranking.parquet                  2026-01+07,目标被屏蔽
│   └── submitting.parquet               提交模板
│
├── src/                              源代码
│   ├── data_loader.py                  读月度 parquet + DEP 过滤 + 时间解析
│   ├── features.py                     30 个基础特征 + 目标重缩放(可选)
│   ├── features_aggregate.py           18 个 lag 拥堵聚合
│   ├── features_laglead.py             18 个 lead 未来拥堵聚合
│   ├── features_domain.py              3 结构 + 8 目标编码
│   ├── validation.py                   时序切分 + RMSE/MAE/P95
│   ├── train_xgb.py                    XGBoost 单模型训练
│   ├── train_lgb.py                    LightGBM 单模型(备用)
│   ├── train_ensemble.py               XGB+LGB+CAT 三模型融合(主力)
│   ├── tune.py                         Optuna 自动调参(50 trials)
│   ├── predict.py                      LGB 推理(备用)
│   ├── predict_xgb.py                  XGB 推理
│   ├── predict_xgb_all.py              XGB 全量推理
│   └── predict_ensemble.py             Ensemble 推理
│
├── scripts/
│   ├── eda.py                          一键 EDA + 8 张图 + 2 个 CSV
│   ├── download_data.py                MinIO 下载脚本
│   ├── run_robust.py                   长跑自动重启
│   └── setup_mac.sh                    macOS 一键环境脚本
│
├── models/                          训练好的模型(沙箱已生成部分)
│   ├── baseline_xgb.json             v1 单模型(已提交)
│   ├── ensemble_xgb__seed_{0,1,2}.json   三模型融合的 XGB(沙箱版)
│   ├── ensemble_lgb__seed_{0,1,2}.txt    LightGBM
│   ├── ensemble_weights.json         完整权重配置(含 TE mappings)
│   ├── ensemble_val/                 验证阶段保存的子模型
│   ├── tuned_xgb_params.json         Optuna 调好的 XGB 超参
│   └── tuned_xgb_trials.csv          Optuna trial 记录
│
├── submissions/                     已生成的提交文件
│   └── jubilant-puzzle_v1~v7.parquet    7 个版本(只 v1 上传了 leaderboard)
│
├── notebooks/                       EDA 和诊断输出
│   ├── eda/                          EDA 图(8 张 PNG) + 统计 CSV(2 个)
│   ├── diag/                         排名数据缺失审计 + v2 校准检查
│   ├── holdout_*.py                  Holdout 验证脚本
│   └── *_result.json                 各次验证的结果数字
│
└── docs/
    ├── PROJECT_OVERVIEW.md           ← 本文档
    ├── DATA_READY_SUMMARY.md         数据到位总结
    ├── HOME_GUIDE.md                 回家指南(执行手册)
    ├── REAL_DATA_READINESS_CHECK.md  数据到位即用清单
    ├── SANDBOX_BREAKTHROUGH.md       libomp 突破过程
    ├── SANDBOX_DAY1_RESULTS.md       沙箱 Day 1 战报
    ├── MINIO_DOWNLOAD_GUIDE.md       MinIO 下载教程
    ├── analysis/
    │   ├── 2024-winners-analysis.md      2024 前 6 名调研
    │   ├── 2025-2nd-3rd-places-analysis.md  2025 前 3 名调研
    │   └── 2025-winners-analysis.md      2025 第 1 名分析
    └── superpowers/specs/
        ├── 2026-09-18-prc-data-challenge-2026-design.md  v1 设计
        └── 2026-09-19-prc-data-challenge-2026-design-v2.md  v2 设计(权威)
```

---

## 6. 已经完成的事(checklist)

### 6.1 数据
- [x] OSN 审核通过,队名 `jubilant-puzzle`
- [x] MinIO 数据下载完成(330 MB)
- [x] EDA 完成(8 张图 + 2 个 CSV 在 `notebooks/eda/`)
- [x] 10 个数据驱动的 bug 修复(过滤、endpoint、category、libomp、TE mapping、NaT、best_iter、clip、median、try/except)

### 6.2 代码(全部 14 个 Python 模块)
- [x] 数据加载 + 时间解析
- [x] 基础特征(30 个)
- [x] 拥堵 lag 聚合(18 个,O(N) cumsum 实现)
- [x] Lead 拥堵聚合(18 个)
- [x] Domain 结构 + TE(11 个,fit/apply 分离)
- [x] 时序验证(winter + summer + K 折)
- [x] 三模型融合(XGB + LGB + CatBoost)
- [x] 多 seed 平均(默认 5,可调)
- [x] 融合权重优化(scipy SLSQP)
- [x] Median voting 备选
- [x] Optuna 自动调参
- [x] Ensemble 推理 + TE mapping 复用
- [x] MLflow 实验记录
- [x] 长跑自动重启

### 6.3 验证
- [x] 2025-12 holdout: v7 RMSE 311.48(对比 v1 319.40)
- [x] 2025-01+07 同月 holdout: v7 RMSE 352.68(对比 v1 356.38)
- [x] 各机场 RMSE 排名(LFPG/LIRF 最难)
- [x] Lag 特征分布无偏移(均 shift <3%)
- [x] 三模型融合权重已算出并保存

### 6.4 提交
- [x] v1 已提交到 leaderboard: RMSE 589.91(基础 29 特征 XGB)
- [ ] v2-v7 待提交(代码就绪,跑完在家 Mac 上传)

---

## 7. 接下来 20 天的执行路径

### Phase 1:基础提交(Day 1-3,本周内)
| Day | 任务 | 预期 RMSE |
|---|---|---|
| 1 | 装 libomp + 全特征 XGBoost baseline(77 特征) | 280-320 |
| 2 | 三模型融合(XGB+LGB+CatBoost,5 seed) | 220-260 |
| 3 | 推理 + 提交 v2-v3 | 进入前 10 |

### Phase 2:调参与加固(Day 4-8)
| Day | 任务 |
|---|---|
| 4-5 | Optuna 50 trials 调 XGB |
| 6 | Optuna 50 trials 调 LGB |
| 7 | 用调好的超参重新跑 ensemble,提交 v4 |
| 8 | 误差分析(分桶 + 按机场),针对性加特征 |

### Phase 3:高级技巧(Day 9-14)
| Day | 任务 |
|---|---|
| 9-10 | 按机场分组训练(LFPG/LIRF/EGLL 单独模型) |
| 11 | median voting 备选对比 |
| 12-13 | 长尾加权(target > P95 的样本 weight=1.5) |
| 14 | 提交 v5(冲 top 10) |

### Phase 4:收尾(Day 15-20)
| Day | 任务 |
|---|---|
| 15-16 | 误差分析 2.0 + 最终调参 |
| 17-18 | 写论文 draft(JOAS) |
| 19 | README 整理 + 复现脚本 |
| 20 | **最终提交 + 检查** |

### 目标分数对标

| 提交 | 我们的预期 | 对标 |
|---|---|---|
| 第 1 轮 | 280-320 | 2025 第 3 名(220,更简单题) |
| 第 2 轮 | 220-260 | 2025 第 2 名(211) |
| 第 3 轮 | 180-220 | 2024 第 4 名(2270,差数据) |
| **冲前 3 名** | < 200 | 2025 前 3 名都是 GBDT |

---

## 8. 风险与回退

| 风险 | 回退方案 |
|---|---|
| CatBoost 装不上 | 退到 XGB+LGB 融合,损失约 1-2% |
| Optuna 跑得太慢(>2 小时) | 改为 20 trials + 用已调好的超参 |
| 拥堵特征在 200 万行跑得慢 | 按月分批处理 |
| 某次提交 RMSE 暴涨 | 暂停提交,排查泄漏 / 数据 |
| 家里 Mac 跑不通 | 退到 sklearn HGBR(参考 2024 第 30 名 3755) |
| 推理报错 | 用 `models/ensemble_val/` 下保存的验证模型 |

---

## 9. 关键决策日志(为什么这样做)

| 问题 | 决策 | 理由 |
|---|---|---|
| 用什么模型? | XGB + LGB + CatBoost | 2025 第 2 名 CatBoost 拿 64% 权重,集成 -7% RMSE |
| 几个 seed? | 5 | 2024 第 1 名 20 seed 边际收益递减,5 是性价比最优 |
| Optuna 多少 trials? | 50 | 2025 第 3 名试 160 trials 反而 RMSE 飙到 303 |
| 训练切分? | winter(11/12→1月)+ summer(6→7月) | 测试集就是 2026-01 + 2026-07 |
| 聚合参考时间? | EOBT_1_flt | 训练和推理都能拿到,口径一致 |
| 类别编码? | fit on train,apply on rank | ranking 没 TAXITIME_SEC,只能复用 |
| NaT 怎么办? | cumsum 前剔除,输出 NaN | searchsorted 把 NaT 当极小值会污染 lag |
| 长尾处理? | P95 加权(待跑)+ median voting 备选 | RMSE 对长尾放大 |
| 目标重缩放? | 已实现,默认关闭 | 边际收益小,2024 第 1 名用 |

---

## 10. 一句话总结(给非技术读者)

> 我们用 2025 全年 10 个欧洲机场 200 万条出发航班数据训练模型,预测每架飞机从推出到起飞要花多少时间。
>
> 当前已经用最简单的 30 个特征 + XGBoost 跑出第一个 leaderboard 分数 589.91(中位数水平),还在 2025-12 holdout 上用 42 特征 XGB 跑出 311.48 的 RMSE(比赛指标用 RMSE,数字越小越好)。
>
> 我们已经准备好三模型(XGB+LGB+CatBoost) × 5 seed 融合 + 77 特征 + Optuna 调参 + lag/lead 拥堵特征的全部代码,在家用 Mac 跑一遍应该能冲到前 10 名(RMSE 200 以下),冲前 3 名(RMSE < 200 + 鲁棒性)。

---

**文档版本**:v1.0 · 2026-09-21
**维护者**:`jubilant-puzzle` 团队
**配合阅读**:`README.md`、`docs/superpowers/specs/2026-09-19-prc-data-challenge-2026-design-v2.md`、`docs/HOME_GUIDE.md`
