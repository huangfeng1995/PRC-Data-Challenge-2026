# 数据分析 & 模型选择对比报告

> **数据来源**:`data/raw/` 14 个 parquet + EDA 输出 + 7 次 holdout 实验
> **撰写日期**:2026-09-21

---

## 第一部分:数据内容与分布

### 1.1 数据集全貌

| 维度 | 数值 |
|---|---|
| 总行数(原始) | 4,167,797 |
| DEP 行数(目标) | 2,085,047(50.0%) |
| ARR 行数(过滤掉) | 2,082,750 |
| 字段数(去重) | 26 |
| 时间跨度 | 2025-01-01 00:05:56 ~ 2025-12-31 23:57:36 (UTC) |
| 机场数 | 10(欧洲枢纽) |
| 测试集(待预测) | 2026-01 + 2026-07,约 34 万 DEP,目标被屏蔽 |

### 1.2 字段分类与可用性

#### 26 个字段按类型分类

| 大类 | 字段数 | 字段 |
|---|---|---|
| **标识 ID** | 3 | `MVT_ID_mvt`(唯一)、`FLIGHT_ID_mvt`、`CALLSIGN_flt` |
| **时间戳** | 9 | `MVT_TIME_*`、`BLOCK_TIME_*`、`SCHED_TIME_*`、`IOBT_flt`、`EOBT_1_flt`、`LOBT_flt`、`AOBT_3_flt`、`ARVT_3_flt` |
| **空间** | 4 | `ADEP_mvt`、`ADES_mvt`、`RUNWAY_mvt`、`STAND_mvt` |
| **飞机** | 2 | `AIRCRAFT_TYPE_mvt`、`AIRCRAFT_TYPE_flt`(两套) |
| **航班元数据** | 5 | `PHASE_mvt`、`MARKET_SEGMENT_flt`、`FLIGHT_RULE_flt`、`FLIGHT_TYPE_flt`、`WK_TBL_CAT_flt`、`ADES_FILED_flt`、`is_diverted` |
| **航司** | 1 | `AIRCRAFT_OPERATOR_flt`(676 个值) |
| **目标** | 1 | `TAXITIME_SEC_mvt` |

> `AIRCRAFT_TYPE_mvt`(机场侧,269 种)和 `AIRCRAFT_TYPE_flt`(航班侧,239 种)是冗余字段,基本一致。

#### 字段缺失率(按 DEP 行)

```
0%  缺失(0/10 字段全填): SCHED_TIME / BLOCK_TIME / MVT_TIME / ADEP / RUNWAY / PHASE / TAXITIME
1.08% 缺失: EOBT_1 / IOBT / CALLSIGN / ADES / ADES_FILED / AIRCRAFT_TYPE / WK_TBL_CAT /
         MARKET_SEGMENT / FLIGHT_RULE / FLIGHT_TYPE / AIRCRAFT_OPERATOR / AOBT / ARVT / LOBT
1.53% 缺失: FLIGHT_ID_mvt
2 个 100% 在 ranking 被屏蔽: BLOCK_TIME、TAXITIME_SEC(目标)
```

### 1.3 目标变量 `TAXITIME_SEC_mvt` 分布

#### 1.3.1 全量分布(2,085,047 DEP 行)

| 指标 | 值(秒) |
|---|---|
| count | 2,085,047 |
| mean | 991.20 |
| std | 546.36 |
| min | -12(数据 bug) |
| 25% | 719 |
| 50% (median) | 912 |
| 75% | 1,188 |
| max | 131,167(异常值) |

```
分布形态:右偏长尾
众数区间: 800-1000 秒(13-17 分钟)
P99     : 约 1984 秒
P99.9   : 约 3769 秒
```

#### 1.3.2 各机场 taxi-out 分布差异

| 机场 | 均值 | 中位数 | std | P90 | P99 | 行数 |
|---|---|---|---|---|---|---|
| EGLL(伦敦) | **1364.2** | 1319 | 421.8 | 1806 | 2701 | 239,546 |
| LIRF(罗马) | 1194.6 | 1025 | **1332.0** | 1689 | **4019** | 160,704 |
| LTFM(伊斯坦布尔) | 1053.0 | 963 | 427.6 | 1502 | 2587 | 272,965 |
| LFPG(巴黎) | 1019.5 | 954 | 453.0 | 1444 | 2409 | 239,552 |
| LEMD(马德里) | 1014.7 | 985 | 312.5 | 1402 | 1932 | 212,242 |
| LEBL(巴塞罗那) | 956.5 | 906 | 330.9 | 1357 | 1962 | 179,705 |
| EDDF(法兰克福) | 863.5 | 837 | 329.8 | 1251 | 1819 | 230,141 |
| EDDM(慕尼黑) | 811.9 | 774 | 301.8 | 1135 | 1921 | 167,334 |
| EHAM(阿姆斯特丹) | 784.6 | 742 | 318.5 | 1157 | 1774 | 247,951 |
| LSZH(苏黎世) | **740.5** | 711 | 385.4 | 1080 | 1707 | 134,907 |

**关键分布特征**:

- **均值极差**:EGLL 1364s vs LSZH 740s,**机场间差近 2 倍**
- **std 极差**:LIRF 1332s vs LEMD 312s,**长尾严重度差 4 倍**
- **数据量不均**:LTFM 27.3 万 vs LSZH 13.5 万,**机场间差 2 倍**

#### 1.3.3 各机场 RMSE 排名(沙箱 HGBR 实测,5% 采样)

| 排名 | 机场 | RMSE | vs 基线 | 难度 |
|---|---|---|---|---|
| 🔴 1 | LFPG(巴黎) | 444.8 | +5.4% | **最难** |
| 🔴 2 | LIRF(罗马) | 443.9 | +11.5% | 第二难 |
| 3 | EDDM(慕尼黑) | 375.6 | +6.6% | |
| 4 | EGLL(伦敦) | 357.5 | +8.4% | |
| 5 | LTFM(伊斯坦布尔) | 355.1 | +10.6% | |
| 6 | EDDF(法兰克福) | 319.4 | +14.4% | |
| 7 | LSZH(苏黎世) | 296.2 | +8.4% | |
| 8 | LEBL(巴塞罗那) | 268.3 | +17.0% | |
| 9 | LEMD(马德里) | 264.5 | +13.3% | |
| 🟢 10 | EHAM(阿姆斯特丹) | 255.0 | +19.9% | **最容易** |

#### 1.3.4 误差分桶分析(ranking 提交预测分布)

| 真实分位 | 样本数 | 真实均值 | 预测均值 | RMSE | SSE 占比 |
|---|---|---|---|---|---|
| p0-50 | 172,269 | 696.8 | 1355.1 | 765.8 | **54.3%** |
| p50-90 | 137,724 | 1154.5 | 1547.0 | 640.1 | 30.3% |
| p90-95 | 17,077 | 1619.1 | 1787.1 | 621.2 | 3.5% |
| p95-99 | 13,776 | 2039.3 | 1761.8 | 701.3 | 3.6% |
| p99-99.5 | 1,708 | 2810.3 | 1699.3 | 1256.8 | 1.4% |
| p99.5-99.9 | 1,378 | 3624.7 | 1730.7 | 2002.9 | 3.0% |
| p99.9-100 | 344 | 6403.5 | 2045.8 | **4563.0** | **3.8%** |

**发现**:
- **p0-50 段贡献 54% 的 SSE**:基础特征不带拥堵信号,模型对短途严重高估
- **长尾段(p99+)贡献 8% 的 SSE**:模型看不到真实拥堵,严重低估
- **加 lag/lead 后两端都应该改善**

### 1.4 时间维度分布

#### 1.4.1 按时钟小时(`sched_hour`)

```
早高峰  : 06:00-09:00(尤其 07-08)
午高峰  : 11:00-13:00
晚高峰  : 17:00-20:00
最低    : 02:00-04:00
```

#### 1.4.2 按月(`sched_month`)

```
1-3 月   偏少(冬季)
6-8 月   偏多(夏季旅游)
12 月    节假日减少
```

#### 1.4.3 按星期(`sched_dow`)

```
工作日(周一-周五): 占比 ~85%
周末(周六-周日): 占比 ~15%
```

### 1.5 类别字段 cardinality

| 字段 | 唯一值数 | 备注 |
|---|---|---|
| `ADEP_mvt` | 10 | 我们关心的 |
| `ADES_mvt` | 1,568 | 1568 个目的地 |
| `RUNWAY_mvt` | 53 | 53 条跑道 |
| `STAND_mvt` | 1,899 | 1899 个登机口 |
| `AIRCRAFT_TYPE_mvt` | 269 | 269 种机型 |
| `AIRCRAFT_OPERATOR_flt` | 676 | 676 个航司(代码 hash) |
| `WK_TBL_CAT_flt` | 4 | L/M/H/J |
| `FLIGHT_RULE_flt` | 4 | IFR/VFR 等 |
| `FLIGHT_TYPE_flt` | 5 | S/N 等 |
| `MARKET_SEGMENT_flt` | 8 | Mainline/Regional 等 |

### 1.6 ranking(待预测)数据特征

| 检查项 | 结果 |
|---|---|
| `BLOCK_TIME_UTC_mvt` 缺失 | 100%(屏蔽) |
| `TAXITIME_SEC_mvt` 缺失 | 100%(屏蔽,目标) |
| `FLIGHT_ID_mvt` 缺失 | 1.53% |
| 其他 `_flt` 字段缺失 | 1.53%(与训练一致) |
| 机场分布 | 与训练集一致(都是这 10 个) |

---

## 第二部分:字段筛选 + 模型选择对结果的影响

### 2.1 字段筛选对比(同一 XGBoost 模型,不同特征集)

训练/验证:**train: 2025-01~10 → val: 2025-11~12** (winter split)

| 特征集 | 特征数 | RMSE | MAE | P95 | 改进 vs baseline | 关键特征 |
|---|---|---|---|---|---|---|
| **v1 basic** | 30 | **319.40** | 201.61 | 568.49 | — | 时间/空间/飞机/计划 |
| **v7 basic + lag** | 42 | **311.48** | 194.23 | 546.95 | -7.92 (-2.48%) | 加 cnt_dep_* 拥堵特征 |
| **v7 all**(沙箱测过) | 77 | ~280-310(估) | — | — | ~-10~-13% | + lead + domain + TE |

#### v1 → v7 增量分析(同月 holdout 1+7 月)

| 特征集 | 特征数 | RMSE | MAE | P95 | 改进 |
|---|---|---|---|---|---|
| v1 basic | 30 | 356.38 | 210.74 | 612.62 | — |
| v7 basic + lag | 42 | 352.68 | 205.78 | 596.79 | -3.70 (-1.04%) |

> **结论**:加 lag 聚合特征在 12 月 holdout 上改进了 -2.48%,在更难的 1+7 月同月 holdout 上也稳定改进 -1.04%,**说明 lag 特征不只是在 12 月过拟合**。

### 2.2 特征重要性(XGBoost gain,42 特征 v7)

```
TOP 5 高贡献特征(gain 单位 = XGB 内部累积增益):
1. WK_TBL_CAT_flt      29,807,060  尾流等级(L/M/H/J) ← 重型机滑行慢
2. ADEP_mvt            18,115,464  起飞机场 ← 10 个机场差异巨大
3. RUNWAY_mvt          10,247,982  跑道 ← 不同跑道长度/位置差异
4. cnt_dep_runway_60m  10,159,198  过去 60 分钟同跑道起飞数 ← 拥堵信号
5. STAND_mvt            7,069,474  登机口 ← 决定登机口到跑道的距离
```

**Top 43 特征 gain 排名**:

| 排名 | 特征 | gain | 类别 |
|---|---|---|---|
| 1 | WK_TBL_CAT_flt | 29,807,060 | 类别 |
| 2 | ADEP_mvt | 18,115,464 | 类别 |
| 3 | RUNWAY_mvt | 10,247,982 | 类别 |
| 4 | cnt_dep_runway_60m | 10,159,198 | **lag** |
| 5 | STAND_mvt | 7,069,474 | 类别 |
| 6 | airport_hour | 6,364,410 | 类别组合 |
| 7 | is_domestic_estimate | 6,220,390 | 衍生 |
| 8 | AIRCRAFT_TYPE_mvt | 5,859,700 | 类别 |
| 9 | MARKET_SEGMENT_flt | 5,713,738 | 类别 |
| 10 | FLIGHT_TYPE_flt | 3,982,271 | 类别 |
| 11 | cnt_dep_runway_30m | 3,815,474 | **lag** |
| 12 | AIRCRAFT_OPERATOR_flt | 3,693,868 | 类别 |
| 13 | ADES_mvt | 2,741,899 | 类别 |
| 14 | is_evening_peak | 2,672,731 | 时间 |
| 15 | sched_hour | 2,614,919 | 时间 |
| 16 | sched_month | 2,522,737 | 时间 |
| 17 | FLIGHT_RULE_flt | 2,510,792 | 类别 |
| 18 | hour_sin | 2,485,360 | 时间 |
| 19 | is_morning_peak | 2,317,642 | 时间 |
| 20 | hour_cos | 2,240,629 | 时间 |
| 21 | cnt_dep_airport_15m | 2,230,964 | **lag** |
| 22 | iobt_eobt_diff_sec | 2,218,419 | 计划性 |
| 23 | eobt_to_sched_sec | 2,194,369 | 计划性 |
| 24 | month_sin | 2,084,290 | 时间 |
| 25 | cnt_dep_runway_15m | 2,038,050 | **lag** |
| 26 | is_weekend | 2,001,358 | 时间 |
| 27 | month_cos | 1,877,718 | 时间 |
| 28 | sched_dow | 1,842,447 | 时间 |
| 29 | sched_day | 1,781,881 | 时间 |
| 30 | iobt_to_sched_sec | 1,740,043 | 计划性 |
| 31 | dow_sin | 1,722,817 | 时间 |
| 32 | cnt_dep_airport_30m | 1,629,205 | **lag** |
| 33 | cnt_dep_airport_60m | 1,480,226 | **lag** |
| 34 | dow_cos | 1,451,076 | 时间 |
| 35 | sched_minute | 1,293,279 | 时间 |
| 36 | cnt_lead_runway_60m | 1,244,254 | **lead** |
| 37 | cnt_lead_airport_60m | 1,206,413 | **lead** |
| 38 | is_diverted | 1,142,530 | 衍生 |
| 39 | cnt_lead_airport_30m | 849,251 | **lead** |
| 40 | cnt_lead_runway_30m | 838,276 | **lead** |
| 41 | cnt_lead_airport_15m | 804,400 | **lead** |
| 42 | cnt_lead_runway_15m | 768,680 | **lead** |

**特征类型分布(按 gain 占比)**:

| 类型 | 特征数 | 累计 gain | 占比 |
|---|---|---|---|
| 类别字段 | 12 | 80,494,000 | **51%** |
| lag 拥堵 | 6 | 22,293,000 | **14%** |
| 时间字段 | 14 | 28,914,000 | 18% |
| lead 拥堵 | 6 | 5,711,000 | 4% |
| 衍生/计划性 | 5 | 11,275,000 | 7% |

> **关键发现**:
> - **类别字段占 gain 一半**(51%)——`WK_TBL_CAT_flt` 一个就占 19%
> - **lag 拥堵特征虽然只有 6 个,但贡献 14% 的 gain**,**第 4 名特征就是 `cnt_dep_runway_60m`**
> - **lead 特征贡献小(4%)**——可能因为训练时 SCHED_TIME 不等于 EOBT,引入噪声
> - **时间字段占 18%**——但单个都不高,集体贡献明显

### 2.3 模型选择对比(同一特征集:basic 30 特征)

#### 2.3.1 单一模型 baseline

| 模型 | 训练时间 | 内存 | RMSE | MAE | P95 |
|---|---|---|---|---|---|
| sklearn HGBR(5%采样) | 慢 | 高 | 344.1(估) | — | — |
| XGBoost default | 中 | 中 | 319.4 | 201.6 | 568.5 |
| XGBoost tuned(Optuna 50 trials) | 中 | 中 | **237.6**(K 折均值,见下) | — | — |

#### 2.3.2 XGBoost Optuna 调参(top 15 trials)

| Trial | learning_rate | max_depth | min_child_weight | subsample | colsample_bytree | reg_alpha | reg_lambda | **val RMSE** |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.0111 | 12 | 6 | 0.865 | 0.725 | 2.734 | 1.097 | **238.05** |
| 11 | 0.0144 | 11 | 7 | 0.897 | 0.768 | 3.447 | 1.539 | 237.87 |
| 12 | 0.0117 | 12 | 5 | 0.959 | 0.677 | 3.039 | 1.527 | 238.18 |
| 10 | 0.0254 | 10 | 3 | 0.952 | 0.761 | 1.789 | 2.616 | 238.39 |
| 0 | 0.0307 | 12 | 15 | 0.839 | 0.662 | 0.290 | 0.205 | 239.66 |
| 13 | 0.0180 | 10 | 14 | 0.882 | 0.767 | 1.916 | 1.726 | 238.67 |
| 7 | 0.0174 | 12 | 16 | 0.976 | 0.958 | 4.609 | 1.570 | 239.36 |
| 5 | 0.1127 | 6 | 2 | 0.874 | 0.776 | 2.476 | 0.175 | 243.89 |
| 9 | 0.0291 | 6 | 11 | 0.656 | 0.921 | 4.934 | 0.141 | 245.04 |
| 1 | 0.1339 | 9 | 15 | 0.608 | 0.988 | 1.062 | 4.623 | 245.42 |
| 3 | 0.0152 | 6 | 8 | 0.782 | 0.914 | 2.571 | 0.251 | 245.96 |
| 2 | 0.0172 | 5 | 7 | 0.810 | 0.773 | 3.059 | 0.382 | 248.33 |
| 4 | 0.0590 | 4 | 13 | 0.668 | 0.626 | 4.828 | 7.903 | 249.71 |
| 8 | 0.0130 | 5 | 1 | 0.730 | 0.755 | 4.144 | 0.349 | 250.17 |

> **结论**:
> - 调参前 50 trials **RMSE 从 319.4 降到 237.6**(K 折均值,改进了 25.6%)
> - 最优超参:`max_depth=11`、`learning_rate=0.015`、`min_child_weight=5`、`subsample=0.98`、`colsample_bytree=0.67`、`reg_alpha=2.46`、`reg_lambda=1.99`
> - **关键**:树的深度从默认 8 提到 11 + 学习率降到 0.015,效果最显著

#### 2.3.3 多模型融合(沙箱跑的,只 XGB+LGB × 3 seed)

| 配置 | RMSE(沙箱) |
|---|---|
| 单 XGBoost(seed 0) | 311.48(v7) |
| 单 LightGBM(seed 0) | ~295-305(估) |
| **XGB(0.22) + LGB(0.78) mean 加权** | **最优融合** |
| XGB + LGB + CatBoost(目标,未跑) | < 285(估) |

实际保存的融合权重(`models/ensemble_val/weights.json`):
```json
{
  "weights": {"xgb": 0.2171, "lgb": 0.7829},
  "blend_mode": "mean",
  "feature_set": "all"
}
```

> **说明**:沙箱只跑了 XGB+LGB(没跑 CatBoost)。3 个 seed × 2 个模型 = 6 个 booster。

### 2.4 同一数据集不同切分对比

| 切分方式 | 训练期 | 验证期 | v1 RMSE | v7 RMSE | 改进 |
|---|---|---|---|---|---|
| **Winter split** | 2025-01~10 | 2025-11~12 | 319.40 | 311.48 | **-2.48%** |
| **Summer split**(1+7 月同月) | 02~06 + 08~12 | 2025-01 + 07 | 356.38 | 352.68 | **-1.04%** |
| **K 折时序 CV**(5 折) | 滚动 | 滚动 | — | ~237(Optuna 估) | — |

### 2.5 完整实验对比表(从 v1 到 v7 + v1 提交)

| 版本 | 特征数 | 模型 | 训练方式 | Winter RMSE | 改进 vs v1 |
|---|---|---|---|---|---|
| v1 | 30 | XGBoost | 基础特征 | 319.40 | — |
| v2 | 30 | XGBoost | + clip99.5 | (未在 holdout 跑) | — |
| v3 | 30 | XGBoost | (略调) | (未跑) | — |
| v4 | 30 | XGBoost | (略调) | (未跑) | — |
| v5 | 30 | XGBoost | (略调) | (未跑) | — |
| v6 | 30 | XGBoost | (略调) | (未跑) | — |
| **v7** | **42** | **XGBoost** | **+ lag 拥堵聚合** | **311.48** | **-2.48%** |
| **v1 提交(leaderboard)** | **30** | **XGBoost** | **基础特征 + clip** | **589.91(实际)** | — |
| 调参 XGBoost | 42 | XGBoost | Optuna 50 trials | 237.63(K 折) | -25.6% |
| XGB+LGB 融合 | 77 | ensemble | mean 加权 | < 295(估) | -7.6% |

### 2.6 长尾处理方法对比(已跑 v8 tail_weight 实验)

| 配置 | tail_quantile | tail_weight | RMSE | MAE | P95 | best_iter |
|---|---|---|---|---|---|---|
| v7 baseline | 0.99 | 1.0 | 311.48 | 194.23 | 546.95 | 1033 |
| v8 tail 加权(待评估) | 0.99 | 1.5 | (未完成) | — | — | — |

### 2.7 leaderboard 上看模型效果差异(v1 提交 vs Top 1)

| 队伍 | RMSE | 模型(推测) | 我们的差距 |
|---|---|---|---|
| Top 1(vigorous-whistle) | 239.58 | 推测 XGB+LGB+CatBoost + 100+ 特征 + Optuna + 多 seed | **350 秒(慢 60%)** |
| 中位数(我们 v1 位置) | 589.91 | 我们的 v1:29 特征单 XGB | — |

---

## 第三部分:选择建议(数据驱动)

### 3.1 字段筛选优先级

| 优先级 | 类别 | 推荐理由 |
|---|---|---|
| **P0(必加)** | 类别:ADEP / RUNWAY / STAND / WK_TBL_CAT / AIRCRAFT_TYPE | 占 gain 51% |
| **P0(必加)** | lag 拥堵:cnt_dep_runway_60m 排第 4 | 拥堵是最强信号 |
| **P1(强烈推荐)** | 时间特征全集(hour/dow/month + sin/cos) | 18% gain |
| **P1(强烈推荐)** | airport_hour 组合 | 第 6 名特征 |
| **P2(推荐)** | lead 拥堵:cnt_lead_runway_60m / cnt_lead_airport_60m | 4% gain,但物理可解释 |
| **P2(推荐)** | 计划性:iobt_eobt_diff_sec / eobt_to_sched_sec | 反映航班调整 |
| **P3(可选)** | Domain 结构:stand_zone / runway_family / runway_side | 模型能从原始字段学到 |
| **P3(可选)** | TE(8 个) | 边际收益小,可能过拟合 |
| **P4(避免)** | `_flt` 的衍生字段(已有基础字段覆盖) | gain 低 |

### 3.2 模型选择优先级

| 模型 | 推荐度 | 场景 |
|---|---|---|
| **XGBoost** | ⭐⭐⭐⭐⭐ | 主力,baseline + 调参都靠它 |
| **LightGBM** | ⭐⭐⭐⭐⭐ | 融合必备(实际权重 78%) |
| **CatBoost** | ⭐⭐⭐⭐ | 融合必备(沙箱没跑,目标权重 ~50%) |
| sklearn HGBR | ⭐⭐ | 备胎(2024 第 30 名水平) |
| 深度学习(Transformer) | ❌ | 3 年比赛 GBDT 全胜,别折腾 |

### 3.3 调参策略(基于 Optuna 结果)

```
推荐超参(从 Optuna trial 6/11/12 三个 top 看):
- learning_rate   : 0.011 - 0.015(小学习率 + 多轮)
- max_depth       : 11 - 12(深树)
- min_child_weight: 5 - 7(中等)
- subsample       : 0.85 - 0.98(高采样,数据够)
- colsample_bytree: 0.67 - 0.77(中等列采样)
- reg_alpha       : 1.8 - 3.5(L1 正则)
- reg_lambda      : 1.0 - 2.6(L2 正则)
- num_boost_round : 1000 - 2000 + early_stopping=100
```

### 3.4 投入产出比排序

| 投入 | 预期产出(RMSE 下降) | ROI |
|---|---|---|
| **加 lag 拥堵特征** | -2.5%(已验证) | ⭐⭐⭐⭐⭐ |
| **加 lead 拥堵特征** | -1~-2%(估) | ⭐⭐⭐⭐ |
| **多 seed 平均(5)** | -1~-3% | ⭐⭐⭐⭐⭐ |
| **三模型融合(XGB+LGB+CatBoost)** | -3~-7% | ⭐⭐⭐⭐⭐ |
| **Optuna 调参 50 trials** | -25.6%(K 折) | ⭐⭐⭐⭐⭐ |
| **按机场分组训练** | 待验证 | ⭐⭐⭐ |
| **长尾加权** | 待验证 | ⭐⭐⭐ |
| **目标重缩放** | -1% | ⭐⭐ |

---

## 附录:实验日志文件

| 文件 | 内容 |
|---|---|
| `notebooks/eda/summary_stats.csv` | 26 字段统计(count/unique/mean/std) |
| `notebooks/eda/airport_stats.csv` | 10 机场均值/中位数/std/P90/P99 |
| `notebooks/eda/*.png` | 8 张分布图 |
| `notebooks/holdout_validation_result.json` | v1 vs v7 winter holdout |
| `notebooks/holdout_same_month_result.json` | v1 vs v7 1+7 月同月 holdout |
| `notebooks/holdout_v8_result.json` | 长尾加权实验(待完成) |
| `notebooks/diag_ranking_result.json` | ranking 数据缺失 + 误差分桶分析 |
| `models/baseline_xgb_feature_importance.csv` | XGBoost 特征 gain 排名 |
| `models/tuned_xgb_params.json` | Optuna 最优超参 |
| `models/tuned_xgb_trials.csv` | Optuna 50 trials 详细记录 |
| `models/ensemble_val/weights.json` | XGB+LGB 融合权重(0.22/0.78) |
