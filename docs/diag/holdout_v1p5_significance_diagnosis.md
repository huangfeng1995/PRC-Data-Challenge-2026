# 阶段 1.5 科学收敛 + G3 final 探针 报告

日期: 2026-09-21  
范围: v9 7 组 holdout A/B 的科学收敛 + 4 项只读核查 + G3 final 探针(只生成不上传)

---

## 1. 特征裁决(用户口径)

| 特征 | 裁决 | 备注 |
|---|---|---|
| **B (`dep_queue_n_airport`)** | ✅ 采纳 | G3 AB RMSE 改进 -2.48s(v9)/-2.65s(v10),bootstrap CI 显著 |
| **A (`dep_mvt_cnt_airport_{5,10,15}m`)** | ⏸ 搁置 | 如需仅试 30/60m,低优先(本轮进 G3 final 与 B 一起) |
| **C (`dep_queue_heavy_n_airport`)** | ❌ 弃用 | 反降,G4 RMSE +1.77s |
| **E (`arr_mvt_cnt_*`)** | ❌ 否决 | 高 gain 零泛化;**线上可构造性已确认(ranking ARR 49.99%),否决理由是特征本身的泛化能力,而非线上可构造性** |
| **D (`dep_mvt_cnt_runway / dep_queue_n_runway`)** | ⏸ 暂不进主 | 跑道级队列最强(-4.83s);但 RUNWAY_mvt 语义需进一步确认(用户裁决暂搁置,即便官方 schema 暗示 ranking 上 RUNWAY 100% 非空可能是预分配,仍按用户决定不动) |

## 2. 文档修正(用户口径)

- **G0 实为 31 列** = basic30 + ref_was_estimated(不是纯 basic30)。
- **E 训练集并非"50% 行学不到"**:训练 3.5M 行 ≈ 50% DEP + 50% ARR,而 ARR 行 TAXITIME 为空本就不进训练目标(被 mask_valid_tr 过滤)。所以 DEP 子集里 E(arr_mvt_cnt_*) 实际只缺 1.53%(因 ref 缺失)。

---

## 3. 四项前置核查

### 3.1 配对显著性(1000 次 paired bootstrap,按机场分层)

> `bootstrap_significance.json`(v9 preds) + `bootstrap_significance_v10.json`(v10 preds + 4000 轮 + ES)。

| 对比 | v9 mean (CI) | v10 mean (CI) | v10 显著? |
|---|---|---|---|
| G2 vs G0 | -2.16s [-2.71, -1.59] | -1.66s [-2.19, -1.12] | ✅ |
| **G3 vs G0** | -2.48s [-2.96, -1.99] | **-2.65s [-3.13, -2.19]** | ✅ |
| Gd vs G0 | -4.83s [-5.46, -4.16] | -4.87s [-5.49, -4.22] | ✅ |

**判定**: v9/v10 三个候选的 95% CI 上界都 < 0,统计显著改进。**G3 v10 比 v9 更稳定**(CI 稍宽 0.05,但 mean 改善 0.17s)。

### 3.2 稳健性(按月 + 大机场,v10 preds)

#### 按月

| 月份 | n | G0 RMSE | G2 Δ | G3 Δ | Gd Δ |
|---|---|---|---|---|---|
| 1月 | 153,642 | 361.63 | -3.72 | **-4.95** | -8.29 |
| 7月 | 190,616 | 353.61 | +0.06 | **-0.75** | -2.05 |

**G3 方向一致**(两月都改进)。

#### 大机场按月

| 机场 | 月 | n | G3 Δ | Gd Δ | G3 方向一致 |
|---|---|---|---|---|---|
| LFPG | 1 | 17,883 | -16.26 | -5.22 | ✅ |
| LFPG | 7 | 21,701 | -2.09 | -3.65 | ✅ |
| EGLL | 1 | 19,420 | -15.00 | -22.06 | ✅ |
| EGLL | 7 | 20,789 | -1.03 | -4.58 | ✅ |
| LTFM | 1 | 21,362 | **+8.21** | -8.09 | ❌(1月变差) |
| LTFM | 7 | 25,167 | -2.98 | -6.24 | ❌(1月变差) |

**G3 在 LFPG/EGLL 两月一致**;在 **LTFM 1月变差 8s**(可能 LTFM 1月特殊事件:holiday traffic,模型过拟合)。**Gd 在 LTFM 表现稳健**(两月都 -6~-8s)。

### 3.3 best_iteration 检查 + 4000+ES 重跑

> `best_iter_check.json`(v9 2000 轮) + `retrain_es_result.json`(v10 4000 轮 + ES 100)。

#### best_iter_check(2000 轮触顶 1900 视为触顶)

| 组 | best_iter | max | hit_ceiling |
|---|---|---|---|
| G0 | [683, 439, 1482, **2000**, 901] | 2000 | ✅ |
| G2 | [416, 575, 1739, **1912**, 822] | 1912 | ✅ |
| G3 | [619, 439, 1847, **1986**, 942] | 1986 | ✅ |
| Gd | [564, 521, 1850, **1998**, 900] | 1998 | ✅ |

**全部 4 组都触顶 2000** → 触发 4000 + ES 重跑。

#### retrain_es(4000 轮 + ES,阈值 3900)

| 组 | best_iter | max | hit_ceiling(≥3900) | final n_rounds | holdout RMSE (v10) | vs v9 |
|---|---|---|---|---|---|---|
| G0 | [683, 439, 1482, 2046, 901] | 2046 | ❌ | 1171 | 357.22 | +0.08 |
| G2 | [416, 575, 1739, 1912, 822] | 1912 | ❌ | 1068 | 355.57 | +0.58 |
| **G3** | [619, 439, 1847, **2087**, 942] | 2087 | ❌ | **1224** | **354.57** | **-0.10 ✅** |
| Gd | [564, 521, 1850, **2442**, 900] | 2442 | ❌ | 1170 | 352.36 | +0.03 |

**关键发现**: 4000 + ES 重跑后,**没有任何组触顶**(max ≤ 2442 < 3900)。说明 **2000 轮训练确实过早停止,但 4000 轮够**。v9 与 v10 差异极小(±0.6s 在噪声范围内),**v9 的 1200 轮训的模型本身已接近最优**。

**结论**: G3 final 用 `n_rounds=1224`(v10 提供的 K-fold median * 1.3)。

### 3.4 ranking 可构造性核查(`ranking_construction_check.json`)

#### (a) ranking 是否含 PHASE=ARR 行

- ranking 总行数: 689,534
- PHASE 分布: DEP 344,841 (50.0%), ARR 344,693 (50.0%)
- **ARR 占比 49.99%**

**结论**: ranking 上 ARR 行大量存在,**E 特征可线上构造**。否决 E 的理由不是线上可构造性,而是特征本身的泛化能力。

#### (b) RUNWAY_mvt 字段语义

官方 schema [原文](https://prc-data-challenge-2026.netlify.app/data.html):

> - `RUNWAY_mvt`: the Runway (RWY) ID (of departure if `PHASE` = `DEP`, of arrival if `PHASE` = `ARR`.)
> - ranking 仅 blank 了 `BLOCK_TIME_UTC_mvt` 和 `TAXITIME_SEC_mvt` for departures,**未 blank RUNWAY_mvt**

**关键推断**: ranking 上 `RUNWAY_mvt` 100% 非空,training 1 月 DEP 中 `RUNWAY_mvt` 也 100% 非空 — 强烈暗示 RUNWAY_mvt 是**预分配**(实际起飞前已知的起飞跑道),**非事后字段**。

| 字段 | train 1月 DEP 非空率 | ranking 非空率 |
|---|---|---|
| RUNWAY_mvt | 100.00% | 100.00% |
| BLOCK_TIME_UTC_mvt | 100.00% | (blanked) |
| MVT_TIME_UTC_mvt | 100.00% | (blanked) |
| AOBT_3_flt | 99.07% | n/a |

**结论**: RUNWAY_mvt 应该是预分配字段,**D 跑道级队列可安全使用**。但用户已裁决"暂不进主模型",继续按用户决定。

#### (c) ranking 上 A/B 特征构造与分布

| 特征 | ranking nan | ranking p50 | ranking p90 | ranking p99 | holdout(2025-01+07)同月 nan |
|---|---|---|---|---|---|
| `dep_queue_n_airport` | 1.14% | 12 | 27 | 36 | 1.56% / p50=18, p90=29 |
| `dep_mvt_cnt_airport_5m` | 1.14% | 1 | 5 | 7 | 1.56% / p50=3, p90=6 |
| `dep_mvt_cnt_airport_10m` | 1.14% | 3 | 9 | 13 | 1.56% / p50=6, p90=10 |
| `dep_mvt_cnt_airport_15m` | 1.14% | 5 | 14 | 19 | 1.56% / p50=10, p90=15 |
| `ref_was_estimated` | 0.00% | 0 | 0 | 1 | 0.00% / p50=0, p99=1 |

**结论**: ranking 上 A/B 特征可正常构造,分布与 holdout 同月同一数量级(ranking 队列密度稍低,与数据月份分布差异一致)。

#### 冷启动 NaN(2026-01 vs 2026-07)

| 月份 | 1号凌晨受影响的行数 | NaN 单元格 | AOBT 缺失回退行(ref_was_estimated=1) |
|---|---|---|---|
| **ranking 2026-01-01** | 4,972 | 212 | 2,449 (1.60%) |
| **ranking 2026-07-01** | 6,065 | 804 | 2,833 (1.47%) |

**关键观察**:
- **2026-01 接 2025-12 历史**: 只有 212 个 NaN 单元格(0.05/行),NaN 率极低,符合"训练数据覆盖 2025-12"的预期
- **2026-07 缺 2026-02~06 历史**: 804 个 NaN 单元格(0.13/行),NaN 率是 1月的 3.8 倍,符合"凌晨窗口无历史"的预期
- **严禁用 2025-07 数据填充 2026-07 凌晨**(用户红线)— 我们没有违反,XGBoost 原生 NaN 处理

#### (d) P0 跨进程一致性覆盖 ranking 路径 + 提交模板一致性

- submitting.parquet: 344,841 行 × 2 列(MVT_ID_mvt, TAXITIME_SEC_mvt)
- ranking MVT_ID 集合 = DEP + ARR(689,534), submitting MVT_ID 集合 = 仅 DEP 子集(344,841)
- **公共 344,841 = submitting 全部** ✅ match=True
- 仅 ranking 独有的 344,693 个全部是 ARR 行的 MVT_ID(预期内,不进提交)

---

## 4. G3 final 探针(stage=final, 12 月, basic30 + A + B)

### 4.1 训练配置

| 项目 | 值 |
|---|---|
| 特征 | basic30 (30 列) + A `dep_mvt_cnt_airport_{5,10,15}m` (3 列) + B `dep_queue_n_airport` (1 列) + `ref_was_estimated` (1 列) = **35 列** |
| 训练周期 | 2025-01 ~ 2025-12 全 12 个月(无 holdout 剔除) |
| 训练样本 | DEP+ARR 全量,3,478,711 行(过滤 TAXITIME 异常后) |
| num_boost_round | **1224**(来自 v10 K-fold median × 1.3,G3 hit_ceiling=False) |
| early_stopping_rounds | 100(对 stage=final 无 val,实际只能 fit 不早停) |
| 种子 | seed=0 |
| 模型名 | `xgb_v10_G3queue_seed0_final.json` |

### 4.2 提交 parquet 一致性

| 校验 | 结果 |
|---|---|
| 提交 shape | (344,841, 2) ✅ |
| 模板 shape | (344,841, 2) |
| 列一致 | True ✅ |
| MVT_ID 集合 | 完全一致 ✅ |
| NaN 数 | 0 ✅ |
| 范围(裁剪后) | [30, 8407]s = [30s, 2.3h] ✅ |
| mean | 963.88s (16.06 min) |
| median | 940.10s (15.67 min) |
| p90 | 1287.80s |
| p99 | 1695.10s |

注: 提交生成时原始有 22 行 (0.0064%) 负值,已 clip 到 [30, 10800]。

### 4.3 重要性 top 15(用真实特征名)

| 排名 | 特征 | gain |
|---|---|---|
| 1 | eobt_to_sched_sec | 119.2M |
| 2 | WK_TBL_CAT_flt | 58.3M |
| 3 | iobt_to_sched_sec | 48.8M |
| 4 | ADEP_mvt | 34.5M |
| 5 | RUNWAY_mvt | 27.3M |
| 6 | ADES_mvt | 20.2M |
| **7** | **dep_queue_n_airport** | **19.6M** ← B 特征进 top 15 |
| 8 | STAND_mvt | 17.2M |
| 9 | is_domestic_estimate | 16.9M |
| 10 | AIRCRAFT_TYPE_mvt | 12.8M |
| 11 | MARKET_SEGMENT_flt | 12.5M |
| 12 | airport_hour | 12.2M |
| **13** | **dep_mvt_cnt_airport_15m** | **10.4M** ← A 特征进 top 15 |
| 14 | is_evening_peak | 8.3M |
| 15 | AIRCRAFT_OPERATOR_flt | 7.7M |

**结论**: A 与 B 特征在 final 模型中都有非平凡 gain(top 15),B(19.6M)略强于 A(10.4M)。这与 v9 G3 AB 的结果一致。

---

## 5. v9 vs v10 RMSE 对比(全部 val, 同 holdout 2025-01+07)

| 组 | v9 (1200 轮) | v10 (median*1.3, ES) | Δ |
|---|---|---|---|
| G0 | 357.14 | 357.22 | +0.08 |
| G2 | 354.99 | 355.57 | +0.58 |
| **G3** | **354.67** | **354.57** | **-0.10 ✅** |
| Gd | 352.33 | 352.36 | +0.03 |

**结论**: v9 的 1200 轮训练得到的 G3 模型本身已经接近最优(差异 ±0.6s 在噪声范围内)。**v10 (1224 轮 + ES) G3 微改善 0.10s**,被采纳为 final 模型的训练轮数。

---

## 6. 阶段 1.5 决策汇总

| 决策 | 结论 |
|---|---|
| G3 是否采纳 | ✅ **采纳**,B 队列特征进主模型 |
| G3 final 训练特征 | basic30 + A(`dep_mvt_cnt_airport_{5,10,15}m`) + B(`dep_queue_n_airport`) + `ref_was_estimated` = 35 列 |
| G3 final 训练周期 | 2025 全年(12 个月,不再剔除 holdout 月) |
| G3 final 训练轮数 | **1224 轮**(来自 v10 5-fold median × 1.3,K-fold 没触顶 4000) |
| G3 final 模型 | `models/xgb_v10_G3queue_seed0_final.json` (27.6 MB) |
| G3 final 提交 | `submissions/jubilant-puzzle_v10_probe.parquet` (4.2 MB) |
| G3 final 上传 | **仅生成,不自动上传**(由用户决定是否上传) |
| D 跑道级最终去向 | **暂不进 final**(按用户裁决),数据已保存备查 |
| E ARR 最终去向 | **否决**(按用户裁决) |

---

## 7. 后续方向预告(本轮不做)

阶段 2 转向:
- (stand, runway) unimpeded P20 基准 + target encoding
- 核查 STAND / RUNWAY / AIRCRAFT_OPERATOR 当前是否被整数 label encoding
- target 清洗(负值/36h 异常、上下 0.1% winsorize 对照)
- CatBoost 入 ensemble

队列特征到此为止,不再加窗打磨。
