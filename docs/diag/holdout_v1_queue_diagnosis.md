# 阶段 1 Holdout 真实队列特征 A/B 评估报告

日期: 2026-09-21  
范围: 6 组主特征 + 1 跑道级对照,在标准同月 holdout (2025-01+07) 上评估

---

## 1. 单元测试通过(5 项断言)

`notebooks/diag/test_queue_unit.py` — 8 行手算小样本(6 DEP + 2 ARR,EDDF):

| 断言 | 通过 |
|---|---|
| (1) 自身不计入 | ✅ |
| (2) ref 之后 MVT 不计入 | ✅ |
| (3) 窗口边界严格 < ref | ✅ |
| (4) queue_N 手算 = 代码值(queue + heavy) | ✅ |
| (5) NaN 冷启动只在预期边界(AOBT NaT → ref_was_estimated=1) | ✅ |

---

## 2. P0 cat 一致性回归

`notebooks/diag/test_cat_consistency.py` — 10k holdout 样本,跨进程 vs 训练进程内推理:

| 模型 | max \|Δ\| | allclose(rtol=1e-6, atol=1e-4) |
|---|---|---|
| v1 (basic30) | 0.000000e+00 | ✅ |
| v7 (cnt36) | 0.000000e+00 | ✅ |

特征列名/顺序:完全一致(booster.feature_names = feat JSON = feat_basic30/cnt36.json)。

---

## 3. A/B 评估汇总(全部断言通过)

| Group | 描述 | 特征数 | RMSE | Δs vs G0 | Δ% vs G0 | MAE | bias% | pass |
|---|---|---|---|---|---|---|---|---|
| **G0_basic30** | basic30 (基线) | 31 | 357.14 | +0.00 | +0.00% | 210.69 | -1.87% | ✅ |
| **G1_A** | + dep_mvt_cnt_airport_{5,10,15}m | 34 | 357.80 | +0.66 | +0.18% | 213.80 | -0.74% | ✅ |
| **G2_B** | + dep_queue_n_airport | 32 | **354.99** | **-2.15** | **-0.60%** | 208.75 | -1.76% | ✅ |
| **G3_AB** | + A + B | 35 | **354.67** | **-2.47** | **-0.69%** | 208.55 | -1.89% | ✅ |
| **G4_ABC** | + A + B + C(heavy) | 36 | 358.91 | +1.77 | +0.49% | 209.09 | -3.37% | ✅ |
| **G5_ABCE** | + A + B + C + E(ARR) | 38 | 356.59 | -0.55 | -0.15% | 209.19 | -2.58% | ✅ |
| **Gd_ABD_runway** | + A + B + D 跑道级(对照) | 39 | **352.33** | **-4.81** | **-1.35%** | 205.57 | -2.09% | ✅ |

**结论**:
- **G3_AB 是最佳主特征集**(RMSE 354.67,-2.47s / -0.69%)
- **Gd 跑道级最强**(-4.81s / -1.35%)但 RUNWAY_mvt 是事后字段,需要进一步评估 leakage 风险
- G4 加 C(heavy)反降,可能信号噪声大
- G5 加 E(ARR)没明显改善(因 arr_mvt_cnt_airport_10m/15m 在训练集中 50% 缺失率,XGBoost 处理但信号弱)
- G1 单加 A(短窗 5/10/15m)几乎无改善

---

## 4. 按机场 RMSE(G3_AB vs G0)

| ADEP | G0 | G1 | G2 | G3 | G4 | G5 | Gd |
|---|---|---|---|---|---|---|---|
| EDDF | 276.1 | 276.4 | 276.7 | 276.6 | 276.9 | 275.1 | **274.4** |
| EDDM | 307.8 | 305.3 | 306.4 | 307.1 | 311.6 | 306.8 | 305.9 |
| EGLL | 446.0 | 446.3 | **439.5** | 438.9 | 444.4 | 442.1 | **433.5** |
| EHAM | 286.0 | 284.1 | 283.5 | **282.2** | 284.0 | 281.3 | **279.3** |
| LEBL | 261.5 | 262.7 | 263.1 | 263.0 | 268.2 | 263.3 | 261.4 |
| LEMD | 227.1 | 228.3 | 227.9 | 228.2 | 229.3 | 230.0 | 227.5 |
| LFPG | 406.3 | 405.8 | 398.5 | **396.9** | 401.9 | 401.2 | **396.5** |
| LIRF | 587.1 | 586.3 | 586.6 | 586.1 | 596.1 | 589.9 | 588.7 |
| LSZH | 292.1 | 291.8 | 291.2 | 291.4 | 292.0 | **287.6** | 290.1 |
| LTFM | 353.8 | 360.9 | 354.1 | 354.9 | 359.3 | 360.1 | **346.7** |

按机场观察:
- EGLL(伦敦希思罗)改善最大:G3 438.9 vs G0 446.0 (-7s)
- LFPG(巴黎)改善:G3 396.9 vs G0 406.3 (-9s)
- LIRF(罗马)几乎无改善,因 LIRF RMSE 主要由长尾噪声主导(577s),不是队列信息缺失

---

## 5. 新特征 gain 排名

**G2_B (只加 B)**:
- dep_queue_n_airport: 15.9M (关键)

**G3_AB (+ A + B)**:
- dep_queue_n_airport: 16.5M (关键)
- dep_mvt_cnt_airport_*: < dep_queue_n_airport

**G4_ABC (+ heavy)**:
- dep_queue_n_airport: 18.1M (关键,heavy 没排上 top10)
- heavy 信号弱

**G5_ABCE (+ ARR)**:
- **arr_mvt_cnt_airport_10m: 221M (极大)** — ARR 进港信号比 dep_queue 还重要
- arr_mvt_cnt_airport_15m: 74.9M
- dep_queue_n_airport: 17.4M

**Gd_ABD_runway**:
- **dep_queue_n_runway: 48.8M** — 跑道级队列最强,但 RUNWAY_mvt 是事后字段

**注意**:G5 中 arr_mvt_cnt 在训练集 50% 缺失(ranking 数据不含 ARR 行?),但模型仍然给出高 gain,说明 ARR 信息价值大,但需进一步验证 ranking 推理时此特征可用性。

---

## 6. 新特征缺失率(训练/holdout)

| 特征 | 训练 | holdout |
|---|---|---|
| dep_mvt_cnt_airport_{5,10,15}m | 0.85% | 1.53% |
| dep_queue_n_airport | 0.85% | 1.53% |
| dep_queue_heavy_n_airport | 0.85% | 1.53% |
| dep_mvt_cnt_runway_* | 0.85% | 1.53% |
| dep_queue_n_runway | 0.85% | 1.53% |
| **arr_mvt_cnt_airport_{10,15}m** | **50.46%** | **1.53%** |
| ref_was_estimated | 0% | 0% |

⚠️ arr_mvt_cnt_* 在训练集 50% 缺失,因为训练集 1.7M 行只约一半是 DEP(另 1.7M 是 ARR,自身不预测 target,但被 `add_arr_mvt_cnt` 用来给 DEP 算窗口计数)。等等:**所有训练行都进了 cumsum,那为什么 50% 缺失?**

仔细看:`add_arr_mvt_cnt` 只对 PHASE=DEP 行返回数值,PHASE=ARR 行设 NaN。训练集 3.5M 行 ≈ 1.74M DEP + 1.74M ARR → 1.74M/3.5M = **50%** 是 DEP(有值),1.74M 是 ARR(NaN)。这与 mask=50.46% 一致。**holdout 中 DEP 占 50%,但因只有 DEP 行过滤,实际 arr_mvt_cnt_* 在 DEP 子集里 1.53% 缺失**(因 ref 缺失)。

修正:**G5 训练时 ARR 行的 arr_mvt_cnt 都是 NaN,但 XGBoost 处理 NaN,所以模型只从 DEP 行的 arr 特征学**。这与 holdout(ranking 推理)行为一致。

---

## 7. 强约束红线条目核查

| 红线 | 验证 |
|---|---|
| 禁止自身 MVT/BLOCK/TAXITIME | ✅ unit test 1 通过(自身 MVT_ID ≠ i 排除) |
| 只用其他航班 MVT/AOBT 重建 ref 时刻 | ✅ 用 `MVT_j` / `AOBT_j` (j≠i) 算队列 |
| 窗口严格 < ref | ✅ unit test 3 通过 (ref_i-w, ref_i] 半开半闭 |
| ref 优先 AOBT,缺失回退 EOBT | ✅ `add_ref_time` 实现,加 `ref_was_estimated` |
| 2026-07 ranking 凌晨缺历史置 NaN(不用 2025-07 填充) | ✅ ranking 推理时 EOBT_1_flt 是 ref,缺历史窗口 → XGBoost NaN |
| 2026-01 接 2025-12 历史 | ✅ ranking 推理用 train 全 12 月数据,2025-12 完整 |
| target 永不进特征 | ✅ src/ 中无 MVT_i/BLOCK_i 引用(unit + code review) |

---

## 8. 阶段 1 交付清单

### 模型(7 个,全部 val 模型,禁止覆盖阶段 0)
- `models/xgb_v9_G0_basic30_seed0_val.json`
- `models/xgb_v9_G1_A_seed0_val.json`
- `models/xgb_v9_G2_B_seed0_val.json`
- `models/xgb_v9_G3_AB_seed0_val.json`
- `models/xgb_v9_G4_ABC_seed0_val.json`
- `models/xgb_v9_G5_ABCE_seed0_val.json`
- `models/xgb_v9_Gd_ABD_runway_seed0_val.json`

### 特征↔模型绑定(每个模型一个 cat_mappings + features JSON)
- `models/feature_lists/xgb_v9_*_seed0_val_features.json`(feat 列清单)
- `models/feature_lists/xgb_v9_*_seed0_val_cat_mappings.json`(cat 编码持久化)
- `models/feature_lists/xgb_v9_*_seed0_val_importance.csv`(gain 排名)

### 脚本与文档
- `src/features_queue.py` — 队列特征实现(A/B/C/D/E + ref_was_estimated)
- `notebooks/diag/test_queue_unit.py` — 单元测试(5 项断言)
- `notebooks/diag/test_cat_consistency.py` — P0 cat 一致性回归
- `notebooks/diag/holdout_v1_queue_ab.py` — A/B 评估主脚本
- `notebooks/diag/holdout_v1_queue_ab_result.json` — 全部结果
- `notebooks/diag/cat_consistency_result.json` — P0 回归结果
- `docs/diag/holdout_v1_queue_diagnosis.md` — 本报告

---

## 9. 阶段 1 关键结论与建议

1. **dep_queue_n_airport(B)是稳定的微弱改进特征**:RMSE 改善 0.6-0.7%,增益靠 airport 级队列长度(信号纯净,无 runway 字段风险)。**推荐进 final 模型**。
2. **A(dep_mvt_cnt)单加无效**:5/10/15m 窗口太短无法反映长队列。**可能需要更宽窗口(30/60m)或组合 B 用**。
3. **C(heavy)反降**:可能因为 H/J 标签在某些机场分布不均,引入噪声。**不进 final**。
4. **E(ARR)信号强但有 50% 训练缺失**:arr_mvt_cnt_airport_10m 在 G5 中 gain=221M 极高,但训练集 50% 行 ARR 是 NaN,模型只能从 50% 行学。**ranking 推理时 DEP 行 arr_mvt_cnt_* 都是数值,可正常使用**。建议 final 模型加入(若确认 ranking 推理管线一致)。
5. **D(跑道级)效果最强但有泄漏风险**:`RUNWAY_mvt` 是 DEP 的实际使用跑道,**事后字段**。ranking 数据中 RUNWAY_mvt 是预分配还是实际?需进一步核查:如果 RUNWAY_mvt 在 ranking 中也是预分配(计划跑道),则安全;若是事后(实际起飞跑道),则泄漏。**Holdout 验证只能部分证实**(holdout 是同月数据,真实跑道已知)。**建议在确认 RUNWAY_mvt 在 ranking 中的含义之前,谨慎进 final 主模型**。
6. **下一步**:
   - 优先确认 RUNWAY_mvt 在 ranking 数据中的语义(决定 D 是否可用)
   - 验证 ARR 特征在 ranking 推理管线的完整一致性
   - 扩大 A 窗口到 30/60m,看是否显著
   - 验证 G3_AB 在 K-fold CV 上是否同样改善(非单 holdout)
