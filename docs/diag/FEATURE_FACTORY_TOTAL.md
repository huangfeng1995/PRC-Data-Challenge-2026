# 特征总表 (feature-factory)

**铁律**：ranking 上 `BLOCK_TIME_UTC_mvt` 50.6% 缺失、`TAXITIME_SEC_mvt` 50.6% 缺失（目标被屏蔽），
所有特征只能用 ranking 全字段。来源证据：`data/sample/ranking.parquet` (5000 行)。

| ranking 字段 | 非空 | 状态 |
|---|---|---|
| MVT_TIME_UTC_mvt | 5000/5000 | ✅ |
| SCHED_TIME_UTC_mvt | 5000/5000 | ✅ |
| IOBT_flt | 4943/5000 | ✅ |
| EOBT_1_flt | 4943/5000 | ✅ |
| AOBT_3_flt | 4943/5000 | ✅ |
| ARVT_3_flt | 4943/5000 | ✅ |
| LOBT_flt | 4943/5000 | ✅ |
| ADEP_mvt / ADES_mvt / RUNWAY_mvt / STAND_mvt / MVT_ID_mvt | 5000/5000 | ✅ |
| AIRCRAFT_TYPE_mvt / AIRCRAFT_OPERATOR_flt / WK_TBL_CAT_flt | 4943~4996/5000 | ✅ |
| ~~BLOCK_TIME_UTC_mvt~~ | 2531/5000 | ❌ **禁用** |
| ~~TAXITIME_SEC_mvt~~ | 2531/5000 | ❌ **目标被屏蔽** |
| FLIGHT_TYPE_mvt | 0/5000 | ❌ |
| OPERATOR_ID_flt | 0/5000 | ❌ |

---

## A. 队列（21 列，ranking 兼容）

`src/features_factory_batchA.py::add_batchA_queue_features(df)`

| 特征 | 源字段 | 业务定义 | sample nn | 单测结果 | 业务假设 | 预期方向 |
|---|---|---|---|---|---|---|
| `wait_rwy_{10,20,30,60}_ap` | AOBT/MVT/MVT_ID | ref_i − AOBT_j 均值（j 已 AOBT 未 MVT） | 0.9%~4.1% | ✅ 5 行手算全对 | 已推出机在跑道等候时间 | **正** |
| `n_mvt_rwy_{10,30,60}` | MVT/MVT_ID | 过去 w min 同跑道已起飞数 | 99.3% | ✅ 5 行手算全对 | 跑道吞吐历史 | **正（活跃度）** |
| `n_dep_queue_rwy_{15,30,60}` | AOBT/MVT/MVT_ID | 瞬时同跑道已 AOBT 未 MVT | 99.3% | ✅ 5 行手算全对 | 跑道排队密度 | **正** |
| `n_dep_queue_ap_{15,30,60}` | AOBT/MVT/MVT_ID | 瞬时同机场已 AOBT 未 MVT | 99.3% | ✅ | 机场排队密度 | **正** |
| `n_arr_rwy_{15,30,60}` | ARVT/ADES | 过去 w min 同跑道 ARVT 数 | 100% | ✅ DEP 全 0 合理 | ARR 占用跑道 | **正** |
| `rwy_arr_active_180s` | ARVT/ADES | ref_i-180s 同跑道 ARVT | 100% | ✅ | G4 风险单列 | **正** |

口径：ref_i = AOBT_3_flt if notna else EOBT_1_flt。窗口左开右闭 `(ref_i - w, ref_i]`。排除自身 MVT_ID。机场隔离 DEP=ADEP, ARR=ADES。

---

## B. 目标编码 TE（5 列 + 推理映射）

`src/features_factory_batchB.py::add_batchB_te_features(df) → (df_oof, mappings)`
`apply_batchB_te_features(df_test, mappings)`

| 特征 | 源字段 | 业务定义 | 训练 OOF nn | ranking 推理 nn | 单测结果 | 备注 |
|---|---|---|---|---|---|---|
| `te_ap_std_rwy_median` | ADEP/STAND/RUNWAY/TAXITIME | 4 层回退中位数（primary→fallback→global） | 99.3% | **100%** | ✅ OOF 防泄漏验证 | 真实值待全量 |
| `te_ap_std_rwy_lowdemand` | 同上 + 25 百分位 | 非拥堵期基准版 | 99.3% | 100% | ✅ | 同上 |
| `te_ap_rwy_hour` | ADEP/RUNWAY/AOBT_hour | 对照 TE | 99.3% | 100% | ✅ | sample 小时稀疏 |
| `te_ap_std` | ADEP/STAND | 对照 TE | 99.3% | 100% | ✅ | |
| `te_ap_operator` | ADEP/AIRCRAFT_OPERATOR | 对照 TE | 99.3% | 100% | ✅ | |

**铁律 1 关键**：ranking 上 `TAXITIME_SEC_mvt` 缺失 → 必须用训练时 `mappings` 推理填特征。  
**真实值标"待全量"**：sample 700 行太稀疏，OOF 误差大。全量 33 万行才能见真章。

---

## C. outlier 指纹（7 列，ranking 兼容）

`src/features_factory_batchC.py::add_batchC_outlier_features(df)`

| 特征 | 源字段 | 业务定义 | 训练 1s | ranking 1s | 单测结果 | 业务假设 | 预期方向 |
|---|---|---|---|---|---|---|---|
| `outlier_stand_match` | ADEP/STAND + xplane stands | STAND 精确匹配（去机型后缀 + LR 后缀） | 41.6% | 55.7% | ✅ 5 行手算 | 失配 → 数据脏 → taxi 不可信 | **正**（risk_score 子项） |
| `outlier_rwy_match` | ADEP/RUNWAY + xplane runways | RUNWAY 匹配（去 LR + 0 前缀） | 37.6% | 49.3% | ✅ | 同上 | **正** |
| `outlier_no_aobt` | AOBT_3_flt | AOBT 缺失 | 0.7% | 1.1% | ✅ | 无法算 ref_i | **正** |
| `outlier_aobt_plan_bad` | AOBT/EOBT | \|AOBT-EOBT\| > 20min | 6.7% | 11.2% | ✅ | 计划 vs 实际推出偏差大 | **正** |
| `outlier_aobt_sched_clash` | AOBT/SCHED | \|AOBT-SCHED\| < 5min | 18.6% | 13.1% | ✅ | 几乎无 delay（数据可疑）替代 BLOCK clash | **正** |
| `outlier_crossday` | AOBT.date/SCHED.date | 跨日（AOBT 日期 ≠ SCHED 日期）替代 MVT-BLOCK | 0.1% | 4.9% | ✅ | 数据错误 | **正** |
| `outlier_risk_score` | 上 6 列 | 加权和（crossday=2.0, plan_bad=1.5, no_aobt=1.0, rwy=0.8, stand/clash=0.5） | μ=0.71 max=3.30 | μ=1.01 max=4.80 | ✅ 5 行手算全对 | 综合 outlier 风险 | **正** |

注：旧批次 3 的 BLOCK-based clash/crossday/score 已废弃（铁律 1）。

---

## D. 计划/时间（8 列）

`src/features_factory_batchD.py::add_batchD_plan_features(df)`

| 特征 | 源字段 | 业务定义 | sample nn | 单测结果 | 业务假设 | 预期方向 |
|---|---|---|---|---|---|---|
| `plan_aobt_eobt_min` | AOBT/EOBT | (AOBT − EOBT) / 60 | 99.3% μ=5.02 | ✅ | 实际推出 vs 计划偏差 | **正** |
| `plan_lobt_iobt_min` | LOBT/IOBT | (LOBT − IOBT) / 60 | 99.3% μ=0.0 | ✅ | 内部协调 | — |
| `plan_aobt_sched_min` | AOBT/SCHED | (AOBT − SCHED) / 60 | 99.3% μ=-41.26 | ✅ | 实际 vs 计划时段偏差 | **正** |
| `plan_eobt_iobt_min` | EOBT/IOBT | (EOBT − IOBT) / 60 | 99.3% μ=-0.16 | ✅ | 计划对齐 | — |
| `plan_lobt_eobt_min` | LOBT/EOBT | (LOBT − EOBT) / 60 | 99.3% μ=0.16 | ✅ | 滑行后 vs 计划 EOBT | — |
| `plan_iobt_eobt_consist` | IOBT/EOBT | \|IOBT − EOBT\| < 60s → 1 | 1s 占比 65% | ✅ | 内部协调好 → taxi 短 | **负** |
| `ap_hour_coord_peak` | ADEP/AOBT_hour | (ADEP, hour) 航班数 > P80 → 1 | 1s 占比 37% | ✅ | 协调高峰 → 拥堵 | **正** |
| `ap_hour_density_rank` | ADEP/AOBT_hour | 5 等分密度分位 | 1~5 | ✅ | 越忙 → taxi 长 | **正** |

---

## E. 官方框架新机制（4 列）

`src/features_factory_batchE.py::add_batchE_official_features(df)`

| 特征 | 源字段 | 业务定义 | sample nn | 单测结果 | 业务假设 | 预期方向 |
|---|---|---|---|---|---|---|
| `flow_proxy_aobt_sched_min` | AOBT/SCHED | (AOBT − SCHED) / 60 | 99.3% μ=-41 max=465 | ✅ | 真实推出 vs 计划时段系统性后移（机场流量调控代理） | **正** |
| `flow_proxy_late_15p` | AOBT/SCHED/AOBT.date | 同 (ADEP, AOBT.date) 内 \|Δ\| > 15min 占比 | 99.3% μ=0.57 max=1.00 | ✅ | 高占比 → 机场拥堵 | **正** |
| `stand_bearing_to_rwy_deg` | ADEP/STAND/RUNWAY + xplane 坐标 | haversine + bearing 方位角 (0~360°) | **45.3%** μ=124 | ✅ | 距跑道远/角度大 → taxi 长 | **正** |
| `active_rwys_in_window` | ADEP/RUNWAY/MVT + ARVT(可选) | ref_i-15min 同机场活跃跑道数 | 100% μ=0.03 max=2 | ✅ | 跑道少 → 拥堵 | **正** |

---

## F. 可选（未实现 / 标"待全量"）

| 特征 | 原因 |
|---|---|
| 前序重型机数量 (WK_TBL_CAT ∈ {H,J}) | 队列过滤逻辑可加，但 sample 上重型航班极少，统计不可靠 |
| 最近重型机距离 | 同上 |

---

## 已剔除（铁律 1 命中 / 排名兼容性失败）

| 特征 | 原因 |
|---|---|
| `mu_rwy_*` (旧版用 TAXITIME_SEC_mvt 均值) | ranking 上 50% 缺失，已被 `wait_rwy_*_ap`（用 AOBT）替代 |
| `outlier_block_sched_clash` | 用 BLOCK_TIME，ranking 上 50% 缺失 → 替换为 `outlier_aobt_sched_clash` |
| `outlier_crossday_slip` (MVT-BLOCK) | 同上 → 替换为 `outlier_crossday`（AOBT.date vs SCHED.date） |

旧 `src/features_factory_batch1.py` / `batch3.py` 保留作 dev 历史，未删。

---

## 全量 add-one 验证建议（Mac 地基重建后）

按铁律 1，ranking 推理管线需要：
1. **训练**：在 `data/raw` 上跑 `add_batchA_queue_features` + `add_batchC_outlier_features` + `add_batchD_plan_features` + `add_batchE_official_features`（无 target 也能跑）；
2. **TE 拟合**：`add_batchB_te_features(df_train)` → 存 `mappings`；
3. **ranking 推理**：先 `apply_batchB_te_features(rank, mappings)` 补 TE 列，再跑 A/C/D/E；
4. **进入模型管线**：train_ensemble 的 `build_full_features` 里加上 `extra_features=['A','B','C','D','E']` 开关。
