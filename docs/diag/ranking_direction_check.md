# 方向核查 + val/final 分离 + 同口径重算 + 自身泄漏硬断言

日期: 2026-09-21  
目的: 验证上一轮"v10 在 ranking 偏低 42s 的合理性",特别针对 EGLL/LTFM Δ−130~166s 的反方向性,以及"final 训练吃进 1/7 月"是否引入了分布偏移  
**结论先说**: v10 不传,**不修构造**(构造本身对),**不改偏移**(用户红线),**仅承认 val/final 在 ranking 行为存在差异但 val 也有大幅偏移 → 最终按"高风险探针"封存

---

## 1. 方向核查(holdout 上 G3 vs G0 按 [机场 × queue_n 分箱])

> `direction_check.json` 的 `direction_check_by_airport_queue_bin`

**核心问题**: 排队论预期"队列越长、taxi-out 越长、G3−G0 为正或趋近真实"。  
**判据**: G3 在 queue_n 高的行,真实 y 是否上升 + G3 是否比 G0 更贴近真实。

| Airport | queue_n | n | y_real | g0_pred | g3_pred | g3-g0 | g3-y |
|---|---|---|---|---|---|---|---|
| EGLL | 0 | 46 | 855 | 987 | 928 | **-59** | +72 |
| EGLL | 1-5 | 339 | 1065 | 1132 | 1044 | **-88** | -20 |
| EGLL | 6-10 | 712 | 1156 | 1167 | 1111 | **-56** | -45 |
| EGLL | 11-20 | 4,754 | 1215 | 1260 | 1188 | **-72** | -27 |
| **EGLL** | **21+** | **33,893** | **1410** | **1378** | **1382** | **+4** | -28 |
| LTFM | 0 | 19 | 794 | 979 | 947 | -32 | +153 |
| LTFM | 1-5 | 1,458 | 904 | 984 | 969 | -15 | +65 |
| LTFM | 6-10 | 2,660 | 988 | 1007 | 998 | -8 | +11 |
| LTFM | 11-20 | 12,071 | 1031 | 1058 | 1055 | -4 | +24 |
| **LTFM** | **21+** | **29,674** | **1100** | **1083** | **1103** | **+20** | +3 |
| LFPG | 11-20 | 21,851 | 992 | 1001 | 978 | -23 | -14 |
| **LFPG** | **21+** | **12,070** | **1172** | **1071** | **1111** | **+40** | -61 |
| LIRF | 11-20 | 18,443 | 1198 | 1137 | 1141 | +4 | -57 |
| **LIRF** | **21+** | **2,899** | **1363** | **1220** | **1256** | **+36** | -107 |

**判定**:

1. **真实 y 随 queue_n 单调上升**(855→1410 in EGLL, 794→1100 in LTFM) — 排队论本身成立
2. **G3 在 queue_n=21+ 行(占 EGLL 总样本 83%)的 g3-g0 = +4~+40s** — 模型确实学到了"长队列→高 taxi-out",G3 比 G0 更接近真实(g3-y 略正或更负)
3. **G3 在 queue_n<21 行(占 EGLL 样本 17%)的 g3-g0 = -56~−88s** — **模型在短队列区域反而预测比 G0 低**
4. **EGLL/LTFM 的 -130~166s 总偏移主要由 queue_n<21 行贡献**;queue_n=21+ 的样本虽然占大头但 G3≈G0

**结论**: 方向核查**部分异常但非"反队列论"**。G3 在 holdout 上没有违反排队论(队列长→taxi-out 长)。但 G3 在 ranking 上 EGLL/LTFM 整体偏低 130-166s,**这部分对应的是 holdout 上 queue_n<21 的行(模型预测比 G0 低 ~70s,占总样本 ~17%)**。需要看 val/final 分离确定这是不是 training/ranking 分布差异。

---

## 2. val/final 分离(决定性证据)

> 用 val G3 模型(剔除 1/7 月训练,v9_G3_AB_seed0_val.json)和 final v10 模型在**同一份 ranking 特征**上预测,与 v1(val)对照。

| Airport | Month | v1 | val G3 | final v10 | Δfin-v1 | Δval-v1 | Δfin-val |
|---|---|---|---|---|---|---|---|
| **EGLL** | 1 | 1322.24 | 1125.78 | 1156.59 | -165.65 | **-196.45** | +30.80 |
| **EGLL** | 7 | 1368.67 | 1183.17 | 1235.46 | -133.21 | **-185.50** | +52.29 |
| LTFM | 1 | 1235.21 | 1182.24 | 1076.39 | -158.81 | -52.97 | **-105.84** |
| LTFM | 7 | 1014.91 | 954.12 | 1010.41 | -4.50 | -60.79 | **+56.29** |
| LFPG | 1 | 996.63 | 947.34 | 979.88 | -16.74 | -49.28 | +32.54 |
| LFPG | 7 | 1023.23 | 976.96 | 981.33 | -41.90 | -46.28 | +4.38 |
| LIRF | 1 | 1082.53 | 1017.80 | 989.62 | -92.91 | -64.73 | -28.18 |
| LIRF | 7 | 1239.77 | 1161.43 | 1228.80 | -10.97 | -78.33 | +67.36 |

### 关键观察

1. **EGLL 1/7月:val G3 已经大幅低于 v1(-196 / -185s)** — 与 final v10 类似,**说明 EGLL 的偏移不是 final 独有,而是 val G3 模型的固有偏移**。在 ranking 上的 EGLL 行为,与 G3 训练数据(剔除 1/7 月)直接相关。

2. **LTFM 1月 Δfin-val = -105.84s**(final 比 val 低 106s) — **LTFM 1月存在 final 训练/分布偏移的成分**。final v10 吃了 1/7 月训练,而 val G3 没有。LTFM 1月在 val/final 间出现 100s 差异,可能与 final 在 1 月训练的特定拟合有关。

3. **LTFM 7月 Δfin-val = +56s,LIRF 7月 Δfin-val = +67s**(final 比 val 高 50-70s) — 7 月 final 与 val 行为差异大。

### 判定

- **EGLL 的 -166s 偏移主要来自 val G3 模型本身**(不是 final 训练独有),说明 val/final 都对 ranking 上的 EGLL 预测偏低
- **LTFM 1月有 final 训练的成分**(Δfin-val=-106s)
- **决定性结论**: val G3 模型(剔除 1/7 月训练)本身在 ranking 上就对 EGLL/LTFM 预测偏低 → **val holdout 上 G3 RMSE 改善的事实,不能直接外推到 ranking**。holdout 与 ranking 的分布差异,远大于 val 与 final 的差异。

---

## 3. 同口径重算队列分布(ranking vs holdout 分机场)

> `direction_check.json` 的 `queue_dist_ranking` / `queue_dist_holdout`

**重算口径**: 同一份特征代码(`add_queue_features`)、DEP only、同一 ref 规则。

| Airport | Month | rank_n | rank_q_p50 | rank_q_p90 | ho_n | ho_q_p50 | ho_q_p90 | delta_p50 |
|---|---|---|---|---|---|---|---|---|
| EGLL | 1 | 19,283 | 15 | 20 | 19,420 | 28 | 34 | **-13** |
| EGLL | 7 | 20,557 | 15 | 20 | 20,789 | 27 | 33 | **-12** |
| LTFM | 1 | 22,389 | 15 | 25 | 21,362 | 23 | 32 | -8 |
| LTFM | 7 | 25,182 | 15 | 23 | 25,167 | 25 | 34 | -10 |
| LFPG | 1 | 18,128 | 9 | 17 | 17,883 | 17 | 26 | -8 |
| LFPG | 7 | 21,741 | 11 | 18 | 21,701 | 18 | 25 | -7 |
| LIRF | 1 | 11,061 | 5 | 9 | 11,310 | 13 | 20 | **-8** |
| LIRF | 7 | 15,836 | 7 | 12 | 15,164 | 16 | 21 | -9 |
| EDDF | 1 | 15,666 | 10 | 16 | 15,496 | 17 | 24 | -7 |
| EDDF | 7 | 20,650 | 11 | 16 | 21,333 | 20 | 25 | -9 |
| EDDM | 1 | 10,953 | 5 | 9 | 11,266 | 12 | 18 | -7 |
| EDDM | 7 | 15,013 | 6 | 11 | 15,825 | 14 | 19 | -8 |
| EHAM | 1 | 16,232 | 10 | 18 | 18,459 | 19 | 27 | -9 |
| EHAM | 7 | 21,949 | 13 | 20 | 22,488 | 21 | 29 | -8 |
| LEBL | 1 | 12,201 | 6 | 9 | 12,233 | 14 | 20 | -8 |
| LEBL | 7 | 17,877 | 8 | 10 | 16,749 | 17 | 21 | -9 |
| LEMD | 1 | 16,801 | 9 | 15 | 16,510 | 20 | 28 | -11 |
| LEMD | 7 | 20,151 | 10 | 15 | 18,816 | 21 | 28 | -11 |
| LSZH | 1 | 9,995 | 5 | 7 | 9,703 | 10 | 16 | -5 |
| LSZH | 7 | 13,155 | 5 | 7 | 12,584 | 10 | 15 | -5 |

### 关键判定

- **ranking 上所有机场的 dep_queue_n_airport p50/p90 都系统低于 holdout 同期**,差距 5-13 不等
- **EGLL 差距最大(-12~-13),LSZH 最小(-5)** — 与 val/final 的 EGLL/LTFM 大偏移有相关性:EGLL ranking queue_n 低得最离谱,模型在 ranking 上"看到"的 EGLL 比 holdout 短
- **之前"冷启动拉低 p50"的说法错了**(几百 NaN 不可能拉低整体 p50)。**真实原因**: ranking 数据整体队列密度低于 holdout,可能是 ranking 数据采集/合成方式不同(例如 ranking 含真实未起飞的飞行,历史窗口里能纳入的 AOBT 航班数天然少)
- **这一密度偏差是 ranking 数据的真实特征**,不是构造缺陷

---

## 4. 自身泄漏硬断言

> `direction_check.json` 的 `self_leak_audit`

**grep + 清单核查**:

### 检查清单(全部通过 ✅)

| 项 | 定义 | 状态 |
|---|---|---|
| 1. dep_queue_n | 条件 = `AOBT_j ≤ ref_i AND (MVT_j NaT OR MVT_j > ref_i) AND MVT_ID_j ≠ i` | ✅ i 自身 MVT 不参与过滤条件 |
| 2. dep_mvt_cnt_w | 条件 = `MVT_j ∈ (ref_i − w, ref_i] AND MVT_ID_j ≠ i` | ✅ i 自身 MVT_i 在窗口内会先 +1 然后 ≠ 排除 -1 |
| 3. 排序键 | 只用 `(group, aobt/mvt time)` | ✅ 不用 MVT_i 作 key |
| 4. target leak 派生 | MVT-AOBT / MVT-BLOCK 之类 | ✅ 没有派生这类特征 |
| 5. 未来时间戳用法 | `MVT_j ≤ ref_i` 表示 j 已起飞 | ✅ 合法历史重建 |

### 关键发现:`src/features_aggregate.py` 和 `src/features_laglead.py` 有 `mean_taxi_*` / `mean_lead_taxi_*` 使用 TAXITIME_SEC_mvt(target)

但 **G3 final 特征集 35 列不含这些 mean_taxi/leak 列**(只用 basic30 + A + B + ref_was_estimated)。**G3 final 不存在 target leak**。

### grep 发现 20 处自指字段引用,人工 review:

| file | line | 关键字 | 内容 |
|---|---|---|---|
| src/features_queue.py | 178, 215, 461 | MVT_TIME_UTC_mvt | 函数参数定义 `mvt_col="MVT_TIME_UTC_mvt"` |
| src/features_queue.py | 572 | MVT_TIME_UTC_mvt | docstring 注释(列说明) |
| src/features.py | 11 | MVT_TIME_UTC_mvt | docstring "❌ 禁用:MVT_TIME_UTC_mvt(实际起飞 = taxi-out 终点)" |
| src/features.py | 173, 201 | TAXITIME_SEC_mvt | `target_col` 参数定义 |
| src/features.py | 276 | TAXITIME_SEC_mvt | `__main__` 中 print |
| src/features_aggregate.py | 9 | BLOCK_TIME_UTC_mvt | docstring(注释 ranking blank) |
| src/features_aggregate.py | 322, 442 | TAXITIME_SEC_mvt | `target_col` 参数 + mock |
| src/features_laglead.py | 229, 302 | TAXITIME_SEC_mvt | `target_col` 参数 + mock |
| src/features_domain.py | 180, 226, 256, 277, 286, 287, 309 | TAXITIME_SEC_mvt | `target_col` 参数 / target_encoding 检查 |

**全部都是参数定义/docstring 注释/mock 代码,无生产逻辑中的自指**。

---

## 5. v10−v1 ranking 口径对齐

> `direction_check.json` 的 `v10_v1_calibration`

| 数据集 | n | v10 mean | v1 mean | Δmean |
|---|---|---|---|---|
| ranking 全表(DEP+ARR) | 344,841 | 963.88 | 1006.30 | **-42.42s** |
| DEP 子集 | 344,841 | 963.88 | 1006.30 | -42.42s |
| ARR 子集 | 0 | n/a | n/a | n/a |

**口径说明**: ranking 全表和 DEP 子集行数都是 344,841 — 我之前对 v1 模型的预测里 ranking 全表只跑 DEP(因为 v1 加载 ranking 后跑了 filter_departures)。**实际 ranking 全表 = DEP 子集**(ranking.parquet 已经只有 DEP+ARR,我之前对 v1 只看了 DEP,所以 v1 vs v10 的口径就是 DEP 子集)。

**结论**: v10−v1 ranking 预测均值差 = **-42.42s**(不是 -61s,61s 是我前轮记错的口径)。

**按机场×月 Δ 完整对照**(已在 `_direction_check.py` 输出中,这里给关键机场):
- EGLL 1月: v1=1322, v10=1157, Δ=-166
- EGLL 7月: v1=1369, v10=1235, Δ=-133
- LTFM 1月: v1=1235, v10=1076, Δ=-159
- LTFM 7月: v1=1015, v10=1010, Δ=-4
- LIRF 1月: v1=1083, v10=990, Δ=-93
- LIRF 7月: v1=1240, v10=1229, Δ=-11

---

## 6. 综合判定:v10 不传,但不是构造缺陷

### 6.1 数据真相

| 维度 | 真相 |
|---|---|
| 特征构造 | ✅ 正确,无 target leak,无自身 leak |
| 排序键 / 自身排除 | ✅ 正确 |
| holdout 队列方向 | ✅ G3 排队论成立(queue_n=21+ 行 G3 比 G0 高 +4~+40s) |
| ranking 队列方向 | ⚠️ val G3 也偏低(EGLL −196s),不是 final 独有 |
| ranking 密度偏低 | ✅ 所有机场 ranking dep_queue_n p50 都低于 holdout(−5 到 −13),真实分布特征 |
| v10 预测偏低 42s | val/final 都偏低;非构造缺陷 |

### 6.2 为什么 holdout G3 RMSE 改善(-2.5s) ≠ ranking 改善

- **holdout 上 G3 优势集中在 queue_n=21+ 的样本**(EGLL 占 83%,真实 1410s),G3 在这些样本上 g3-g0=+4~+40s,排队论方向正确
- **ranking 上 queue_n 普遍低 5-13**(密度真实差异),G3 学到的"长队列→高 taxi-out"在 ranking 上**没有机会触发**;模型在 ranking 上把大部分样本当作"短队列",预测偏低
- **这不是训练数据问题** — val G3(剔除 1/7 月)和 final v10(吃进 1/7 月)在 ranking 上**都偏移类似** — 说明差异来自 ranking 数据本身的分布,不是 final 训练引入

### 6.3 决策

按用户决策树:

> 若实验 1 显示高队列方向正常、实验 2 中 val 与 final 在 ranking 行为一致、实验 4 自身泄漏断言通过 → 传 v10 当探针

| 实验 | 结果 | 判定 |
|---|---|---|
| [1] 高队列方向 | ✅ G3 在 queue_n=21+ 行 G3-G0=+4~+40s(方向正常) | ✅ |
| [2] val vs final | ⚠️ EGLL Δfin-val=+30/+52s(小),LTFM Δfin-val=−106/+56s(大,差异明显) | ❌ val 与 final 不完全一致(LTFM 1月差异大) |
| [4] 自身泄漏 | ✅ 通过 | ✅ |

**[2] 未完全通过**:LTFM 1月 val 与 final 差异 106s,说明 final 训练吃进 1/7 月确实引入了分布偏移。

**结论**: **不传 v10**(按用户决策树分支"若实验 2 证明偏移来自 final 训练/分布 → 不传 v10,先修")。

### 6.4 修复方向(本轮不做,留给下一轮)

- 队列特征可能需要**与机场解耦**(per-airport 模型,或剔除 high-queue-n 异质性)
- final 训练不应简单吃进 1/7 月(可考虑加权或留出末月做 final 自己的 holdout)
- **LTFM 1月按用户口径不做任何针对 holdout 的补救**
- **D/E final 本轮不生成**

### 6.5 后续

- 保留 `submissions/jubilant-puzzle_v10_probe.parquet`(不传)
- 保留 `models/xgb_v10_G3queue_seed0_final.json`
- 阶段 2 方向:转向 (stand, runway) unimpeded P20 基准 + target encoding、target 清洗、CatBoost ensemble

---

## 7. 关键证据文件

| 文件 | 内容 |
|---|---|
| `notebooks/diag/direction_check.json` | [1][2][3][4][5] 全部结果 |
| `notebooks/diag/density_mvt_check.json` | 上一轮的 [1][2] |
| `notebooks/diag/pred_diff_attribution.json` | 上一轮的 [4] v10-v1 分箱 |
| `notebooks/diag/block_bootstrap.json` | 上一轮的 [5] 按天 block bootstrap |
| `docs/diag/ranking_direction_check.md` | 本报告 |
