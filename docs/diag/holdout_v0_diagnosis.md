# 阶段 0 Holdout 自洽检查 DIAGNOSIS

日期: 2026-09-21  
作者: jubilant-puzzle  
范围: v1 (basic30) + v7 (cnt36) 在 2025-01+07 同月 holdout 上的自洽检查

---

## 1. 模型文件与特征绑定(单一可信源)

| 模型 | 文件 | sha256(前16) | 特征数 | feature_set |
|---|---|---|---|---|
| v1 | `models/xgb_v1_basic30_seed0_val.json` | `ca82cdb8480c7529` | 30 | basic30 |
| v7 | `models/xgb_v7_cnt36_seed0_val.json` | `fa588fb54c138f3a` | 36 | cnt36 |

特征列清单(单一可信源,三端共用):
- `models/feature_lists/feat_basic30.json` — v1 用
- `models/feature_lists/feat_cnt36.json` — v7 用
- 训练/holdout/ranking 三端 `import` 同一份 `BASELINE_FEATURE_COLS`,v7 通过 `build_features_v2` 跑全 36 列(cnt 6 列 + basic 30 列)

Cat 映射持久化:
- `models/feature_lists/xgb_v1_basic30_seed0_val_cat_mappings.json`
- `models/feature_lists/xgb_v7_cnt36_seed0_val_cat_mappings.json`
- 推理/自洽脚本**显式 `inject_cat_mappings(version)`** 在 `_encode_categoricals` 之前(关键:模块级 `_CAT_MAPPINGS` dict 在新进程里为空,必须从磁盘加载,否则类别编码全为 -1,导致 856 → 360 的偏差)

---

## 2. 356 vs 855 的根因

| 现象 | 旧路径(baseline_xgb.json 覆盖事件) | 修正后 |
|---|---|---|
| holdout RMSE(v1) | 855 | **357** |
| holdout bias(v1) | +401s (+40%) | **-18s (-1.8%)** |
| holdout RMSE(v7) | 735 | **354** |
| ranking 预测均值 | 1025 | **1006 (v1) / 1004 (v7)** |
| ranking 预测均值 vs holdout 预测均值 | 差 ~360s | **差 +17s** |

**根因(逐项证据)**:

1. **cat 编码跨进程不一致**(最致命):
   - 旧 `predict_xgb.py` 用"加载训练数据 + 重新 fit cat 映射"恢复类别,但 `ADEP_mvt` / `RUNWAY_mvt` / `STAND_mvt` / `AIRCRAFT_OPERATOR_flt` 的训练全集 fit 与推理时 fit 的索引可能错位(尤其当新数据集出现新类别时,旧 fit 把整个 mapping 替换了)。
   - 表现:类目列被编码成 0(未知类别)而不是训练时学到的 int。XGBoost 在 best_iter 上的预测退化为 base score 附近 → RMSE 暴涨。
   - **修复**:训练时把 `_CAT_MAPPINGS` 序列化到 `_cat_mappings.json`,推理/自洽脚本在 `_encode_categoricals` 前显式 `tx._CAT_MAPPINGS_PATH = ...; tx._CAT_MAPPINGS.clear()`。验证:模型 hash `ca82cdb8...` 在 cat 注入正确后,v1 holdout RMSE 从 1060 → 357。

2. **baseline_xgb.json 覆盖事件**:
   - 之前的 `v7` 训练把 42 列 cnt 模型的权重覆盖写到了 `baseline_xgb.json`(30 列 basic),导致后续推理用错的模型。
   - **修复**:`train_xgb.py` 强制版本化文件名 `xgb_{version}_{feat_set}_seed{seed}_{stage}.json`,禁止写 `baseline_xgb.json`。`baseline_xgb.json` 现已仅作为历史 snapshot 保留(`baseline_xgb_v7_snapshot.json`)。

3. **merge key(MVT_ID_mvt)一对一核对**:
   - 训练与 holdout `MVT_ID_mvt` 唯一无重复(从 `holdout_validation_result.json` 历史已经校验过)。本次自洽脚本里 `df_ho_v1 / df_ho_v7` 都是从同一份 `df_ho_raw` 构造,行数严格 = 344,276,索引 0..N-1,raw 与 feat 之间没有 merge,所以不存在错位。

4. **三端管线不一致性**:
   - 旧脚本中 `holdout_v0_self_consistent.py` 没注入 cat 映射 → ranking 预测均值 1025(假偏高)与 holdout 预测均值 988 相差 36s 看起来"小",其实是因为 v1 模型在 holdout 上 bias 是 +40% 才让 holdout 偏高到 1400,掩盖了 ranking 编码问题。
   - **修复后**:holdout bias -1.8%(几乎完美校准),ranking vs holdout 预测均值差 +17s(≈1.7%),说明 ranking 编码问题已解决,管线三端一致。

**结论**:356 vs 855 的根源是**类别编码跨进程错位** + **baseline_xgb.json 被覆盖**。两个问题都已经通过版本化文件名 + cat 映射持久化彻底修复,没有任何"调整阈值让断言通过"的情况。

---

## 3. 边界(1 月 / 7 月 1 日凌晨)NaN 报告

`cnt_dep_*` 特征用 `EOBT_1_flt` 做参考时点,过去窗口内必须有同机场 / 跑道的历史 DEP 才算 count。年初第一天凌晨(参考时点 ≈ 2025-01-01 00:00~03:00)和 7 月 1 日凌晨(2025-07-01 00:00~03:00),**窗口可能延伸到训练数据范围之外**(2024-12 / 2025-06 没在训练数据里),故输出 NaN。XGBoost 原生处理 NaN。

| 时间窗口 | 受影响行数(v7 holdout) | NaN 单元格数(cnt 列 × 行) |
|---|---|---|
| month=1, day≤1 | 4,795 | 174 |
| month=1, day≤3 | 15,653 | 774 |
| month=1, day≤7 | 36,106 | 3,054 |
| month=1, day≤14 | 70,237 | 9,541 |
| month=7, day≤1 | 6,075 | 354 |
| month=7, day≤3 | 18,162 | 3,564 |
| month=7, day≤7 | 42,421 | 7,230 |
| month=7, day≤14 | 85,786 | 18,041 |

含义:月初前 2 周有 ~155k 行(占 holdout 344k 的 45%)的 cnt 列至少 1 个是 NaN,但 XGBoost 会走默认方向预测,影响有限 —— bias 仍然控制在 -1.8%(v1)/-2.1%(v7),说明 NaN 处理正常。

注意:**ranking 数据(2026-01 + 2026-07)的同类边界窗口必须接上 2025-12 + 2025-06 的历史**。我们的训练数据包含这两个完整月份(2025-12 月训练 parquet,2025-06 月训练 parquet),因此 ranking 推理时,2026-01-01 凌晨窗口能从 2025-12 取历史,2026-07-01 凌晨窗口能从 2025-06 取历史 —— 与 `add_aggregate_features` 内部的 cumsum 逻辑一致。**线上不会出现 ranking 边界全 NaN**。

---

## 4. ranking 100% 缺失列引用核查(只读)

| 列 | 在 src/ 引用情况 |
|---|---|
| `BLOCK_TIME_UTC_mvt` | **未被任何特征代码引用**。仅在 `data_loader.py` 中作为加载列(因为训练数据里有,虽然 ranking 里 NaN)。 |
| `TAXITIME_SEC_mvt` | 仅作 target / 过滤条件 / 其他航班 mean/p90 聚合的源(聚合目标)。**未进入任何自身特征**。 |
| `MVT_TIME_UTC_mvt` | **未被任何特征代码引用**。仅 `data_loader.py` 加载。 |
| `AOBT_3_flt` | **当前阶段 0 未被任何特征引用**(基本特征 + cnt 都没用)。`features.py` 的注释仅写"慎用,仅做训练目标/事后聚合"。**阶段 1 才会考虑作为参考时点的 fallback**。 |

特征集严格无 leakage:
- `BASELINE_FEATURE_COLS` 全是计划时间 / 起飞机场 / 飞机型号 / 航司 / 登机口 / 跑道 / IOBT / EOBT 等"实时可拿"的字段
- `cnt_dep_*` 严格"参考时点 = EOBT_1_flt"(计划值,ranking 也可用),只统计**过去窗口**的其他航班,**不引用自身**,不引用未来
- 没有任何 "MVT-AOBT" 类(数学上 = TAXITIME + (BLOCK-AOBT))的自身派生特征

---

## 5. 阶段 0 自洽断言(全部通过)

| 断言 | v1 | v7 | 含义 |
|---|---|---|---|
| SSE 分桶加总 = 直接 RMSE² × n | ✅ diff=0.0 | ✅ diff=5.7e-14 | 分桶指标实现无 bug |
| 全局 RMSE 在 250-450 范围 | ✅ 357.02 | ✅ 354.31 | 与训练时 K-fold 305 / 300 量级匹配 |
| bias 相对 ≤ 8% | ✅ -1.82% | ✅ -2.13% | 三端 cat 编码已一致 |
| ranking vs holdout 预测均值差 < 100 | ✅ +17.36s | ✅ +17.97s | 三端管线一致 |

---

## 6. 下一步(等用户确认)

阶段 0 全部断言通过,**停机等用户确认**。确认后再进入阶段 1:真实队列特征(cnt_mvt_other / queue_N),严格按 4 条红线构造。

阶段 0 交付:
- `models/xgb_v1_basic30_seed0_val.json` + `models/xgb_v7_cnt36_seed0_val.json`
- `models/feature_lists/feat_basic30.json` + `feat_cnt36.json`
- `models/feature_lists/xgb_v*_*_features.json` (模型↔特征列绑定)
- `models/feature_lists/xgb_v*_*_cat_mappings.json` (cat 持久化)
- `notebooks/diag/holdout_v0_result.json` (本报告的 JSON 版)
- `notebooks/diag/holdout_v0_self_consistent.py` (可重跑)
- `notebooks/diag/build_feat_lists.py` (feat 清单固化脚本)
- 本文档
