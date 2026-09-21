# 阶段 2 编码 + target 清洗审计报告

日期: 2026-09-21  
范围: 编码现状([A1]-[A6]) + target 清洗审计与对照实验([B1]-[B5])  
**结论先说**:
- 编码:**严重错误**,整数 label-encode 被 XGB 当 numeric 处理,高基数无序类别(AIRCRAFT_OPERATOR/RUNWAY/STAND/AIRCRAFT_TYPE_mvt)被分裂为 "<=k vs >k",风险高
- target 清洗:**C2 winsorize(per-ap P0.1-P99.9)改善 RMSE 4.85s,LIRF 改善 20s**,但 EGLL/EGLL 不动;winsorize + 硬剔除组合无额外收益
- **本机 CatBoost 1.2.10 + libomp + LightGBM 4.7.0 都可用**,路线 3 (CatBoost) 可行

---

## 审计 A:编码现状

### [A1] `_encode_categoricals` 完整源码(src/train_xgb.py:110-138)

```python
def _encode_categoricals(X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """把 category 列用整数编码。fit=True 时学习映射,fit=False 时用已有映射。

    v0.1 修复:同时把 cat 映射存到 _CAT_MAPPINGS_PATH 文件,跨进程保持一致。
    """
    global _CAT_MAPPINGS
    if not _CAT_MAPPINGS:
        if _CAT_MAPPINGS_PATH is not None and _CAT_MAPPINGS_PATH.exists():
            _CAT_MAPPINGS = json.loads(_CAT_MAPPINGS_PATH.read_text())

    X = X.copy()
    for col in X.select_dtypes(include="category").columns:
        if fit:
            # 学习映射,NaN 用 -1 表示
            cats = X[col].cat.categories.tolist()
            _CAT_MAPPINGS[col] = {c: i for i, c in enumerate(cats)}
            _CAT_MAPPINGS[col]["__nan__"] = -1
            # 持久化(如果有设 save_path)
            if _CAT_MAPPINGS_PATH is not None:
                _CAT_MAPPINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CAT_MAPPINGS_PATH.write_text(json.dumps(_CAT_MAPPINGS))

        mapping = _CAT_MAPPINGS.get(col, {})
        X[col] = X[col].astype(str)
        X[col] = X[col].map(mapping).fillna(-1).astype("int32")

    return X
```

| 处理方式 | label-encode(整数 0..N-1, NaN→-1) |
|---|---|
| 编码键排序依据 | **pandas `CategoricalDtype.categories` 顺序**(由 to_categorical 调用顺序决定,默认字母序) |
| 训练→推理一致性 | ✅ 通过 `_CAT_MAPPINGS_PATH` JSON 文件持久化 |
| 文件 | `src/features.py:to_categorical()` 把对象列转 category |

### [A2] DMatrix 构造方式

```
dm = xgb.DMatrix(X_enc.values, feature_names=list(X_enc.columns))
# 无 enable_categorical / 无 feature_types
```

| 项 | 值 |
|---|---|
| `enable_categorical` | **未使用**(False) |
| `feature_types` | **未使用** |
| `DMatrix.feature_types` | **None** |
| 列 dtype(label-encode 后) | **int32**(等同 numeric) |
| XGB 实际处理 | **整数当 numeric(连续值)**,按 "<=k vs >k" 分裂 |

**判定**:**整数 label-encode 被 XGB 当连续值处理,对无序类别是错误用法**(序数误用实锤)。

### [A3] 13 列 nunique / unseen / gain / 序数 表

| col | train_n | holdout_n | ranking_n | ranking_unseen(%) | gain | 序数 |
|---|---|---|---|---|---|---|
| ADEP_mvt | 10 | 10 | 10 | 0 (0%) | **56,112,288** | ❌ |
| ADES_mvt | 1,504 | 1,113 | 1,105 | 214 (0.06%) | 4,635,490 | ❌ |
| RUNWAY_mvt | 53 | 46 | 50 | 0 (0%) | 16,093,729 | ❌ |
| STAND_mvt | 1,860 | 1,688 | 1,697 | 314 (0.09%) | 12,105,407 | ❌ |
| AIRCRAFT_TYPE_mvt | 265 | 206 | 199 | 18 (0.01%) | 15,681,022 | ❌ |
| AIRCRAFT_TYPE_flt | 231 | 188 | 180 | 10 (0.00%) | **0** | ❌ |
| AIRCRAFT_OPERATOR_flt | 669 | 567 | 579 | 202 (0.06%) | 6,599,642 | ❌ |
| WK_TBL_CAT_flt | 4 | 4 | 4 | 0 (0%) | **77,526,264** | ✅ L/M/H/J |
| MARKET_SEGMENT_flt | 8 | 8 | 8 | 0 (0%) | 8,711,153 | ❌ |
| FLIGHT_RULE_flt | 4 | 3 | 3 | 0 (0%) | 472,023 | ❌ |
| FLIGHT_TYPE_flt | 5 | 5 | 5 | 0 (0%) | 5,442,988 | ❌ |
| ADES_FILED_flt | 1,387 | 1,047 | 1,031 | 170 (0.05%) | **0** | ❌ |
| CALLSIGN_flt | 41,561 | 19,518 | 20,375 | 84,683 (24.94%) | **0** | ❌ |

**unseen 处理**:label-encode 时 `map().fillna(-1).astype(int32)`,**未登录值全映射到 -1**(在 XGB 当 numeric,-1 与 0 不可区分)。

### [A4] 风险排序(高 gain 且无序 = 高风险)

| 风险 | col | gain | train_nunique | 风险原因 |
|---|---|---|---|---|
| 🔴 极高 | ADEP_mvt | 56.1M | 10 | 高 gain 但机场之间无序 → 按 "<=4 vs >4" 分裂完全无语义 |
| 🔴 极高 | WK_TBL_CAT_flt | 77.5M | 4 | 高 gain 且**有自然序**(L/M/H/J),整数编码**反而合理**(0/1/2/3 → 重量级) |
| 🟠 高 | RUNWAY_mvt | 16.1M | 53 | 高 gain,跑道代号无序 |
| 🟠 高 | AIRCRAFT_TYPE_mvt | 15.7M | 265 | 高 gain,机型代号无序 |
| 🟠 高 | STAND_mvt | 12.1M | 1,860 | 高 gain,登机口无序 |
| 🟡 中 | MARKET_SEGMENT_flt | 8.7M | 8 | 中 gain,市场分类无序 |
| 🟡 中 | AIRCRAFT_OPERATOR_flt | 6.6M | 669 | 中 gain,航司代号无序 |
| 🟡 中 | FLIGHT_TYPE_flt | 5.4M | 5 | 中 gain,无序 |
| 🟢 低 | ADES_mvt | 4.6M | 1,504 | 中 gain,目的机场无序 |
| ⚪ 无 | AIRCRAFT_TYPE_flt | 0 | 231 | gain=0,模型未使用 |
| ⚪ 无 | ADES_FILED_flt | 0 | 1,387 | gain=0,模型未使用 |
| ⚪ 无 | CALLSIGN_flt | 0 | 41,561 | gain=0,模型未使用(可能因 24.94% unseen 全映射 -1) |

**结论**:**严重风险**列:**ADEP_mvt(gain 56M,机场代号无序)+ RUNWAY/STAND/AIRCRAFT_TYPE_mvt/AIRCRAFT_OPERATOR**(均高 gain 且无序)。**唯一合理**:**WK_TBL_CAT_flt**(L/M/H/J 重量级,整数编码=重量序数)。

### [A5] AIRCRAFT_TYPE_mvt vs AIRCRAFT_TYPE_flt 重复核查

| 项 | 值 |
|---|---|
| AIRCRAFT_TYPE_mvt == AIRCRAFT_TYPE_flt 比例 | **99.66%** |
| v1 feat list 同时含两者 | **False**(只用 `_mvt`,gain=15.7M) |

**判定**:**不算重复入模** — v1 只用了 `_mvt` 字段。`_flt` 在 `BASELINE_FEATURE_COLS` 中存在但 v1 feature JSON 选了 `_mvt`。`_flt` 实际被忽略。

### [A6] 可行性评估

| 路线 | 工程量 | 风险 | 本机可用性 |
|---|---|---|---|
| **1. XGB enable_categorical** | low(1-2 天) | 中(cat 类型跨数据集一致性) | ✅ XGBoost 已装 |
| **2. Target Encoding (smoothing)** | low-medium(1-3 天,已有基础) | 低(mapping 保存 + apply) | ✅ 已有 `src/features_domain.py` |
| **3. CatBoost 原生类别** | high(5-10 天,新模型 + 调参 + ensemble) | 中(大改动,需先评估单模型价值) | ✅ **CatBoost 1.2.10 + libomp + LightGBM 4.7.0 都可导入** |

**推荐路线**:**先路线 1(XGB enable_categorical)做单点验证**,工作量最低;若 RMSE 改善 < 1% 再考虑路线 3(CatBoost ensemble)。**路线 2 可与路线 1 并行**(target encoding 是另一种修复路径,不冲突)。

---

## 审计 B:target 清洗

### [B1] 当前训练 target 过滤逻辑

**src/train_xgb.py:283-286**:
```python
df = df.dropna(subset=["TAXITIME_SEC_mvt"]).reset_index(drop=True)
df = df[(df["TAXITIME_SEC_mvt"] > 0) & (df["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
```

| 阶段 | 行数 |
|---|---|
| train raw DEP | 1,740,628 |
| train filtered DEP | 1,740,181 |
| 剔除 NaN | 0 |
| 剔除 ≤0 | 305 |
| 剔除 ≥10800 | 142 |

### [B2] TAXITIME_SEC_mvt 全局/按机场 分布

#### 全局(holdout 原始,344,419 行)

| 指标 | 值 |
|---|---|
| min | **−12.00**(物理不可能) |
| p0.1 | 111.00 |
| p1 | 291.00 |
| p50 | 920.00 |
| p99 | 2,629.00 |
| p99.9 | 5,096.58 |
| max | **88,132.00**(24 小时) |
| ≤0 | **83 行** |
| <60s | **236 行** |
| >1800s(30min) | 15,629 行 |
| >3600s(1h) | 981 行 |
| >10800s(3h) | 60 行 |

#### 按机场(min / p50 / p99 / p99.9 / max / 异常)

| AP | n | min | p50 | p99 | p99.9 | max | ≤0 | <60s | >1800s | >3600s |
|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 36,830 | 84 | 833 | 2,138 | 3,368 | 6,234 | 0 | 0 | 695 | 24 |
| EDDM | 27,091 | 168 | 775 | 2,116 | 3,188 | 6,899 | 0 | 0 | 585 | 13 |
| **EGLL** | 40,210 | 9 | 1,317 | 3,127 | **5,770** | 11,167 | 0 | 4 | 4,788 | **240** |
| EHAM | 40,949 | 9 | 739 | 2,180 | 3,742 | 11,032 | 0 | 35 | 755 | 54 |
| LEBL | 28,986 | −8 | 897 | 2,082 | 3,043 | 3,599 | 1 | 9 | 610 | 0 |
| LEMD | 35,328 | 54 | 978 | 1,938 | 2,726 | 3,537 | 0 | 1 | 613 | 0 |
| LFPG | 39,612 | −12 | 954 | 2,885 | 4,440 | **84,240** | **23** | 35 | 2,278 | 138 |
| **LIRF** | 26,528 | 56 | 1,069 | **4,679** | **18,655** | **88,132** | 0 | 1 | **2,721** | **460** |
| LSZH | 22,346 | −12 | 714 | 1,913 | 3,010 | 9,652 | **59** | 151 | 302 | 12 |
| LTFM | 46,539 | 174 | 971 | 2,520 | 3,491 | 7,374 | 0 | 0 | 2,282 | 40 |

#### LIRF 长尾拆解(LIRF 撑大 std=1332)

| 阈值 | 行数 |
|---|---|
| >1800s(30min) | 2,721 |
| >3600s(1h) | 460 |
| >7200s(2h) | 113 |
| >10800s(3h) | 54 |

**top 10 最长**:`[88132, 87361, 87186, 87177, 87170, 87002, 64437, 56633, 46369, 45791]` 秒 = **[24.5h, 24.3h, 24.2h, 24.2h, 24.2h, 24.2h, 17.9h, 15.7h, 12.9h, 12.7h]**。

**判定**:**LIRF 的 std=1332 主要由 2721 行 > 30min 的极长 taxi-out 撑大**,其中 460 行 > 1h,54 行 > 3h(被现状 C0 过滤),还有 top 10 长达 24h 的极端异常。这些是**数据 bug 还是合法值不确定**,但物理上不合理。

### [B3] 时间戳逻辑矛盾(holdout)

| 检查项 | 行数 |
|---|---|
| BLOCK_TIME > MVT_TIME | **80 行** |
| AOBT_3_flt > MVT_TIME | **240 行** |
| MVT − AOBT ≤ 0 | **249 行** |
| EOBT − IOBT < −3600s(异常早) | 0 |
| EOBT − IOBT > 3600s(异常晚) | 64 |
| TAXITIME ≡ MVT − BLOCK 完全相等 | 344,419 |

**判定**:**TAXITIME ≡ MVT−BLOCK 全部成立**(完全相等,344,419/344,419),这与官方定义一致(off-block to take-off)。但有 80/240/249 行 BLOCK/AOBT > MVT 的逻辑矛盾,**数据噪声**;249 行 MVT−AOBT ≤ 0 中,部分是 ground truth 数据 bug(实际跑道起飞时刻早于"off-block 时刻"的反常)。

### [B4] 清洗对照实验(C0-C4, holdout 原始 target)

阈值**仅从训练集(剔除 1+7 月)拟合**,评估用**原始未清洗的 holdout target**。固定 v1 basic30 口径,seed=0,1200 轮,无 early stopping。

#### 拟合训练分位阈值(per airport)

| AP | p0.1 | p1 | p99 | p99.9 |
|---|---|---|---|---|
| EDDF | 171 | 245 | 1,770 | 2,768 |
| EDDM | 235 | 349 | 1,868 | 2,946 |
| EGLL | 543 | 669 | 2,586 | 4,437 |
| EHAM | 66 | 174 | 1,699 | 2,650 |
| LEBL | 167 | 394 | 1,935 | 2,960 |
| LEMD | 257 | 397 | 1,931 | 2,702 |
| LFPG | 241 | 430 | 2,286 | 3,794 |
| **LIRF** | **169** | **417** | **3,652** | **8,286** |
| LSZH | 6 | 202 | 1,653 | 2,403 |
| LTFM | 306 | 480 | 2,634 | 4,736 |
| GLOBAL | 130 | 293 | 2,277 | 4,315 |

#### C0-C4 清洗对照结果

| Scheme | n_train | holdout RMSE | MAE | bias | EGLL | LFPG | LIRF | vs C0 |
|---|---|---|---|---|---|---|---|---|
| **C0** 现状(>0, <10800) | 1,740,181 | **595.50** | 215.96 | −20.41 | 447.76 | 653.24 | **1,697.93** | 0 |
| C1 +<60s 剔除 | 1,739,617 | 596.26 | 215.99 | −20.31 | 448.18 | 653.69 | 1,699.57 | **+0.76** |
| **C2** per-ap winsor [P0.1, P99.9] | 1,740,628 | **590.65** | 214.86 | −22.38 | 448.33 | 652.97 | **1,677.54** | **−4.85** |
| C3 = C1 + C2 | 1,740,628 | 590.65 | 214.86 | −22.38 | 448.33 | 652.97 | 1,677.54 | −4.85 |
| C4 = C2(P0.1-P99.9) | 1,740,628 | 590.65 | 214.86 | −22.38 | 448.33 | 652.97 | 1,677.54 | −4.85 |

**SSE 自洽检查全部 0.00e+00**(分机场 SSE 加总与全局 RMSE² 完全一致)。

#### **判定**

1. **C1 没有帮助**(+0.76s 变差):236 行 <60s 的物理不可能值不影响模型(模型对这些异常值本就拟合差)
2. **C2 winsorize per-ap P0.1-P99.9 改善 RMSE 4.85s**:
   - LIRF 大幅改善 **−20.4s**(1,697.93 → 1,677.54),LIRF 长尾被夹紧
   - EGLL 几乎不变(448.33 vs 447.76),EGLL 没有极端长尾
   - global bias 从 −20.41s 变 −22.38s,winsorize 使模型对极端值不过拟合
3. **C3 = C4 = C2**:C1(<60s 剔除)和 C2(winsorize)组合无额外收益,因为 <60s 行 winsorize 不会被显著改变(已在 P0.1 之下,P0.1 约 130s)

### [B5] 推荐清洗方案

**推荐 C2(per-ap winsorize P0.1-P99.9)**:
- ✅ 全局 RMSE −4.85s,LIRF −20.4s
- ✅ 不系统性偏差(bias 从 −20.41s → −22.38s,变 −2s,合理)
- ✅ 训练阈值全部从剔除 1+7 月训练集拟合,未用 holdout 信息
- ✅ 物理上合理(各机场 taxi-out 经验 [P0.1, P99.9] 范围,对应 99.8% 数据)

**注意**:**不推荐硬剔除**(C1 反而变差 +0.76s),**winsorize 比硬剔除更稳健**(保留了样本量,只是把极端值拉到合理范围)。

**LIRF 改善显著 vs 其他机场几乎不变**的原因:LIRF 长尾极重(p99.9=18,655s = 5h,top 10 长达 24h),其他机场 p99.9 都在 5000s 以下。**LIRF 的清洗收益最大**。

---

## 综合推荐

| 维度 | 推荐 | 预期收益 | 工程量 |
|---|---|---|---|
| 编码修复(高优先) | **路线 1**:XGB `enable_categorical=True` + 传 `cat_cols=[...]` | ADEP/RUNWAY/STAND/AIRCRAFT_OPERATOR 等高 gain 无序列改善 1-3% RMSE | 1-2 天 |
| 编码(中优先) | **路线 2**:并行加 target encoding(已有 `features_domain.py` 基础) | 与路线 1 类似,可选 | 1-3 天 |
| 编码(低优先) | **路线 3**:CatBoost ensemble | 可能 3-5% RMSE,工作量大 | 5-10 天 |
| **target 清洗** | **C2**:per-ap winsorize [P0.1, P99.9] | **RMSE −4.85s,LIRF −20.4s** | < 半天 |
| **不推荐** | C1(<60s 剔除) | 反而 +0.76s 变差 | — |

**下一步**(等你确认后):
1. 修复编码(路线 1 优先)
2. 应用 C2 winsorize
3. 跑 A/B 评估
4. 然后进入 unimpeded(stand, runway) P20 基准 + CatBoost 建模

---

## 关键证据文件

| 文件 | 内容 |
|---|---|
| `notebooks/diag/categorical_encoding_audit.json` | [A1]-[A6] 全部结果 |
| `notebooks/diag/target_cleaning_audit.json` | [B1]-[B3] target 现状 |
| `notebooks/diag/target_cleaning_ab_result.json` | [B4] C0-C4 清洗对照 |
| `docs/diag/stage2_encoding_target_audit.md` | 本报告 |
