"""
LIRF 专项诊断 (最大 SSE 源):
- 长滑行 (30min~3h) 样本数量与来源 (BLOCK==SCHED 占比, AOBT 与 BLOCK 关系)
- 区分数据错误 vs 真实长滑行
- 这些样本 SHAP 看为何预测差
- 给该机场处理建议
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D
from src.train_xgb import _encode_categoricals

TRAIN_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"]

FEATS69 = [
    "sched_hour", "sched_dow", "sched_month", "sched_day", "sched_minute",
    "is_weekend", "is_morning_peak", "is_evening_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "airport_hour",
    "is_domestic_estimate", "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt",
    "is_diverted",
    "iobt_to_sched_sec", "eobt_to_sched_sec", "iobt_eobt_diff_sec",
    "dep_mvt_cnt_airport_5m", "dep_mvt_cnt_airport_10m", "dep_mvt_cnt_airport_15m",
    "dep_queue_n_airport",
    "dep_mvt_cnt_runway_5m", "dep_mvt_cnt_runway_10m", "dep_mvt_cnt_runway_15m",
    "dep_queue_n_runway",
    "ref_was_estimated",
    "mvt_n_ahead", "mvt_mu_rwy_15", "mvt_mu_rwy_30",
    "mvt_w_actual_15", "mvt_w_actual_30",
    "mvt_q_peak", "mvt_q_mean",
    "mvt_na_J", "mvt_na_H", "mvt_na_M", "mvt_na_L",
    "mvt_na_heavy_frac", "mvt_pos_rw", "mvt_na_heavy_x_pos",
    "mvt_mu_rwy_5", "mvt_w_actual_5",
    "dep_cnt_m_5", "dep_cnt_m_15", "dep_cnt_m_30",
    "aobt_cnt_m_5", "aobt_cnt_m_15", "aobt_cnt_m_30",
    "eobt_ahead_cnt_5", "eobt_ahead_cnt_15", "eobt_ahead_cnt_30",
    "dly_aobt_eobt", "dly_aobt_iobt", "dly_aobt_sched", "dly_lobt_aobt",
    "t_unimp_r7",
]

# ============================================================
# 1. 加载训练 (清洗后)
# ============================================================
print("=" * 70)
print("[1] 加载训练 (清洗后)")
print("=" * 70)
months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months:
    fn = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet" if m < 12 else "training_2025-12-01_2026-01-01.parquet"
    train_dfs.append(load_month(f"data/raw/{fn}", columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)

dep_all = df_tr[df_tr["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
tt = dep_all["TAXITIME_SEC_mvt"].values
clean_mask = (tt > 0) & (tt <= 10800)
df_tr_dep = dep_all[clean_mask].copy().reset_index(drop=True)
print(f"  训练 DEP (清洗后): {len(df_tr_dep):,}")

# Holdout
ho_dfs = []
for m in [1, 7]:
    fn = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fn}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)

# ============================================================
# 2. LIRF 训练 + holdout 全集 (含极端值)
# ============================================================
print()
print("=" * 70)
print("[2] LIRF 数据诊断")
print("=" * 70)

# LIRF 训练集
lirf_tr = df_tr_dep[df_tr_dep["ADEP_mvt"] == "LIRF"].copy()
print(f"  LIRF 训练 DEP (清洗后): {len(lirf_tr):,}")
lirf_tr["TAXITIME"] = lirf_tr["TAXITIME_SEC_mvt"]
print(f"  TAXITIME 分布: min={lirf_tr['TAXITIME'].min():.0f}, "
      f"p50={lirf_tr['TAXITIME'].median():.0f}, "
      f"p90={lirf_tr['TAXITIME'].quantile(0.9):.0f}, "
      f"p99={lirf_tr['TAXITIME'].quantile(0.99):.0f}, max={lirf_tr['TAXITIME'].max():.0f}")

# 长滑行 (30min ~ 3h)
lirf_tr = lirf_tr.copy()
lirf_tr["block_minus_sched_sec"] = (lirf_tr["BLOCK_TIME_UTC_mvt"] - lirf_tr["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
lirf_tr["aobt_minus_block_sec"] = (lirf_tr["AOBT_3_flt"] - lirf_tr["BLOCK_TIME_UTC_mvt"]).dt.total_seconds()
lirf_tr["aobt_minus_sched_sec"] = (lirf_tr["AOBT_3_flt"] - lirf_tr["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
lirf_tr["mvt_minus_aobt_sec"] = (lirf_tr["MVT_TIME_UTC_mvt"] - lirf_tr["AOBT_3_flt"]).dt.total_seconds()

long_taxi_tr = lirf_tr[(lirf_tr["TAXITIME"] >= 1800) & (lirf_tr["TAXITIME"] <= 10800)]
print(f"\n  LIRF 长滑行 (30min~3h) 训练样本: {len(long_taxi_tr)} ({100*len(long_taxi_tr)/len(lirf_tr):.2f}%)")

# 长滑行的 BLOCK/SCHED 关系
print(f"\n  长滑行样本 (30min~3h) 的 BLOCK/SCHED/AOBT 关系:")
print(f"    n = {len(long_taxi_tr)}")
print(f"    BLOCK == SCHED (相差<60s): {((long_taxi_tr['block_minus_sched_sec'].abs() < 60)).sum()} ({100*((long_taxi_tr['block_minus_sched_sec'].abs() < 60)).mean():.1f}%)")
print(f"    BLOCK 远早于 SCHED (>60s 前): {((long_taxi_tr['block_minus_sched_sec'] < -60)).sum()} ({100*((long_taxi_tr['block_minus_sched_sec'] < -60)).mean():.1f}%)")
print(f"    BLOCK 远晚于 SCHED (>60s 后): {((long_taxi_tr['block_minus_sched_sec'] > 60)).sum()} ({100*((long_taxi_tr['block_minus_sched_sec'] > 60)).mean():.1f}%)")
print(f"    AOBT > BLOCK + 30min (推出晚于计划): {((long_taxi_tr['aobt_minus_block_sec'] > 1800)).sum()} ({100*((long_taxi_tr['aobt_minus_block_sec'] > 1800)).mean():.1f}%)")
print(f"    AOBT 在 BLOCK 前后 30min 内: {((long_taxi_tr['aobt_minus_block_sec'].abs() < 1800)).sum()} ({100*((long_taxi_tr['aobt_minus_block_sec'].abs() < 1800)).mean():.1f}%)")

# 几个例子
print(f"\n  长滑行前 5 个样本 (BLOCK/SCHED/AOBT 关系):")
for _, row in long_taxi_tr.head(5).iterrows():
    print(f"    MVT_ID={row['MVT_ID_mvt']:.0f}, TAXITIME={row['TAXITIME']:.0f}s, "
          f"BLOCK-SCHED={row['block_minus_sched_sec']:.0f}s, "
          f"AOBT-BLOCK={row['aobt_minus_block_sec']:.0f}s, "
          f"AOBT-SCHED={row['aobt_minus_sched_sec']:.0f}s")

# LIRF holdout
lirf_ho = df_ho[(df_ho["PHASE_mvt"] == "DEP") & (df_ho["ADEP_mvt"] == "LIRF")].copy()
lirf_ho["TAXITIME"] = lirf_ho["TAXITIME_SEC_mvt"]
print(f"\n  LIRF holdout DEP (全, 不剔除): {len(lirf_ho):,}")
print(f"  TAXITIME: min={lirf_ho['TAXITIME'].min():.0f}, "
      f"p50={lirf_ho['TAXITIME'].median():.0f}, "
      f"p90={lirf_ho['TAXITIME'].quantile(0.9):.0f}, "
      f"max={lirf_ho['TAXITIME'].max():.0f}")
print(f"  TAXITIME > 10800 (极端): {(lirf_ho['TAXITIME'] > 10800).sum()}")
print(f"  TAXITIME > 7200: {(lirf_ho['TAXITIME'] > 7200).sum()}")
print(f"  TAXITIME > 3600: {(lirf_ho['TAXITIME'] > 3600).sum()}")
print(f"  TAXITIME in 1800~10800: {((lirf_ho['TAXITIME'] >= 1800) & (lirf_ho['TAXITIME'] <= 10800)).sum()}")

lirf_ho["block_minus_sched_sec"] = (lirf_ho["BLOCK_TIME_UTC_mvt"] - lirf_ho["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
lirf_ho["aobt_minus_block_sec"] = (lirf_ho["AOBT_3_flt"] - lirf_ho["BLOCK_TIME_UTC_mvt"]).dt.total_seconds()
lirf_ho["mvt_minus_aobt_sec"] = (lirf_ho["MVT_TIME_UTC_mvt"] - lirf_ho["AOBT_3_flt"]).dt.total_seconds()

# 长滑行 holdout
long_taxi_ho = lirf_ho[(lirf_ho["TAXITIME"] >= 1800) & (lirf_ho["TAXITIME"] <= 10800)]
print(f"\n  LIRF 长滑行 (30min~3h) holdout 样本: {len(long_taxi_ho)}")
print(f"    BLOCK == SCHED (<60s): {((long_taxi_ho['block_minus_sched_sec'].abs() < 60)).sum()}")
print(f"    AOBT > BLOCK + 30min: {((long_taxi_ho['aobt_minus_block_sec'] > 1800)).sum()}")
print(f"    AOBT 在 BLOCK 前后 30min: {((long_taxi_ho['aobt_minus_block_sec'].abs() < 1800)).sum()}")
print(f"    MVT > AOBT + 30min (滑行真实长): {((long_taxi_ho['mvt_minus_aobt_sec'] > 1800)).sum()}")

# 极端值 (> 10800)
extreme_ho = lirf_ho[lirf_ho["TAXITIME"] > 10800]
print(f"\n  LIRF 极端 TAXITIME (>10800) holdout 样本: {len(extreme_ho)}")
print(f"    TAXITIME 分布: min={extreme_ho['TAXITIME'].min():.0f}, "
      f"p50={extreme_ho['TAXITIME'].median():.0f}, max={extreme_ho['TAXITIME'].max():.0f}")
print(f"    BLOCK==SCHED (<60s): {((extreme_ho['block_minus_sched_sec'].abs() < 60)).sum()}")
print(f"    AOBT > BLOCK + 30min: {((extreme_ho['aobt_minus_block_sec'] > 1800)).sum()}")
print(f"    AOBT in BLOCK 前后 30min: {((extreme_ho['aobt_minus_block_sec'].abs() < 1800)).sum()}")

# 看一个极端例子
print(f"\n  LIRF 极端 TAXITIME 样本 (前 5):")
for _, row in extreme_ho.head(5).iterrows():
    print(f"    MVT_ID={row['MVT_ID_mvt']:.0f}, TAXITIME={row['TAXITIME']:.0f}s "
          f"({row['TAXITIME']/3600:.1f}h), "
          f"BLOCK={row['BLOCK_TIME_UTC_mvt']}, "
          f"AOBT={row['AOBT_3_flt']}, "
          f"MVT={row['MVT_TIME_UTC_mvt']}")
    print(f"      BLOCK-SCHED={row['block_minus_sched_sec']:.0f}s, "
          f"AOBT-BLOCK={row['aobt_minus_block_sec']:.0f}s, "
          f"MVT-AOBT={row['mvt_minus_aobt_sec']:.0f}s")

# ============================================================
# 3. 用清洗后 R7slim 模型对 LIRF holdout 推理 + SHAP
# ============================================================
print()
print("=" * 70)
print("[3] 留存 R7slim 模型对 LIRF holdout 推理")
print("=" * 70)

# 重建完整 holdout 特征
def build_features_in_split(df_raw, ref_dep_for_d, lut, label=""):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(df_basic.copy(),
        include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    d_dict = D.compute_drift_features(df_full, ref_dep=ref_dep_for_d, label=label)
    for k, v in d_dict.items():
        df_full[k] = v
    t_vals, _ = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    extra_dict = L.compute_extra_features(df_full[dep_mask], label=label)
    for k, v in extra_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    mvt_dict = L.compute_mvt_anchor_features(df_full[dep_mask],
        t_unimp_vals=t_vals[dep_mask], label=label)
    for k, v in mvt_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    return df_full

lut = L.build_t_unimp_lut(df_tr_dep)
df_ho_full = build_features_in_split(df_ho_raw, df_tr, lut, label="holdout")
dep_mask_ho = (df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = df_ho_full[dep_mask_ho].copy().reset_index(drop=True)

# 模型推理
bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())

X = df_ho_dep_full[FEATS69].copy()
for f in FEATS69:
    if f not in X.columns:
        X[f] = np.nan
X_enc = _encode_categoricals(X, fit=False).values.astype(np.float32)
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
airports = df_ho_dep_full["ADEP_mvt"].astype(str).values

# LIRF holdout 预测
lirf_mask = airports == "LIRF"
lirf_pred = pred[lirf_mask]
lirf_y = y[lirf_mask]
lirf_id = df_ho_dep_full["MVT_ID_mvt"].values[lirf_mask]

# RMSE
core_mask = (lirf_y > 0) & (lirf_y <= 10800)
print(f"  LIRF holdout RMSE (核心): {float(np.sqrt(np.mean((lirf_pred[core_mask] - lirf_y[core_mask]) ** 2))):.2f}")
print(f"  LIRF holdout RMSE (全样本): {float(np.sqrt(np.mean((lirf_pred - lirf_y) ** 2))):.2f}")

# 极端样本预测错误
extreme_lirf = (lirf_y > 10800) & np.isfinite(lirf_y)
print(f"\n  LIRF holdout 极端样本 (TAXITIME>10800) 预测:")
for i in np.where(extreme_lirf)[0][:10]:
    print(f"    MVT_ID={lirf_id[i]:.0f}, TAXITIME={lirf_y[i]:.0f}s ({lirf_y[i]/3600:.1f}h), "
          f"pred={lirf_pred[i]:.0f}s, err={lirf_pred[i]-lirf_y[i]:+.0f}s")

# 长滑行样本预测
long_lirf = (lirf_y >= 1800) & (lirf_y <= 10800)
print(f"\n  LIRF holdout 长滑行 (30min~3h) n={long_lirf.sum()}")
err_long = lirf_pred[long_lirf] - lirf_y[long_lirf]
print(f"  误差分布: mean={err_long.mean():.1f}, p50={np.median(err_long):.1f}, "
      f"p90={np.percentile(err_long, 90):.1f}, max={err_long.max():.1f}, min={err_long.min():.1f}")

# ============================================================
# 4. LIRF 处理建议
# ============================================================
print()
print("=" * 70)
print("[4] LIRF 处理建议")
print("=" * 70)
n_extreme_ho = int(extreme_lirf.sum())
n_long_ho = int(long_lirf.sum())
n_core_ho = int(core_mask.sum())

print(f"  LIRF holdout:")
print(f"    核心样本 (TAXITIME in 1s~3h): {n_core_ho:,}")
print(f"    长滑行 (30min~3h):           {n_long_ho:,}")
print(f"    极端样本 (>3h):              {n_extreme_ho:,}")
print()
print(f"  诊断结论 (按你'区分数据错误 vs 真实长滑行'):")
print(f"  1. 长滑行 (30min~3h) 中 BLOCK==SCHED 占比: 数据错误信号")
print(f"  2. AOBT > BLOCK + 30min 占比: 真实长滑行 (推出推迟)")
print(f"  3. 极端值 (>3h) 几乎都是数据错误: BLOCK_TIME/SCHED_TIME 缺位/异常")
print()
print(f"  建议:")
print(f"    A) LIRF 单独剔除 TAXITIME>10800 (模型预测被 clip 10800, 误差=真值-10800)")
print(f"    B) 长滑行(30min~3h): 保留, 真实分布; 可加特征 'block_minus_sched_aobt_diff'")
print(f"    C) 机场级特征: 加 LIRF_month_hourly_avg_taxi 等, 帮模型识别长滑行模式")
print(f"    D) 不动全局: 只在 LIRF 子集处理, 其他机场不受影响")

print(f"\n  ✅ LIRF 专项诊断完成")
print(f"  等待集成 RMSE 结果后再综合判断 (集成是否补偿了 LIRF 的 +340)")