"""
分析 r7_retrain_step1 重训结果的预测分布
"""
import sys
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

# 直接重新训练一个 seed=42 模型, 但这次看预测 vs 真值分布
from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
from src import train_xgb as tx

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

# 加载训练 (2-12月)
print("加载训练...")
months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months:
    fn = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet" if m<12 else "training_2025-12-01_2026-01-01.parquet"
    train_dfs.append(load_month(f"data/raw/{fn}", columns=TRAIN_COLS))
df_tr = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr, cols=TIME_COLS)
df_tr_dep = df_tr[(df_tr["PHASE_mvt"]=="DEP")&(df_tr["TAXITIME_SEC_mvt"]>0)&(df_tr["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)
print(f"训练 DEP: {len(df_tr_dep):,}")

# LUT
lut = L.build_t_unimp_lut(df_tr_dep)

# 重建训练特征
df_basic = build_basic(df_tr)
df_full = add_queue_features(df_basic, include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
t_vals, _ = L.lookup_t_unimp(df_full, lut)
df_full["t_unimp_r7"] = t_vals
dep_mask = (df_full["PHASE_mvt"]=="DEP").values
extra_dict = L.compute_extra_features(df_full[dep_mask], label="")
for k,v in extra_dict.items():
    df_full[k] = np.nan
    df_full.loc[dep_mask, k] = v
mvt_dict = L.compute_mvt_anchor_features(df_full[dep_mask], t_unimp_vals=t_vals[dep_mask], label="")
for k,v in mvt_dict.items():
    df_full[k] = np.nan
    df_full.loc[dep_mask, k] = v

df_tr_dep_full = df_full[dep_mask].copy().reset_index(drop=True)
y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values

# 加载 holdout
print("加载 holdout...")
ho_dfs = []
for m in [1, 7]:
    fn = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fn}", columns=TRAIN_COLS))
df_ho = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho, cols=TIME_COLS)
df_ho_dep = df_ho[(df_ho["PHASE_mvt"]=="DEP")&(df_ho["TAXITIME_SEC_mvt"]>0)&(df_ho["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)

# 重建 holdout 特征
df_basic_h = build_basic(df_ho)
df_full_h = add_queue_features(df_basic_h, include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
df_full_h["ref_was_estimated"] = df_full_h["AOBT_3_flt"].isna().astype("int8")
df_full_h["is_diverted"] = (df_full_h["ADES_flt"] != df_full_h["ADES_FILED_flt"]).astype("int8")
t_vals_h, _ = L.lookup_t_unimp(df_full_h, lut)
df_full_h["t_unimp_r7"] = t_vals_h
dep_mask_h = (df_full_h["PHASE_mvt"]=="DEP").values
extra_dict_h = L.compute_extra_features(df_full_h[dep_mask_h], label="")
for k,v in extra_dict_h.items():
    df_full_h[k] = np.nan
    df_full_h.loc[dep_mask_h, k] = v
mvt_dict_h = L.compute_mvt_anchor_features(df_full_h[dep_mask_h], t_unimp_vals=t_vals_h[dep_mask_h], label="")
for k,v in mvt_dict_h.items():
    df_full_h[k] = np.nan
    df_full_h.loc[dep_mask_h, k] = v

df_ho_dep_full = df_full_h[dep_mask_h].copy().reset_index(drop=True)
y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values
print(f"Holdout DEP: {len(df_ho_dep_full):,}")

# 类别编码 (训练 fit, holdout 同映射)
X_tr = df_tr_dep_full[FEATS69].copy()
X_tr_enc = tx._encode_categoricals(X_tr, fit=True).values.astype(np.float32)
X_ho = df_ho_dep_full[FEATS69].copy()
X_ho_enc = tx._encode_categoricals(X_ho, fit=False).values.astype(np.float32)

# 训练简化版 (num_boost_round=1000 即可)
import time
best_params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.0147555939449363, "max_depth": 11, "min_child_weight": 5,
    "subsample": 0.9813515130184908, "colsample_bytree": 0.6720708449966734,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "tree_method": "hist", "nthread": -1, "verbosity": 0, "seed": 42,
}
dtrain = xgb.DMatrix(X_tr_enc, label=y_tr, feature_names=FEATS69)
dholdout = xgb.DMatrix(X_ho_enc, label=y_ho, feature_names=FEATS69)

print("\n训练 1500 iter ...")
t0 = time.time()
booster = xgb.train(best_params, dtrain, num_boost_round=1500,
                    evals=[(dtrain, "train"), (dholdout, "holdout")],
                    early_stopping_rounds=100, verbose_eval=200)
print(f"\n训练耗时: {time.time()-t0:.0f}s")

# 训练集预测
pred_tr = booster.predict(dtrain)
pred_tr = np.clip(pred_tr, 30, 10800)
rmse_tr = float(np.sqrt(np.mean((pred_tr - y_tr) ** 2)))
mae_tr = float(np.mean(np.abs(pred_tr - y_tr)))
print(f"\n训练集 in-sample RMSE = {rmse_tr:.2f}, MAE = {mae_tr:.2f}")

# Holdout 预测
pred_ho = booster.predict(dholdout)
pred_ho_raw = pred_ho.copy()
pred_ho = np.clip(pred_ho, 30, 10800)
rmse_ho = float(np.sqrt(np.mean((pred_ho - y_ho) ** 2)))
mae_ho = float(np.mean(np.abs(pred_ho - y_ho)))
print(f"\nHoldout RMSE = {rmse_ho:.2f}, MAE = {mae_ho:.2f}")

# 分析极端误差来源
err_tr = np.abs(pred_tr - y_tr)
err_ho = np.abs(pred_ho - y_ho)
print(f"\n训练集误差分布:")
print(f"  > 1000s: {(err_tr > 1000).sum()}, > 5000s: {(err_tr > 5000).sum()}, > 8000s: {(err_tr > 8000).sum()}")
print(f"  误差 top 20: {sorted(err_tr, reverse=True)[:20]}")

print(f"\nHoldout 误差分布:")
print(f"  > 1000s: {(err_ho > 1000).sum()}, > 5000s: {(err_ho > 5000).sum()}, > 8000s: {(err_ho > 8000).sum()}")
print(f"  误差 top 20: {sorted(err_ho, reverse=True)[:20]}")

# 看最差的几个 holdout 样本
top_err_idx = np.argsort(-err_ho)[:5]
print(f"\nHoldout 最差 5 个预测:")
print(f"{'idx':>4s} {'MVT_ID':>10s} {'ADEP':>6s} {'真值':>8s} {'预测':>8s} {'误差':>8s}")
for i in top_err_idx:
    print(f"{i:>4d} {df_ho_dep_full['MVT_ID_mvt'].iloc[i]:>10.0f} {df_ho_dep_full['ADEP_mvt'].iloc[i]:>6s} "
          f"{y_ho[i]:>8.0f} {pred_ho[i]:>8.0f} {err_ho[i]:>8.0f}")

# 看 pred_raw vs clipped
print(f"\nHoldout pred_raw (未 clip) 分布:")
print(f"  min={pred_ho_raw.min():.0f}, p1={np.percentile(pred_ho_raw,1):.0f}, "
      f"p50={np.median(pred_ho_raw):.0f}, p99={np.percentile(pred_ho_raw,99):.0f}, max={pred_ho_raw.max():.0f}")