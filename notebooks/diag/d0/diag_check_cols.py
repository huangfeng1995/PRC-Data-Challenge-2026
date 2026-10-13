"""
检查 02 月训练数据 + holdout 数据上 FEATS69 的分布, 找错位特征
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L

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

# 加载 02 月
df_feb = load_month("data/raw/training_2025-02-01_2025-03-01.parquet", columns=TRAIN_COLS)
df_feb = parse_time_columns(df_feb, cols=TIME_COLS)
df_feb_dep = df_feb[(df_feb["PHASE_mvt"]=="DEP")&(df_feb["TAXITIME_SEC_mvt"]>0)&(df_feb["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)
lut = L.build_t_unimp_lut(df_feb_dep)

df_basic = build_basic(df_feb)
df_full = add_queue_features(df_basic, include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
if "ADES_flt" in df_full.columns:
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
else:
    df_full["is_diverted"] = 0
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

df_dep = df_full[dep_mask].copy().reset_index(drop=True)
print(f"02月训练 DEP: {len(df_dep):,}")
y = df_dep["TAXITIME_SEC_mvt"].values
print(f"  TAXITIME: min={y.min():.0f}, p50={np.median(y):.0f}, p90={np.percentile(y,90):.0f}, max={y.max():.0f}")

# 逐列
print(f"\n{'特征':28s} {'dtype':>15s} {'n_valid':>8s} {'min':>10s} {'p50':>10s} {'max':>10s}")
print("-" * 90)
for f in FEATS69:
    if f not in df_dep.columns:
        print(f"{f:28s} ⚠️ MISSING")
        continue
    s = df_dep[f]
    try:
        v = pd.to_numeric(s, errors="coerce").values
        valid = ~np.isnan(v)
        if valid.sum() == 0:
            print(f"{f:28s} {str(s.dtype):>15s} {'0':>8s}")
        else:
            print(f"{f:28s} {str(s.dtype):>15s} {valid.sum():>8d} {np.nanmin(v):>10.2f} {np.nanmedian(v):>10.2f} {np.nanmax(v):>10.2f}")
    except Exception as e:
        print(f"{f:28s} {str(s.dtype):>15s} ERROR: {e}")

# 对比留存 R7slim 模型训练时的特征分布
# 看 R7slim 的 feature_names 和我们的是否完全一致
import xgboost as xgb
bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
print(f"\nR7slim 模型 feature_names 数: {len(bst.feature_names)}")
mismatch = []
for i, f in enumerate(FEATS69):
    if i < len(bst.feature_names) and f != bst.feature_names[i]:
        mismatch.append((i, f, bst.feature_names[i]))
if mismatch:
    print(f"⚠️ 列序不一致 ({len(mismatch)} 处):")
    for i, f_my, f_bst in mismatch:
        print(f"  pos {i}: 我的={f_my}, 模型={f_bst}")
else:
    print(f"✅ 列序完全一致")

# 对比训练集特征分布 vs 留存模型训练时的特征 (没有留存, 看模型预测 in-sample)
from src import train_xgb as tx
X = df_dep[FEATS69].copy()
X_enc = tx._encode_categoricals(X, fit=True).values.astype(np.float32)
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
mae = float(np.mean(np.abs(pred - y)))
bias = float(np.mean(pred - y))
print(f"\n留存 R7slim 模型在 02 月训练集上 (in-sample, 模型没见过的):")
print(f"  RMSE={rmse:.2f}, MAE={mae:.2f}, Bias={bias:+.2f}")