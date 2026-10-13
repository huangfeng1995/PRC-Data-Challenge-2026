"""
单月压测: 2025-03 DEP 跑 1 个 LGB seed
- num_leaves=63 (减半), max_depth=11, early stopping
- 高基数 cat 列改用 *_cat_mappings 整数编码 (不再走 LGB 原生 cat)
- 低基数列保留 LGB 原生 cat_features
- 记录: 耗时, iter/s, best_iter, holdout RMSE
"""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
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

# 71 特征 (R7slim 69 + R8drift 2)
FEATS71 = [
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
    "rwy_month_share", "ap_month_hourly_density",
]

# 低基数 cat 列 (保留 LGB 原生)
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
# 高基数 cat 列 → 用整数编码, 走数值列
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
                  "AIRCRAFT_OPERATOR_flt"]

print("=" * 70)
print("[加载数据] 训练 = 2025-02 (LUT 用), 压测训练 = 2025-03, holdout = 2025-01+07")
print("=" * 70)

# 加载 2025-02 (作 LUT 用)
df_feb = load_month("data/raw/training_2025-02-01_2025-03-01.parquet", columns=TRAIN_COLS)
df_feb = parse_time_columns(df_feb, cols=TIME_COLS)
df_feb_dep = df_feb[(df_feb["PHASE_mvt"]=="DEP")&(df_feb["TAXITIME_SEC_mvt"]>0)&(df_feb["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)
print(f"  02月 DEP (LUT用): {len(df_feb_dep):,}")

# 加载 2025-03 (压测训练)
df_mar = load_month("data/raw/training_2025-03-01_2025-04-01.parquet", columns=TRAIN_COLS)
df_mar = parse_time_columns(df_mar, cols=TIME_COLS)
df_mar_dep = df_mar[(df_mar["PHASE_mvt"]=="DEP")&(df_mar["TAXITIME_SEC_mvt"]>0)&(df_mar["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)
print(f"  03月 DEP (压测): {len(df_mar_dep):,}")

# 加载 holdout (1+7)
ho_dfs = []
for m in [1, 7]:
    fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_ho = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho, cols=TIME_COLS)
df_ho_dep = df_ho[(df_ho["PHASE_mvt"]=="DEP")&(df_ho["TAXITIME_SEC_mvt"]>0)&(df_ho["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)
print(f"  Holdout DEP (全): {len(df_ho_dep):,}")

# LUT
lut = L.build_t_unimp_lut(df_feb_dep)

# 重建特征函数
def build_features(df_raw, ref_dep_for_d, lut, label=""):
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

print()
print("=" * 70)
print("[重建特征] 03月 + holdout")
print("=" * 70)

t0 = time.time()
df_mar_full = build_features(df_mar, df_mar, lut, label="mar")
df_mar_dep_full = df_mar_full[df_mar_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
df_mar_dep_full = df_mar_dep_full[(df_mar_dep_full["TAXITIME_SEC_mvt"]>0)&(df_mar_dep_full["TAXITIME_SEC_mvt"]<10800)].reset_index(drop=True)
print(f"  03月 DEP 全套特征: {df_mar_dep_full.shape} (用 {time.time()-t0:.0f}s)")

t0 = time.time()
df_ho_full = build_features(df_ho, df_mar, lut, label="ho")
df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
df_ho_dep_full = df_ho_dep_full[(df_ho_dep_full["TAXITIME_SEC_mvt"]>0)&(df_ho_dep_full["TAXITIME_SEC_mvt"]<10800)].reset_index(drop=True)
print(f"  Holdout DEP 全套特征: {df_ho_dep_full.shape} (用 {time.time()-t0:.0f}s)")

# cat 整数编码 (高基数)
print()
print("=" * 70)
print("[cat 整数编码]")
print("=" * 70)
cat_map_path = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
cat_map = json.loads(cat_map_path.read_text())

for col in HIGH_CARD_CATS:
    if col in df_mar_dep_full.columns:
        m = cat_map.get(col, {})
        df_mar_dep_full[col] = df_mar_dep_full[col].astype(str).map(m).fillna(-1).astype("int32")
        df_ho_dep_full[col] = df_ho_dep_full[col].astype(str).map(m).fillna(-1).astype("int32")

# 低基数 → category dtype (LGB 原生 cat)
for col in LOW_CARD_CATS:
    if col in df_mar_dep_full.columns:
        df_mar_dep_full[col] = df_mar_dep_full[col].astype("category")
        df_ho_dep_full[col] = df_ho_dep_full[col].astype("category")

# 检查 71 列齐
for split, df_check in [("mar", df_mar_dep_full), ("ho", df_ho_dep_full)]:
    miss = [f for f in FEATS71 if f not in df_check.columns]
    if miss:
        print(f"  ⚠️ {split} 缺失: {miss}")
        for f in miss:
            df_check[f] = np.nan

# ============================================================
# LGB 压测
# ============================================================
print()
print("=" * 70)
print("[LGB 压测: 2025-03 跑 1 seed, num_leaves=63, max_depth=11]")
print("=" * 70)

X_mar = df_mar_dep_full[FEATS71].copy()
X_ho = df_ho_dep_full[FEATS71].copy()
y_mar = df_mar_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)

# 只保留低基数 cat_features
lgb_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363, "num_leaves": 63, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1, "seed": 42,
}

t0 = time.time()
lgb_train = lgb.Dataset(X_mar, label=y_mar, categorical_feature=LOW_CARD_CATS)
lgb_holdout = lgb.Dataset(X_ho, label=y_ho, categorical_feature=LOW_CARD_CATS, reference=lgb_train)
print(f"  训练数据: {X_mar.shape}, cat_features={LOW_CARD_CATS}")
print(f"  Holdout: {X_ho.shape}")

t_start = time.time()
booster = lgb.train(
    lgb_params, lgb_train,
    num_boost_round=2500,
    valid_sets=[lgb_holdout],
    callbacks=[
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(100)
    ]
)
total_time = time.time() - t_start
best_iter = booster.best_iteration
pred = booster.predict(X_ho, num_iteration=best_iter)
pred = np.clip(pred, 30, 10800)
rmse = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
mae = float(np.mean(np.abs(pred - y_ho)))
iter_per_sec = best_iter / total_time

print(f"\n[LGB 压测结果]")
print(f"  best_iter: {best_iter}")
print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
print(f"  iter/s: {iter_per_sec:.3f}")
print(f"  RMSE (holdout 全样本): {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  n_train: {len(y_mar)}, n_holdout: {len(y_ho)}")

# 外推全量耗时
print()
print("=" * 70)
print("[外推全量(12月)耗时]")
print("=" * 70)
# 训练样本比例: 全量 1.74M / 单月 ~165K ≈ 10.5x
# 单 seed 压测耗时 推算
# 单月耗时 = total_time (包含 setup + 训练)
# 加载/特征重建占一部分 (估计 60s)
# 训练实际时间 ≈ 80% of total
train_only = total_time * 0.8
scale = 1740181 / len(df_mar_dep)  # 全量 2-12月 DEP ≈ 1.74M (来自之前测试)
est_full_seed = train_only * scale
print(f"  03月训练样本: {len(df_mar_dep):,}")
print(f"  全量(2-12月)训练样本: 1,740,181 (估算)")
print(f"  比例: {scale:.1f}x")
print(f"  压测训练耗时(含加载): {total_time:.1f}s")
print(f"  训练本身估: {train_only:.1f}s")
print(f"  **外推单seed全量: {est_full_seed:.0f}s ({est_full_seed/60:.1f} 分钟)**")
print(f"  3seed 估: {est_full_seed*3/60:.1f} 分钟")

if est_full_seed <= 1800:  # ≤30分钟
    print(f"  ✅ 单seed全量 ≤30分钟, 可立即全量跑")
else:
    print(f"  ⚠️ 仍过慢, 需进一步调参")