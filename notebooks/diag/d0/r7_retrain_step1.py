"""
第一步: R7slim 单库最小验证 (XGB 3 seed, holdout)
- 训练折: 2025-02~12 (剔除 1+7)
- 验证折: 2025-01+07 (holdout)
- 特征: 69 个 (R7slim)
- 训练参数: tuned_xgb_params.json (best trial)
- LUT/类别编码: 只用训练折构建, 禁用 holdout
- 判据: XGB 3seed holdout 均值 ≈ 282-284
"""
import sys, json, time
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

# ============================================================
# 列清单 (严格按留存 R7slim 模型的 69 特征)
# ============================================================
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
assert len(FEATS69) == 69

# 列读取清单
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

# ============================================================
# 1. 加载训练 (2-12月, 剔除 1+7)
# ============================================================
print("=" * 70)
print("[1] 加载训练数据 (2025-02~12)")
print("=" * 70)
months_train = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months_train:
    fname = (f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
             if m < 12 else "training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[
    (df_tr["PHASE_mvt"] == "DEP") &
    (df_tr["TAXITIME_SEC_mvt"] > 0) &
    (df_tr["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  训练 DEP: {len(df_tr_dep):,}")

# ============================================================
# 2. 加载 holdout (1+7月)
# ============================================================
print()
print("=" * 70)
print("[2] 加载 holdout (2025-01+07)")
print("=" * 70)
ho_dfs = []
for m in [1, 7]:
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
df_ho_dep = df_ho[
    (df_ho["PHASE_mvt"] == "DEP") &
    (df_ho["TAXITIME_SEC_mvt"] > 0) &
    (df_ho["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  Holdout DEP: {len(df_ho_dep):,}")

# ============================================================
# 3. 训练折构建 LUT (只用训练折!)
# ============================================================
print()
print("=" * 70)
print("[3] 训练折 LUT (禁用 holdout)")
print("=" * 70)
lut = L.build_t_unimp_lut(df_tr_dep)
print(f"  LUT: L1={len(lut['L1_q'])}, L2={len(lut['L2_q'])}, L3={len(lut['L3_q'])}, global={lut['global_q30']:.0f}s")

# ============================================================
# 4. 训练折重建特征 (只用训练折)
# ============================================================
print()
print("=" * 70)
print("[4] 训练折重建特征 (只用训练折)")
print("=" * 70)

def build_features_in_split(df_raw, ref_dep_for_d, lut):
    """重建全套特征 (与 holdout_r7_rebuild 一致). ref_dep_for_d 用于 D 特征"""
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(
        df_basic.copy(),
        include_a=True, include_b=True, include_c=True,
        include_d=True, include_e=True,
    )
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    if "ADES_flt" in df_full.columns and "ADES_FILED_flt" in df_full.columns:
        df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    else:
        df_full["is_diverted"] = 0

    # D 特征 (跳过 R8drift, 留待第二步)
    # t_unimp
    t_vals, t_src = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals

    # 计数 + dly + MVT (DEP 子集)
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    extra_dict = L.compute_extra_features(df_full[dep_mask], label="")
    for k, v in extra_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v

    mvt_dict = L.compute_mvt_anchor_features(
        df_full[dep_mask],
        t_unimp_vals=t_vals[dep_mask], label="")
    for k, v in mvt_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v

    return df_full

df_tr_full = build_features_in_split(df_tr_raw, df_tr_dep, lut)
dep_mask_tr = (df_tr_full["PHASE_mvt"] == "DEP").values
df_tr_dep_full = df_tr_full[dep_mask_tr].copy().reset_index(drop=True)
print(f"  训练 DEP 全套特征: {df_tr_dep_full.shape}")

# ============================================================
# 5. holdout 重建特征 (LUT 来自训练折, 严格隔离)
# ============================================================
print()
print("=" * 70)
print("[5] Holdout 重建特征 (LUT 来自训练折)")
print("=" * 70)
df_ho_full = build_features_in_split(df_ho_raw, df_tr_dep, lut)
dep_mask_ho = (df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = df_ho_full[dep_mask_ho].copy().reset_index(drop=True)
print(f"  Holdout DEP 全套特征: {df_ho_dep_full.shape}")

# 检查所有 69 特征都存在
for split_name, df_check in [("train", df_tr_dep_full), ("holdout", df_ho_dep_full)]:
    miss = [f for f in FEATS69 if f not in df_check.columns]
    if miss:
        print(f"  ⚠️ {split_name} 缺失: {miss}")
        for f in miss:
            df_check[f] = np.nan

# ============================================================
# 6. 训练参数 (tuned best)
# ============================================================
print()
print("=" * 70)
print("[6] XGB 训练参数 (tuned_xgb_params.json best trial 22)")
print("=" * 70)
best_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "max_depth": 11,
    "min_child_weight": 5,
    "subsample": 0.9813515130184908,
    "colsample_bytree": 0.6720708449966734,
    "reg_alpha": 2.4624169162161342,
    "reg_lambda": 1.9936339729896335,
    "tree_method": "hist",
    "nthread": -1,
    "verbosity": 0,
}
print(f"  params: lr={best_params['learning_rate']}, max_depth={best_params['max_depth']}, "
      f"subsample={best_params['subsample']}, colsample={best_params['colsample_bytree']}")

# ============================================================
# 7. 类别编码: 用训练折 fit, holdout 用同一映射
# ============================================================
print()
print("=" * 70)
print("[7] 类别编码 (训练 fit, holdout 同映射)")
print("=" * 70)
from src import train_xgb as tx

# 训练 fit
X_tr = df_tr_dep_full[FEATS69].copy()
X_tr_enc = tx._encode_categoricals(X_tr, fit=True)
print(f"  训练 fit 后 cat mappings: {len(tx._CAT_MAPPINGS)} 列")

# holdout 用同映射
X_ho = df_ho_dep_full[FEATS69].copy()
X_ho_enc = tx._encode_categoricals(X_ho, fit=False)

# 检查列序一致
assert list(X_tr_enc.columns) == list(X_ho_enc.columns) == FEATS69, "列序不一致!"
print(f"  ✅ 列序一致 (X_tr 和 X_ho 都是 FEATS69 顺序)")

# ============================================================
# 8. 训练 XGB 3 seed + 评估
# ============================================================
print()
print("=" * 70)
print("[8] XGB 3 seed 训练 + holdout 评估")
print("=" * 70)

y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)

X_tr_mat = X_tr_enc.values.astype(np.float32)
X_ho_mat = X_ho_enc.values.astype(np.float32)

dtrain = xgb.DMatrix(X_tr_mat, label=y_tr, feature_names=FEATS69)
dholdout = xgb.DMatrix(X_ho_mat, label=y_ho, feature_names=FEATS69)

seeds = [42, 123, 2024]
results = {}
ens_preds = np.zeros(len(y_ho), dtype=np.float64)

for seed in seeds:
    print(f"\n  --- seed = {seed} ---")
    t0 = time.time()
    params = {**best_params, "seed": seed}
    booster = xgb.train(
        params, dtrain,
        num_boost_round=2500,
        evals=[(dholdout, "holdout")],
        early_stopping_rounds=100,
        verbose_eval=300,
    )
    best_iter = booster.best_iteration + 1
    pred = booster.predict(dholdout, iteration_range=(0, best_iter))
    pred = np.clip(pred, 30, 10800)
    rmse = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
    mae = float(np.mean(np.abs(pred - y_ho)))
    bias = float(np.mean(pred - y_ho))
    print(f"  best_iter={best_iter}, RMSE={rmse:.2f}, MAE={mae:.2f}, Bias={bias:+.2f}, time={time.time()-t0:.1f}s")
    results[seed] = {"rmse": rmse, "mae": mae, "bias": bias, "best_iter": best_iter}
    ens_preds += pred / len(seeds)

ens_rmse = float(np.sqrt(np.mean((ens_preds - y_ho) ** 2)))
print()
print("=" * 70)
print(f"[XGB 3-seed 集成 holdout RMSE]")
print("=" * 70)
print(f"  集成 RMSE = {ens_rmse:.2f}")
for s, r in results.items():
    print(f"  seed {s}: RMSE = {r['rmse']:.2f}")
print()
print(f"  锚点 R7slim = 283.55")
print(f"  判据: 282-284 ✅ 通过 / 明显 >300 ⚠️ 停下")

# 保存
out = {
    "task": "R7slim XGB 3 seed holdout retrain (修复后管线)",
    "params": {**best_params, "seeds": seeds},
    "seeds_results": results,
    "ensemble_rmse": ens_rmse,
    "anchor_r7slim": 283.55,
    "n_train_dep": len(df_tr_dep_full),
    "n_holdout_dep": len(df_ho_dep_full),
    "verdict": "✅ 通过" if ens_rmse <= 290 else ("⚠️ 部分通过" if ens_rmse <= 310 else "❌ 明显偏高, 停下"),
}
out_path = ROOT / "notebooks/diag/d0/r7_retrain_seed3.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  结果已保存: {out_path}")