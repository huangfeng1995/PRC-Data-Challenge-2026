"""
清洗训练数据 + 重训 R7slim 3 seed + 分层报告

口径:
- 训练时清洗: 剔除 TAXITIME <= 0 或 TAXITIME > 10800
- 保留 1800~10800 (30min~3h) 真实长滑行
- 验证时两种口径:
  * 核心 RMSE: 验证集剔除 > 10800 后评估 (口径对齐锚点 283)
  * 全样本 RMSE: 验证集不剔除 (对齐真榜)
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
# 1. 加载训练 (2-12月) + 清洗
# ============================================================
print("=" * 70)
print("[1] 加载训练数据 + 清洗")
print("=" * 70)
months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months:
    fn = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet" if m < 12 else "training_2025-12-01_2026-01-01.parquet"
    train_dfs.append(load_month(f"data/raw/{fn}", columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)

# 全部 DEP
dep_all = df_tr[df_tr["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
print(f"  训练 DEP (全): {len(dep_all):,}")

# 清洗: 剔除 TAXITIME <= 0 或 TAXITIME > 10800
tt = dep_all["TAXITIME_SEC_mvt"].values
clean_mask = (tt > 0) & (tt <= 10800)
n_le_zero = int((tt <= 0).sum())
n_gt_10800 = int((tt > 10800).sum())
print(f"  清洗前: {len(dep_all):,}")
print(f"  剔除 TAXITIME <= 0: {n_le_zero:,}")
print(f"  剔除 TAXITIME > 10800: {n_gt_10800:,}")
print(f"  清洗后保留: {int(clean_mask.sum()):,}")

# 保留 1800~10800 的长滑行
tt_clean = tt[clean_mask]
print(f"\n  清洗后 TAXITIME 分布:")
print(f"    min={tt_clean.min():.1f}, p50={np.median(tt_clean):.1f}, p90={np.percentile(tt_clean, 90):.1f}")
print(f"    1800~3600: {((tt_clean >= 1800) & (tt_clean < 3600)).sum():,}")
print(f"    3600~10800: {((tt_clean >= 3600) & (tt_clean <= 10800)).sum():,}")

df_tr_dep = dep_all[clean_mask].copy().reset_index(drop=True)
print(f"\n  训练 DEP (清洗后): {len(df_tr_dep):,}")

# ============================================================
# 2. 加载 holdout (1+7月)
# ============================================================
print()
print("=" * 70)
print("[2] 加载 holdout (2025-01+07)")
print("=" * 70)
ho_dfs = []
for m in [1, 7]:
    fn = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fn}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
dep_ho_all = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
print(f"  Holdout DEP (全): {len(dep_ho_all):,}")

# ============================================================
# 3. 构建 LUT (用清洗后训练 DEP)
# ============================================================
print()
print("=" * 70)
print("[3] 构建 LUT (清洗后训练数据)")
print("=" * 70)
lut = L.build_t_unimp_lut(df_tr_dep)
print(f"  LUT: L1={len(lut['L1_q'])}, L2={len(lut['L2_q'])}, L3={len(lut['L3_q'])}, "
      f"global={lut['global_q30']:.0f}s")

# ============================================================
# 4. 重建特征函数 (与 holdout_r7_rebuild 一致)
# ============================================================
def build_features_in_split(df_raw, ref_dep_for_d, lut, label=""):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(
        df_basic.copy(),
        include_a=True, include_b=True, include_c=True,
        include_d=True, include_e=True,
    )
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
    mvt_dict = L.compute_mvt_anchor_features(
        df_full[dep_mask],
        t_unimp_vals=t_vals[dep_mask], label=label)
    for k, v in mvt_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    return df_full

# ============================================================
# 5. 重建训练 + holdout 特征
# ============================================================
print()
print("=" * 70)
print("[5] 重建训练 + holdout 特征")
print("=" * 70)

# 训练用清洗后 df_tr_dep 作 ref_dep_for_d (注意: 这只用于 D 特征, 不影响 LUT)
# 实际上 D 特征要"更早"的数据, 用清洗后的训练月本身合理
df_tr_full = build_features_in_split(df_tr_raw, df_tr, lut, label="train")
dep_mask_tr = (df_tr_full["PHASE_mvt"] == "DEP").values
df_tr_dep_full = df_tr_full[dep_mask_tr].copy().reset_index(drop=True)

# 验证: 训练 DEP 全套特征后, 行数应 == 清洗后 DEP
print(f"  训练 DEP 全套特征后: {len(df_tr_dep_full):,}")

# Holdout 用整个 df_ho (DEP+ARR)
df_ho_full = build_features_in_split(df_ho_raw, df_tr, lut, label="holdout")
dep_mask_ho = (df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = df_ho_full[dep_mask_ho].copy().reset_index(drop=True)
print(f"  Holdout DEP 全套特征后: {len(df_ho_dep_full):,}")

# 检查 69 特征齐
for df_check, name in [(df_tr_dep_full, "train"), (df_ho_dep_full, "holdout")]:
    miss = [f for f in FEATS69 if f not in df_check.columns]
    if miss:
        print(f"  ⚠️ {name} 缺失: {miss}")
        for f in miss:
            df_check[f] = np.nan

# ============================================================
# 6. 类别编码 (训练 fit, holdout 同映射)
# ============================================================
print()
print("=" * 70)
print("[6] 类别编码 (训练 fit, holdout 同映射)")
print("=" * 70)
from src import train_xgb as tx

X_tr = df_tr_dep_full[FEATS69].copy()
X_tr_enc = tx._encode_categoricals(X_tr, fit=True)
print(f"  训练 fit 后 cat mappings: {len(tx._CAT_MAPPINGS)} 列")

X_ho = df_ho_dep_full[FEATS69].copy()
X_ho_enc = tx._encode_categoricals(X_ho, fit=False)
assert list(X_tr_enc.columns) == list(X_ho_enc.columns) == FEATS69, "列序不一致!"
print(f"  ✅ 列序一致")

# ============================================================
# 7. 训练 XGB 3 seed
# ============================================================
print()
print("=" * 70)
print("[7] 训练 XGB 3 seed")
print("=" * 70)

best_params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.0147555939449363, "max_depth": 11, "min_child_weight": 5,
    "subsample": 0.9813515130184908, "colsample_bytree": 0.6720708449966734,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "tree_method": "hist", "nthread": -1, "verbosity": 0,
}
print(f"  params: lr={best_params['learning_rate']}, max_depth={best_params['max_depth']}, "
      f"subsample={best_params['subsample']}, colsample={best_params['colsample_bytree']}")

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
        evals=[(dtrain, "train"), (dholdout, "holdout")],
        early_stopping_rounds=100,
        verbose_eval=300,
    )
    best_iter = booster.best_iteration + 1
    pred = booster.predict(dholdout, iteration_range=(0, best_iter))
    pred = np.clip(pred, 30, 10800)
    rmse_core = float(np.sqrt(np.mean(((pred - y_ho) ** 2)[(y_ho > 0) & (y_ho <= 10800)])))
    rmse_all = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
    print(f"  best_iter={best_iter}, RMSE(剔除>10800)={rmse_core:.2f}, RMSE(全)={rmse_all:.2f}, time={time.time()-t0:.1f}s")
    results[seed] = {"rmse_core": rmse_core, "rmse_all": rmse_all, "best_iter": best_iter}
    ens_preds += pred / len(seeds)

# 集成 RMSE
ens_pred = ens_preds
ens_pred_core = ens_pred[(y_ho > 0) & (y_ho <= 10800)]
y_ho_core = y_ho[(y_ho > 0) & (y_ho <= 10800)]
rmse_ens_core = float(np.sqrt(np.mean((ens_pred_core - y_ho_core) ** 2)))
rmse_ens_all = float(np.sqrt(np.mean((ens_pred - y_ho) ** 2)))
print()
print("=" * 70)
print(f"[XGB 3-seed 集成 holdout RMSE]")
print("=" * 70)
print(f"  集成 RMSE (剔除 > 10800): {rmse_ens_core:.2f}")
print(f"  集成 RMSE (全样本):      {rmse_ens_all:.2f}")
for s, r in results.items():
    print(f"  seed {s}: core={r['rmse_core']:.2f}, all={r['rmse_all']:.2f}")
print(f"\n  锚点 R7slim: 283.55 (core 口径)")
print(f"  真榜口径 (全样本)")

# ============================================================
# 8. 分层报告 (固定分层口径)
# ============================================================
print()
print("=" * 70)
print("[8] 固定分层报告")
print("=" * 70)

aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
months_col = pd.to_datetime(df_ho_dep_full["MVT_TIME_UTC_mvt"]).dt.month.values

# 核心 RMSE (剔除 > 10800)
core_mask = (y_ho > 0) & (y_ho <= 10800)

print(f"\n--- 核心 RMSE (剔除 TAXITIME > 10800) ---")
print(f"  整体 (n={core_mask.sum():,}): RMSE={rmse_ens_core:.2f}")

print(f"\n--- 核心 分机场 ---")
print(f"{'ADEP':6s} {'n':>8s} {'RMSE':>10s} {'Bias':>8s}")
for ap in sorted(np.unique(aps)):
    mask = (aps == ap) & core_mask
    n = int(mask.sum())
    if n < 100:
        continue
    rmse = float(np.sqrt(np.mean((ens_pred[mask] - y_ho[mask]) ** 2)))
    bias = float(np.mean(ens_pred[mask] - y_ho[mask]))
    print(f"  {ap:6s} {n:>8d} {rmse:>10.2f} {bias:>+8.2f}")

print(f"\n--- 核心 分月份 ---")
for m in [1, 7]:
    mask = (months_col == m) & core_mask
    n = int(mask.sum())
    rmse = float(np.sqrt(np.mean((ens_pred[mask] - y_ho[mask]) ** 2)))
    bias = float(np.mean(ens_pred[mask] - y_ho[mask]))
    print(f"  {m}月: n={n:,}, RMSE={rmse:.2f}, Bias={bias:+.2f}")

print(f"\n--- 全样本 RMSE (不剔除, 对齐真榜) ---")
print(f"  整体 (n={(np.isfinite(y_ho)).sum():,}): RMSE={rmse_ens_all:.2f}")

print(f"\n--- 全样本 分机场 ---")
print(f"{'ADEP':6s} {'n':>8s} {'RMSE':>10s} {'Bias':>8s}")
for ap in sorted(np.unique(aps)):
    mask = (aps == ap) & np.isfinite(y_ho)
    n = int(mask.sum())
    if n < 100:
        continue
    rmse = float(np.sqrt(np.mean((ens_pred[mask] - y_ho[mask]) ** 2)))
    bias = float(np.mean(ens_pred[mask] - y_ho[mask]))
    print(f"  {ap:6s} {n:>8d} {rmse:>10.2f} {bias:>+8.2f}")

# ============================================================
# 9. 总结
# ============================================================
print()
print("=" * 70)
print("[9] 总结对照")
print("=" * 70)
print(f"  README 锚点 R7slim (核心口径): 283.55")
print(f"  重训 XGB 3-seed (核心):         {rmse_ens_core:.2f} (Δ={rmse_ens_core - 283.55:+.2f})")
print(f"  重训 XGB 3-seed (全样本):       {rmse_ens_all:.2f}")
if abs(rmse_ens_core - 283.55) <= 5:
    print(f"  ✅ 核心 RMSE ≈ 283, 地基通过")
else:
    print(f"  ⚠️ 核心 RMSE 偏离锚点, 需要进一步诊断")

# 保存
out = {
    "task": "R7slim 清洗训练数据 + 重训 XGB 3 seed",
    "params": {**best_params, "seeds": seeds},
    "cleaning": {
        "n_train_dep_raw": int(len(dep_all)),
        "n_drop_le_zero": n_le_zero,
        "n_drop_gt_10800": n_gt_10800,
        "n_train_dep_clean": int(len(df_tr_dep)),
        "kept_long_taxi": "1800~10800 (real long taxi, NOT dropped)",
    },
    "seeds_results": {str(s): r for s, r in results.items()},
    "ensemble_rmse_core": rmse_ens_core,
    "ensemble_rmse_all": rmse_ens_all,
    "anchor_r7slim": 283.55,
    "verdict": "✅ 通过" if abs(rmse_ens_core - 283.55) <= 5 else "⚠️ 偏离",
}
out_path = ROOT / "notebooks/diag/d0/r7_clean_retrain_seed3.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  结果已保存: {out_path}")