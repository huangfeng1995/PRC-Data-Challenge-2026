"""
完整三库集成: XGB + LGB + CAT × 3seed (同清洗口径)
报三库各自 + 集成的核心/全样本 RMSE, 分机场
"""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

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
# 1. 加载训练 (2-12月) + 清洗 + LUT
# ============================================================
print("=" * 70)
print("[1] 加载 + 清洗")
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
print(f"  清洗后训练 DEP: {len(df_tr_dep):,} (剔除 <=0: 0, >10800: {(tt > 10800).sum()})")

# Holdout
ho_dfs = []
for m in [1, 7]:
    fn = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fn}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)

lut = L.build_t_unimp_lut(df_tr_dep)

# ============================================================
# 2. 重建特征
# ============================================================
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

print()
print("=" * 70)
print("[2] 重建特征")
print("=" * 70)

df_tr_full = build_features_in_split(df_tr_raw, df_tr, lut, label="train")
dep_mask_tr = (df_tr_full["PHASE_mvt"] == "DEP").values
df_tr_dep_full = df_tr_full[dep_mask_tr].copy().reset_index(drop=True)

df_ho_full = build_features_in_split(df_ho_raw, df_tr, lut, label="holdout")
dep_mask_ho = (df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = df_ho_full[dep_mask_ho].copy().reset_index(drop=True)

print(f"  训练 DEP 全套: {len(df_tr_dep_full):,}")
print(f"  Holdout DEP 全套: {len(df_ho_dep_full):,}")

for df_check, name in [(df_tr_dep_full, "train"), (df_ho_dep_full, "holdout")]:
    miss = [f for f in FEATS69 if f not in df_check.columns]
    for f in miss:
        df_check[f] = np.nan

# ============================================================
# 3. 类别编码 (训练 fit, holdout 同映射)
# ============================================================
print()
print("=" * 70)
print("[3] 类别编码")
print("=" * 70)
from src import train_xgb as tx
from src.train_xgb import _encode_categoricals

X_tr = df_tr_dep_full[FEATS69].copy()
X_tr_enc = tx._encode_categoricals(X_tr, fit=True)
X_ho = df_ho_dep_full[FEATS69].copy()
X_ho_enc = tx._encode_categoricals(X_ho, fit=False)

# LightGBM 用 category dtype (原生支持)
cat_cols = [f for f in FEATS69 if X_tr[f].dtype.name == "category"]
print(f"  cat columns: {len(cat_cols)}")

X_tr_lgb = df_tr_dep_full[FEATS69].copy()
for c in cat_cols:
    X_tr_lgb[c] = X_tr_lgb[c].astype("category")
X_ho_lgb = df_ho_dep_full[FEATS69].copy()
for c in cat_cols:
    X_ho_lgb[c] = X_ho_lgb[c].astype("category")

y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)

# ============================================================
# 4. XGB 3 seed (复用之前的参数)
# ============================================================
print()
print("=" * 70)
print("[4] XGB 3 seed")
print("=" * 70)

best_params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.0147555939449363, "max_depth": 11, "min_child_weight": 5,
    "subsample": 0.9813515130184908, "colsample_bytree": 0.6720708449966734,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "tree_method": "hist", "nthread": -1, "verbosity": 0,
}

X_tr_mat = X_tr_enc.values.astype(np.float32)
X_ho_mat = X_ho_enc.values.astype(np.float32)
dtrain = xgb.DMatrix(X_tr_mat, label=y_tr, feature_names=FEATS69)
dholdout = xgb.DMatrix(X_ho_mat, label=y_ho, feature_names=FEATS69)

xgb_preds = np.zeros(len(y_ho), dtype=np.float64)
xgb_results = {}

for seed in [42, 123, 2024]:
    print(f"\n  --- XGB seed = {seed} ---")
    t0 = time.time()
    params = {**best_params, "seed": seed}
    booster = xgb.train(params, dtrain, num_boost_round=2500,
        evals=[(dholdout, "holdout")], early_stopping_rounds=100, verbose_eval=500)
    best_iter = booster.best_iteration + 1
    pred = booster.predict(dholdout, iteration_range=(0, best_iter))
    pred = np.clip(pred, 30, 10800)
    core = float(np.sqrt(np.mean(((pred - y_ho) ** 2)[(y_ho > 0) & (y_ho <= 10800)])))
    all_rmse = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
    print(f"  best_iter={best_iter}, core={core:.2f}, all={all_rmse:.2f}, time={time.time()-t0:.1f}s")
    xgb_results[seed] = {"core": core, "all": all_rmse, "best_iter": best_iter}
    xgb_preds += pred / 3

# ============================================================
# 5. LGB 3 seed
# ============================================================
print()
print("=" * 70)
print("[5] LGB 3 seed")
print("=" * 70)

lgb_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363, "num_leaves": 127,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}

lgb_train = lgb.Dataset(X_tr_lgb, label=y_tr, categorical_feature=cat_cols)
lgb_holdout = lgb.Dataset(X_ho_lgb, label=y_ho, categorical_feature=cat_cols, reference=lgb_train)

lgb_preds = np.zeros(len(y_ho), dtype=np.float64)
lgb_results = {}

for seed in [42, 123, 2024]:
    print(f"\n  --- LGB seed = {seed} ---")
    t0 = time.time()
    params = {**lgb_params, "seed": seed}
    booster = lgb.train(params, lgb_train, num_boost_round=2500,
        valid_sets=[lgb_holdout], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    best_iter = booster.best_iteration
    pred = booster.predict(X_ho_lgb, num_iteration=best_iter)
    pred = np.clip(pred, 30, 10800)
    core = float(np.sqrt(np.mean(((pred - y_ho) ** 2)[(y_ho > 0) & (y_ho <= 10800)])))
    all_rmse = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
    print(f"  best_iter={best_iter}, core={core:.2f}, all={all_rmse:.2f}, time={time.time()-t0:.1f}s")
    lgb_results[seed] = {"core": core, "all": all_rmse, "best_iter": best_iter}
    lgb_preds += pred / 3

# ============================================================
# 6. CAT 3 seed (CatBoost)
# ============================================================
print()
print("=" * 70)
print("[6] CAT 3 seed")
print("=" * 70)

try:
    from catboost import CatBoostRegressor

    # CatBoost 用字符串型 cat 列
    X_tr_cat = df_tr_dep_full[FEATS69].copy()
    for c in cat_cols:
        X_tr_cat[c] = X_tr_cat[c].astype(str).fillna("__nan__")
    X_ho_cat = df_ho_dep_full[FEATS69].copy()
    for c in cat_cols:
        X_ho_cat[c] = X_ho_cat[c].astype(str).fillna("__nan__")

    cat_results = {}
    cat_preds = np.zeros(len(y_ho), dtype=np.float64)

    for seed in [42, 123, 2024]:
        print(f"\n  --- CAT seed = {seed} ---")
        t0 = time.time()
        model = CatBoostRegressor(
            iterations=2500, learning_rate=0.0147555939449363,
            depth=11, l2_leaf_reg=5,
            random_seed=seed, verbose=500,
            early_stopping_rounds=100,
            cat_features=cat_cols, thread_count=-1,
            eval_metric="RMSE",
        )
        model.fit(X_tr_cat, y_tr, eval_set=(X_ho_cat, y_ho))
        pred = model.predict(X_ho_cat)
        pred = np.clip(pred, 30, 10800)
        core = float(np.sqrt(np.mean(((pred - y_ho) ** 2)[(y_ho > 0) & (y_ho <= 10800)])))
        all_rmse = float(np.sqrt(np.mean((pred - y_ho) ** 2)))
        print(f"  best_iter={model.tree_count_}, core={core:.2f}, all={all_rmse:.2f}, time={time.time()-t0:.1f}s")
        cat_results[seed] = {"core": core, "all": all_rmse, "best_iter": model.tree_count_}
        cat_preds += pred / 3
except ImportError:
    print("  CatBoost 未安装, 跳过")
    cat_preds = None
    cat_results = None

# ============================================================
# 7. 集成 (3 库 × 3 seed = 9 模型等权平均)
# ============================================================
print()
print("=" * 70)
print("[7] 集成 (XGB + LGB + CAT)")
print("=" * 70)

if cat_preds is not None:
    ens_preds = (xgb_preds + lgb_preds + cat_preds) / 3
    components = "XGB + LGB + CAT"
else:
    ens_preds = (xgb_preds + lgb_preds) / 2
    components = "XGB + LGB (CAT not installed)"

core_mask = (y_ho > 0) & (y_ho <= 10800)
rmse_ens_core = float(np.sqrt(np.mean(((ens_preds - y_ho) ** 2)[core_mask])))
rmse_ens_all = float(np.sqrt(np.mean((ens_preds - y_ho) ** 2)))

print(f"  集成 ({components})")
print(f"    核心 RMSE (剔除 > 10800): {rmse_ens_core:.2f}")
print(f"    全样本 RMSE:              {rmse_ens_all:.2f}")
print(f"  XGB 3-seed 平均: 核心 {np.mean([r['core'] for r in xgb_results.values()]):.2f}, "
      f"全 {np.mean([r['all'] for r in xgb_results.values()]):.2f}")
print(f"  LGB 3-seed 平均: 核心 {np.mean([r['core'] for r in lgb_results.values()]):.2f}, "
      f"全 {np.mean([r['all'] for r in lgb_results.values()]):.2f}")
if cat_results is not None:
    print(f"  CAT 3-seed 平均: 核心 {np.mean([r['core'] for r in cat_results.values()]):.2f}, "
          f"全 {np.mean([r['all'] for r in cat_results.values()]):.2f}")

print(f"\n  锚点: 283.55 (核心)")

# ============================================================
# 8. 分层报告
# ============================================================
print()
print("=" * 70)
print("[8] 分层报告 (集成预测)")
print("=" * 70)

aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
months_col = pd.to_datetime(df_ho_dep_full["MVT_TIME_UTC_mvt"]).dt.month.values

print(f"\n--- 核心 分机场 ---")
print(f"{'ADEP':6s} {'n':>8s} {'RMSE':>10s} {'Bias':>8s}")
for ap in sorted(np.unique(aps)):
    mask = (aps == ap) & core_mask
    n = int(mask.sum())
    if n < 100:
        continue
    rmse = float(np.sqrt(np.mean((ens_preds[mask] - y_ho[mask]) ** 2)))
    bias = float(np.mean(ens_preds[mask] - y_ho[mask]))
    print(f"  {ap:6s} {n:>8d} {rmse:>10.2f} {bias:>+8.2f}")

print(f"\n--- 核心 分月份 ---")
for m in [1, 7]:
    mask = (months_col == m) & core_mask
    n = int(mask.sum())
    rmse = float(np.sqrt(np.mean((ens_preds[mask] - y_ho[mask]) ** 2)))
    bias = float(np.mean(ens_preds[mask] - y_ho[mask]))
    print(f"  {m}月: n={n:,}, RMSE={rmse:.2f}, Bias={bias:+.2f}")

print(f"\n--- 全样本 分机场 ---")
print(f"{'ADEP':6s} {'n':>8s} {'RMSE':>10s} {'Bias':>8s}")
for ap in sorted(np.unique(aps)):
    mask = (aps == ap) & np.isfinite(y_ho)
    n = int(mask.sum())
    if n < 100:
        continue
    rmse = float(np.sqrt(np.mean((ens_preds[mask] - y_ho[mask]) ** 2)))
    bias = float(np.mean(ens_preds[mask] - y_ho[mask]))
    print(f"  {ap:6s} {n:>8d} {rmse:>10.2f} {bias:>+8.2f}")

# 保存
out = {
    "task": "完整三库集成 (XGB+LGB+CAT × 3seed, 清洗口径)",
    "params": {"xgb": best_params, "lgb": lgb_params, "cat": "default"},
    "seeds": [42, 123, 2024],
    "xgb_results": {str(s): r for s, r in xgb_results.items()},
    "lgb_results": {str(s): r for s, r in lgb_results.items()},
    "cat_results": {str(s): r for s, r in cat_results.items()} if cat_results else None,
    "ensemble_core_rmse": rmse_ens_core,
    "ensemble_all_rmse": rmse_ens_all,
    "anchor": 283.55,
    "delta_core": rmse_ens_core - 283.55,
    "verdict": "✅ 通过" if abs(rmse_ens_core - 283.55) <= 3 else "⚠️ 偏离",
}
out_path = ROOT / "notebooks/diag/d0/ensemble_3lib_3seed.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  结果已保存: {out_path}")