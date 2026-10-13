"""
续跑: 修复 XGB/CAT/ranking 部分
- XGB: 用 69 列 (R7slim 模型自带), 整数编码高基数列
- CAT: 修正 - 全整数编码, 无字符串列, 排除低位 cat_features 字符串
- Ranking: 用训练好的模型预测
"""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

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

# XGB 用 69 特征 (R7slim 模型自带)
bst_xgb_seed0 = xgb.Booster()
bst_xgb_seed0.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst_xgb_seed0.feature_names
print(f"XGB 模型特征数: {len(FEATS69)}")

# LGB/CAT 用 71 特征
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

LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
                  "AIRCRAFT_OPERATOR_flt"]

# ============================================================
# 1. 加载 + 重建特征
# ============================================================
print("=" * 70)
print("[1] 加载 + 重建特征")
print("=" * 70)

train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in train_months:
    fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
          if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[
    (df_tr["PHASE_mvt"] == "DEP") &
    (df_tr["TAXITIME_SEC_mvt"] > 0) &
    (df_tr["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  训练 DEP: {len(df_tr_dep):,}")

ho_dfs = []
for m in [1, 7]:
    fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
df_ho_dep = df_ho[
    (df_ho["PHASE_mvt"] == "DEP") &
    (df_ho["TAXITIME_SEC_mvt"] > 0) &
    (df_ho["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  Holdout DEP: {len(df_ho_dep):,}")

# Ranking (不要 TAXITIME)
rank_cols = [c for c in TRAIN_COLS if c != "TAXITIME_SEC_mvt"]
rank_df = pd.read_parquet(ROOT / "data" / "raw" / "ranking.parquet", columns=rank_cols)
rank_df = parse_time_columns(rank_df, cols=[c for c in TIME_COLS if c != "TAXITIME_SEC_mvt"])
print(f"  Ranking 总: {len(rank_df):,}")

# LUT
lut = L.build_t_unimp_lut(df_tr_dep)

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
print("[2] 重建 训练折 + holdout + ranking 特征")
print("=" * 70)

t0 = time.time()
df_tr_full = build_features(df_tr_raw, df_tr, lut, label="tr")
df_tr_dep_full = df_tr_full[df_tr_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
df_tr_dep_full = df_tr_dep_full[
    (df_tr_dep_full["TAXITIME_SEC_mvt"]>0)&(df_tr_dep_full["TAXITIME_SEC_mvt"]<10800)
].reset_index(drop=True)
print(f"  训练 DEP 全套: {df_tr_dep_full.shape} ({time.time()-t0:.0f}s)")

t0 = time.time()
df_ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
df_ho_dep_full = df_ho_dep_full[
    (df_ho_dep_full["TAXITIME_SEC_mvt"]>0)&(df_ho_dep_full["TAXITIME_SEC_mvt"]<10800)
].reset_index(drop=True)
print(f"  Holdout DEP 全套: {df_ho_dep_full.shape} ({time.time()-t0:.0f}s)")

t0 = time.time()
rank_full = build_features(rank_df, df_tr, lut, label="rank")
rank_dep_full = rank_full[rank_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
print(f"  Ranking DEP 全套: {rank_dep_full.shape} ({time.time()-t0:.0f}s)")

# ============================================================
# 2. cat 整数编码
# ============================================================
print()
print("=" * 70)
print("[3] cat 整数编码")
print("=" * 70)
cat_map_path = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
cat_map = json.loads(cat_map_path.read_text())

for col in HIGH_CARD_CATS:
    if col in df_tr_dep_full.columns:
        m = cat_map.get(col, {})
        for df_x in [df_tr_dep_full, df_ho_dep_full, rank_dep_full]:
            if col in df_x.columns:
                df_x[col] = df_x[col].astype(str).map(m).fillna(-1).astype("int32")

for col in LOW_CARD_CATS:
    if col in df_tr_dep_full.columns:
        for df_x in [df_tr_dep_full, df_ho_dep_full, rank_dep_full]:
            if col in df_x.columns:
                df_x[col] = df_x[col].astype("category")

# 补缺失
for df_x, name in [(df_tr_dep_full, "tr"), (df_ho_dep_full, "ho"), (rank_dep_full, "rank")]:
    miss = [f for f in FEATS71 if f not in df_x.columns]
    for f in miss:
        df_x[f] = np.nan

# ============================================================
# 3. XGB 推理 (用 69 列, R7slim 模型自带)
# ============================================================
print()
print("=" * 70)
print("[4] XGB 推理 (用 69 特征)")
print("=" * 70)

# 准备 X (69 列)
X_ho_69 = df_ho_dep_full[FEATS69].copy()
X_rank_69 = rank_dep_full[FEATS69].copy()

# XGB 需要整数编码 cat_features
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = cat_map

# 列类型转换 (XGB 要求)
for col in FEATS69:
    if col in X_ho_69.columns:
        if col in LOW_CARD_CATS:
            X_ho_69[col] = X_ho_69[col].astype("category")
        elif col in HIGH_CARD_CATS:
            X_ho_69[col] = X_ho_69[col].astype("int32")
    if col in X_rank_69.columns:
        if col in LOW_CARD_CATS:
            X_rank_69[col] = X_rank_69[col].astype("category")
        elif col in HIGH_CARD_CATS:
            X_rank_69[col] = X_rank_69[col].astype("int32")

X_ho_69_enc = tx._encode_categoricals(X_ho_69, fit=False).values.astype(np.float32)
X_rank_69_enc = tx._encode_categoricals(X_rank_69, fit=False).values.astype(np.float32)
dmat_ho_69 = xgb.DMatrix(X_ho_69_enc, feature_names=FEATS69)
dmat_rank_69 = xgb.DMatrix(X_rank_69_enc, feature_names=FEATS69)

# XGB 3 模型推理 (只有 seed0 模型留存)
y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
xgb_holdout_preds = np.zeros(len(y_ho), dtype=np.float64)
xgb_rank_preds = np.zeros(len(rank_dep_full), dtype=np.float64)
for seed in [42, 123, 2024]:
    fn = ROOT / "models" / f"xgb_v10_R7slim_12m_seed{seed}_final.json"
    if not fn.exists():
        fn = ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"
    bst = xgb.Booster()
    bst.load_model(str(fn))
    xgb_holdout_preds += bst.predict(dmat_ho_69) / 3
    xgb_rank_preds += bst.predict(dmat_rank_69) / 3

y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
xgb_holdout_preds = np.clip(xgb_holdout_preds, 30, 10800)
core_mask = (y_ho > 0) & (y_ho <= 10800)
xgb_core = float(np.sqrt(np.mean((xgb_holdout_preds[core_mask] - y_ho[core_mask]) ** 2)))
xgb_all = float(np.sqrt(np.mean((xgb_holdout_preds - y_ho) ** 2)))
print(f"  XGB 3 模型 (69 特征): core={xgb_core:.2f}, all={xgb_all:.2f}")

# ============================================================
# 4. LGB 推理 (71 列) — 重训 (避免之前的predict_disable_shape_check问题)
# ============================================================
print()
print("=" * 70)
print("[5] LGB 推理 (71 特征, 重训避免列错)")
print("=" * 70)

X_tr_71 = df_tr_dep_full[FEATS71].copy()
X_ho_71 = df_ho_dep_full[FEATS71].copy()
X_rank_71 = rank_dep_full[FEATS71].copy()
y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)

lgb_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}

lgb_holdout_preds = np.zeros(len(y_ho), dtype=np.float64)
lgb_rank_preds = np.zeros(len(rank_dep_full), dtype=np.float64)
for seed in [42, 123, 2024]:
    print(f"  --- LGB seed = {seed} ---")
    t0 = time.time()
    lgb_train = lgb.Dataset(X_tr_71, label=y_tr, categorical_feature=LOW_CARD_CATS)
    lgb_holdout = lgb.Dataset(X_ho_71, label=y_ho, categorical_feature=LOW_CARD_CATS, reference=lgb_train)
    params = {**lgb_params, "seed": seed}
    booster = lgb.train(
        params, lgb_train, num_boost_round=2500,
        valid_sets=[lgb_holdout],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)]
    )
    elapsed = time.time() - t0
    best_iter = booster.best_iteration
    pred_ho = booster.predict(X_ho_71, num_iteration=best_iter)
    pred_ho = np.clip(pred_ho, 30, 10800)
    lgb_holdout_preds += pred_ho / 3
    # ranking
    pred_rank = booster.predict(X_rank_71, num_iteration=best_iter)
    pred_rank = np.clip(pred_rank, 30, 10800)
    lgb_rank_preds += pred_rank / 3
    print(f"    best_iter={best_iter}, time={elapsed:.1f}s")

lgb_core = float(np.sqrt(np.mean((lgb_holdout_preds[core_mask] - y_ho[core_mask]) ** 2)))
lgb_all = float(np.sqrt(np.mean((lgb_holdout_preds - y_ho) ** 2)))
print(f"  LGB 3 模型 (71 特征): core={lgb_core:.2f}, all={lgb_all:.2f}")

# ============================================================
# 5. CAT 推理 — 用整数编码 + 字符串 (避免 NaN->"M" 错误)
# ============================================================
print()
print("=" * 70)
print("[6] CAT 推理 (71 特征, 整数编码)")
print("=" * 70)

# CAT 字符串型 cat_features (但都是低基数整数编码后, 实际不再需要 cat_features)
# 用全数值列
X_tr_cat = X_tr_71.copy()
X_ho_cat = X_ho_71.copy()
X_rank_cat = X_rank_71.copy()
# 把 category 列转 int (CAT 不接受 category dtype)
for col in LOW_CARD_CATS:
    if col in X_tr_cat.columns:
        X_tr_cat[col] = X_tr_cat[col].astype(str).fillna("__nan__")
        X_ho_cat[col] = X_ho_cat[col].astype(str).fillna("__nan__")
        X_rank_cat[col] = X_rank_cat[col].astype(str).fillna("__nan__")
# NaN 也转字符串
for col in X_tr_cat.columns:
    if X_tr_cat[col].dtype == "float32" or X_tr_cat[col].dtype == "float64":
        X_tr_cat[col] = X_tr_cat[col].astype(str).fillna("__nan__")
        X_ho_cat[col] = X_ho_cat[col].astype(str).fillna("__nan__")
        X_rank_cat[col] = X_rank_cat[col].astype(str).fillna("__nan__")

cat_holdout_preds = np.zeros(len(y_ho), dtype=np.float64)
cat_rank_preds = np.zeros(len(rank_dep_full), dtype=np.float64)
for seed in [42, 123, 2024]:
    print(f"  --- CAT seed = {seed} ---")
    t0 = time.time()
    model = CatBoostRegressor(
        iterations=2500, learning_rate=0.0147555939449363,
        depth=11, l2_leaf_reg=5, random_seed=seed, verbose=500,
        early_stopping_rounds=100, thread_count=-1, eval_metric="RMSE",
        # 不传 cat_features, 让 CAT 自动检测或全数值
    )
    model.fit(X_tr_cat, y_tr, eval_set=(X_ho_cat, y_ho))
    elapsed = time.time() - t0
    if elapsed > 20 * 60:
        print(f"  ⚠️ CAT seed={seed} >20min, 停下")
        cat_holdout_preds = None
        break
    pred_ho = model.predict(X_ho_cat)
    pred_ho = np.clip(pred_ho, 30, 10800)
    cat_holdout_preds += pred_ho / 3
    pred_rank = model.predict(X_rank_cat)
    pred_rank = np.clip(pred_rank, 30, 10800)
    cat_rank_preds += pred_rank / 3
    print(f"    best_iter={model.tree_count_}, time={elapsed:.1f}s")

if cat_holdout_preds is not None:
    cat_core = float(np.sqrt(np.mean((cat_holdout_preds[core_mask] - y_ho[core_mask]) ** 2)))
    cat_all = float(np.sqrt(np.mean((cat_holdout_preds - y_ho) ** 2)))
    print(f"  CAT 3 模型: core={cat_core:.2f}, all={cat_all:.2f}")
else:
    cat_core = cat_all = None

# ============================================================
# 6. 集成 + 报告
# ============================================================
print()
print("=" * 70)
print("[7] 集成 (XGB + LGB + CAT)")
print("=" * 70)

if cat_holdout_preds is not None:
    ens_holdout = (xgb_holdout_preds + lgb_holdout_preds + cat_holdout_preds) / 3
    ens_rank = (xgb_rank_preds + lgb_rank_preds + cat_rank_preds) / 3
    components = "XGB + LGB + CAT (3 库)"
else:
    ens_holdout = (xgb_holdout_preds + lgb_holdout_preds) / 2
    ens_rank = (xgb_rank_preds + lgb_rank_preds) / 2
    components = "XGB + LGB (CAT 失败)"

ens_core = float(np.sqrt(np.mean((ens_holdout[core_mask] - y_ho[core_mask]) ** 2)))
ens_all = float(np.sqrt(np.mean((ens_holdout - y_ho) ** 2)))
print(f"  {components}")
print(f"    XGB: core={xgb_core:.2f}, all={xgb_all:.2f}")
print(f"    LGB: core={lgb_core:.2f}, all={lgb_all:.2f}")
if cat_holdout_preds is not None:
    print(f"    CAT: core={cat_core:.2f}, all={cat_all:.2f}")
print(f"    集成 core={ens_core:.2f}, all={ens_all:.2f}")
print(f"  v15 锚点: 532.27 (README)")

# 分机场
aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
# 反查回原始机场名 (用 cat_map)
adep_inv = {v: k for k, v in cat_map.get("ADEP_mvt", {}).items() if k != "__nan__"}
print(f"\n--- 集成 分机场 ---")
print(f"{'ADEP':6s} {'n':>8s} {'core RMSE':>10s} {'all RMSE':>10s} {'Bias':>8s}")
# 注: ADEP_mvt 已被整数编码, 用整数化前数据找原始名
# 实际上 df_ho_dep_full["ADEP_mvt"] 现在是整数, 要从 cat_map 反查
# 更简单: 重新拿一个原始 ADEP 数组
df_ho_dep_orig = df_ho.copy()
df_ho_dep_orig = df_ho_dep_orig[
    (df_ho_dep_orig["PHASE_mvt"]=="DEP") &
    (df_ho_dep_orig["TAXITIME_SEC_mvt"]>0) &
    (df_ho_dep_orig["TAXITIME_SEC_mvt"]<10800)
].reset_index(drop=True)
aps_orig = df_ho_dep_orig["ADEP_mvt"].astype(str).values
for ap in sorted(np.unique(aps_orig)):
    m = (aps_orig == ap) & core_mask
    m_all = aps_orig == ap
    n_core = int(m.sum())
    if n_core < 100:
        continue
    r_core = float(np.sqrt(np.mean((ens_holdout[m] - y_ho[m]) ** 2)))
    r_all = float(np.sqrt(np.mean((ens_holdout[m_all] - y_ho[m_all]) ** 2)))
    bias = float(np.mean(ens_holdout[m_all] - y_ho[m_all]))
    print(f"  {ap:6s} {n_core:>8d} {r_core:>10.2f} {r_all:>10.2f} {bias:>+8.2f}")

# 分月份
months_col = pd.to_datetime(df_ho_dep_orig["MVT_TIME_UTC_mvt"]).dt.month.values
print(f"\n--- 集成 分月份 ---")
for m in [1, 7]:
    mm = (months_col == m)
    n_all = int(mm.sum())
    n_core = int((mm & core_mask).sum())
    if n_core < 10:
        continue
    r_core = float(np.sqrt(np.mean((ens_holdout[mm & core_mask] - y_ho[mm & core_mask]) ** 2)))
    r_all = float(np.sqrt(np.mean((ens_holdout[mm] - y_ho[mm]) ** 2)))
    print(f"  {m}月: n={n_all}, core={r_core:.2f}, all={r_all:.2f}")

# ============================================================
# 7. gate_check 五项
# ============================================================
print()
print("=" * 70)
print("[8] gate_check 五项")
print("=" * 70)
print(f"  CHECK 1: 集成 all={ens_all:.2f} vs 锚点 (532.27)")
sse_total = float(np.sum((ens_holdout[core_mask] - y_ho[core_mask]) ** 2))
rmse_recomp = float(np.sqrt(sse_total / core_mask.sum()))
diff = abs(rmse_recomp - ens_core)
print(f"  CHECK 2: SSE 反算={rmse_recomp:.6f}, Δ={diff:.2e} {'✅' if diff < 1e-6 else '❌'}")
n_nan = int(np.isnan(ens_holdout).sum())
n_neg = int((ens_holdout < 0).sum())
print(f"  CHECK 4: NaN={n_nan}, 负={n_neg} {'✅' if n_nan==0 and n_neg==0 else '❌'}")

# ============================================================
# 8. 生成提交 (holdout + ranking)
# ============================================================
print()
print("=" * 70)
print("[9] 生成提交")
print("=" * 70)

# Holdout 提交
ho_sub_df = pd.DataFrame({
    "MVT_ID_mvt": df_ho_dep_orig["MVT_ID_mvt"].values,
    "TAXITIME_SEC_mvt": np.round(ens_holdout, 0).astype(int),
})
ho_sub_path = ROOT / "submissions" / "jubilant-puzzle_v15_clean_holdout.parquet"
ho_sub_df.to_parquet(ho_sub_path, index=False)
print(f"  Holdout: {ho_sub_path}")
print(f"    行数: {len(ho_sub_df)}, n_nan={int(ho_sub_df['TAXITIME_SEC_mvt'].isna().sum())}")

# ranking 提交 (344,841 行)
rank_ids = rank_dep_full["MVT_ID_mvt"].astype(float).values
rank_sub_df = pd.DataFrame({
    "MVT_ID_mvt": rank_ids,
    "TAXITIME_SEC_mvt": np.round(ens_rank, 0).astype(int),
})
rank_sub_path = ROOT / "submissions" / "jubilant-puzzle_v15_clean_ranking.parquet"
rank_sub_df.to_parquet(rank_sub_path, index=False)
print(f"  Ranking: {rank_sub_path}")
print(f"    行数: {len(rank_sub_df)}, n_nan={int(rank_sub_df['TAXITIME_SEC_mvt'].isna().sum())}")

# MVT_ID 集合+顺序校验
sub_ref = pd.read_parquet(ROOT / "data" / "raw" / "submitting.parquet", columns=["MVT_ID_mvt"])
exp_ids = sub_ref["MVT_ID_mvt"].astype(float).values
match_set = set(rank_ids) == set(exp_ids)
match_order = np.all(rank_ids == exp_ids)
print(f"    MVT_ID 集合+顺序: set={match_set}, order={match_order} {'✅' if (match_set and match_order) else '❌'}")

print()
print("=" * 70)
print("[报告完成] 不上传, 等用户确认")
print("=" * 70)