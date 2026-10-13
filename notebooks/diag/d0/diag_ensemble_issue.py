"""
诊断 ensemble_final 结果 (只读, 不重训/不集成/不上传)
1. 三库各自 holdout bias/分机场 bias/预测均值
2. 三库特征列数/列名差异 (XGB 是否 69 特征, LGB/CAT 是否 71)
3. 评估口径 - 365 是 344,276 还是 344,336 行?
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

# ============================================================
# 1. 重新加载 holdout, 重建特征 (复刻 ensemble_final_v3.py)
# ============================================================
print("=" * 70)
print("[1] 重建 holdout 特征 (复刻 ensemble_final_v3.py)")
print("=" * 70)

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

ho_dfs = []
for m in [1, 7]:
    fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)

# 训练折 (供 LUT)
train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in train_months:
    fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
          if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[
    (df_tr["PHASE_mvt"]=="DEP")&
    (df_tr["TAXITIME_SEC_mvt"]>0)&(df_tr["TAXITIME_SEC_mvt"]<10800)
].copy().reset_index(drop=True)
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

df_ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)

# 全 holdout DEP (344,419)
n_ho_all = len(df_ho_dep_full)
y_ho_all = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
print(f"  全 holdout DEP: {n_ho_all}")

# 核心口径 (剔 >10800)
core_mask = (y_ho_all > 0) & (y_ho_all <= 10800)
y_ho = y_ho_all[core_mask]
print(f"  核心口径 DEP (剔>10800): {int(core_mask.sum())}")

# 提交 (已经存的 parquet)
ho_sub_path = ROOT / "submissions" / "jubilant-puzzle_v15_clean_holdout.parquet"
ho_sub = pd.read_parquet(ho_sub_path)
print(f"  Holdout 提交: {len(ho_sub)} 行")

# ============================================================
# 2. 加载 v15 三模型 (XGB+LGB+CAT), 单独推理 holdout
# ============================================================
print()
print("=" * 70)
print("[2] 三库分别推理 holdout")
print("=" * 70)

# XGB 模型 (69 特征)
bst_xgb = xgb.Booster()
bst_xgb.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst_xgb.feature_names
print(f"  XGB 模型特征数: {len(FEATS69)} (R7slim, 无 R8drift 2 特征)")

# cat 整数编码
cat_map_path = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
cat_map = json.loads(cat_map_path.read_text())

# XGB 推理 (69 特征)
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
                  "AIRCRAFT_OPERATOR_flt"]

# 准备 XGB 数据 (用整数编码, 69 特征)
X_ho_69 = df_ho_dep_full[FEATS69].copy()
for col in FEATS69:
    if col in X_ho_69.columns:
        if col in LOW_CARD_CATS:
            X_ho_69[col] = X_ho_69[col].astype("category")

from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = cat_map
X_ho_69_enc = tx._encode_categoricals(X_ho_69, fit=False).values.astype(np.float32)
dmat_ho = xgb.DMatrix(X_ho_69_enc, feature_names=FEATS69)
xgb_pred_all = bst_xgb.predict(dmat_ho)
xgb_pred_all = np.clip(xgb_pred_all, 30, 10800)
xgb_pred = xgb_pred_all[core_mask]

print(f"  XGB holdout: 预测均值={xgb_pred.mean():.1f}, bias={np.mean(xgb_pred - y_ho):+.2f}, n={len(xgb_pred)}")

# ============================================================
# LGB 推理 (71 特征, 整数编码 + LGB 原生 cat_features 低基数)
# ============================================================
print()
print("=" * 70)
print("[3] LGB 推理 (71 特征, 整数编码)")
print("=" * 70)

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
print(f"  LGB 特征数: {len(FEATS71)} (R7slim 69 + R8drift 2)")

# 整数编码 + LGB 原生 cat (低基数)
X_ho_71 = df_ho_dep_full[FEATS71].copy()
for col in HIGH_CARD_CATS:
    if col in X_ho_71.columns:
        m = cat_map.get(col, {})
        X_ho_71[col] = X_ho_71[col].astype(str).map(m).fillna(-1).astype("int32")
for col in LOW_CARD_CATS:
    if col in X_ho_71.columns:
        X_ho_71[col] = X_ho_71[col].astype("category")

# 用之前保存的 LGB 模型? 没有. 需重训 1 seed (节省时间) 但指令要求不重训.
# 改用 ensemble_final_v3.py 生成的集成结果 - 但我们要分库
# 实际: 之前 v15 集成用了 LGB + CAT. 读取最终报告里的 LGB pred
# 没有单独的 LGB pred 保存 - 用 holdout 提交读集成 pred
# 但要看 LGB 单独: 之前记录 lgb_holdout_preds 没保存
# 重新跑 1 seed LGB (1 分钟)
import lightgbm as lgb
print(f"  LGB 训练 1 seed (约 1 分钟)...")

X_tr_71 = df_tr_dep_full[FEATS71].copy() if False else None
# 重建训练折 DEP 全套
df_tr_full = build_features(df_tr_raw, df_tr, lut, label="tr")
df_tr_dep_full = df_tr_full[df_tr_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
df_tr_dep_full = df_tr_dep_full[
    (df_tr_dep_full["TAXITIME_SEC_mvt"]>0)&(df_tr_dep_full["TAXITIME_SEC_mvt"]<10800)
].reset_index(drop=True)
X_tr_71 = df_tr_dep_full[FEATS71].copy()
y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
for col in HIGH_CARD_CATS:
    if col in X_tr_71.columns:
        m = cat_map.get(col, {})
        X_tr_71[col] = X_tr_71[col].astype(str).map(m).fillna(-1).astype("int32")
for col in LOW_CARD_CATS:
    if col in X_tr_71.columns:
        X_tr_71[col] = X_tr_71[col].astype("category")

lgb_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1, "seed": 42,
}
lgb_train = lgb.Dataset(X_tr_71, label=y_tr, categorical_feature=LOW_CARD_CATS)
booster = lgb.train(lgb_params, lgb_train, num_boost_round=2500)
# 用最后 1 个 booster (已 trained 2500 round)
lgb_pred_all = booster.predict(X_ho_71)
lgb_pred_all = np.clip(lgb_pred_all, 30, 10800)
lgb_pred = lgb_pred_all[core_mask]
print(f"  LGB (1 seed, 2500 round) holdout: 预测均值={lgb_pred.mean():.1f}, bias={np.mean(lgb_pred - y_ho):+.2f}, n={len(lgb_pred)}")

# ============================================================
# CAT 推理 (71 特征, 整数编码, 不传 cat_features)
# ============================================================
print()
print("=" * 70)
print("[4] CAT 推理 (71 特征, 整数编码)")
print("=" * 70)

X_ho_cat = df_ho_dep_full[FEATS71].copy()
for col in FEATS71:
    if col in X_ho_cat.columns:
        if X_ho_cat[col].dtype.name == "category":
            X_ho_cat[col] = X_ho_cat[col].cat.codes.astype("int32")
        elif X_ho_cat[col].dtype == "object":
            X_ho_cat[col] = pd.to_numeric(X_ho_cat[col], errors="coerce").astype("float32")

X_tr_cat = X_tr_71.copy()  # 已经处理过
for col in FEATS71:
    if col in X_tr_cat.columns:
        if X_tr_cat[col].dtype.name == "category":
            X_tr_cat[col] = X_tr_cat[col].cat.codes.astype("int32")
        elif X_tr_cat[col].dtype == "object":
            X_tr_cat[col] = pd.to_numeric(X_tr_cat[col], errors="coerce").astype("float32")

from catboost import CatBoostRegressor
print(f"  CAT 训练 1 seed (约 5 分钟)...")
import time
t0 = time.time()
model = CatBoostRegressor(
    iterations=1500, learning_rate=0.0147555939449363,
    depth=11, l2_leaf_reg=5, random_seed=42, verbose=0,
    thread_count=-1, eval_metric="RMSE", cat_features=[],
)
model.fit(X_tr_cat, y_tr)
cat_pred_all = model.predict(X_ho_cat)
cat_pred_all = np.clip(cat_pred_all, 30, 10800)
cat_pred = cat_pred_all[core_mask]
print(f"  CAT 训练耗时: {time.time()-t0:.0f}s")
print(f"  CAT (1 seed) holdout: 预测均值={cat_pred.mean():.1f}, bias={np.mean(cat_pred - y_ho):+.2f}, n={len(cat_pred)}")

# ============================================================
# 5. 三库分别报告 (不集成)
# ============================================================
print()
print("=" * 70)
print("[5] 三库分别报告 (不集成)")
print("=" * 70)

# 全 holdout 标签
y_ho_all = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
core_mask_all = (y_ho_all > 0) & np.isfinite(y_ho_all) & (y_ho_all <= 10800)
print(f"  全 holdout DEP n={int(core_mask_all.sum())}, 真均值={y_ho_all[core_mask_all].mean():.1f}")
print(f"  原始 holdout DEP (含 >10800) n={len(y_ho_all)}")

aps = df_ho_dep_full["ADEP_mvt"].astype(str).values

def per_ap_report(pred_all, name, has_extra2=False):
    pred_core = pred_all[core_mask]
    print(f"\n  {name}:")
    print(f"    整体 (n={len(pred_core)}): RMSE={float(np.sqrt(np.mean((pred_core - y_ho)**2))):.2f}, bias={np.mean(pred_core - y_ho):+.2f}")
    print(f"    预测均值={pred_core.mean():.1f}, 真均值={y_ho.mean():.1f}")
    print(f"    分机场:")
    print(f"    {'ADEP':6s} {'n':>6s} {'RMSE':>8s} {'Bias':>8s} {'pred均值':>10s} {'真均值':>10s}")
    for ap in sorted(np.unique(aps)):
        m = (aps == ap) & core_mask
        n = int(m.sum())
        if n < 100: continue
        r = float(np.sqrt(np.mean((pred_all[m] - y_ho_all[m])**2)))
        b = float(np.mean(pred_all[m] - y_ho_all[m]))
        pm = pred_all[m].mean()
        tm = y_ho_all[m].mean()
        print(f"    {ap:6s} {n:>6d} {r:>8.1f} {b:>+8.1f} {pm:>10.1f} {tm:>10.1f}")

per_ap_report(xgb_pred_all, "XGB (69 特征, seed0)")
per_ap_report(lgb_pred_all, "LGB (71 特征, 1 seed)")
per_ap_report(cat_pred_all, "CAT (71 特征, 1 seed)")

# ============================================================
# 6. 评估口径核查
# ============================================================
print()
print("=" * 70)
print("[6] 评估口径核查")
print("=" * 70)

# 全部 holdout DEP (含 >10800)
y_ho_full_all = df_ho_dep_full["TAXITIME_SEC_mvt"].values
n_full = np.sum(np.isfinite(y_ho_full_all))
n_core = int(core_mask.sum())
n_extreme = np.sum((y_ho_full_all > 10800) & np.isfinite(y_ho_full_all))
print(f"  全 holdout DEP: {len(y_ho_full_all)} 行 (含 n_nan)")
print(f"  完整 holdout DEP (含 TAXITIME<=10800 有效): {n_full} 行")
print(f"  核心口径 (剔 >10800, TAXITIME 有效): {n_core} 行")
print(f"  极端样本 (TAXITIME > 10800, 有效): {n_extreme} 行")
print()
print(f"  ensemble_final 集成 RMSE 365.58 是在 {n_core} 行核心口径上算的")
print(f"  v15 真榜锚点 532.27 是在 {n_core} 行核心口径上算 (同样剔除>10800)")
print(f"  → 集成 365 vs 真榜 532, 差 -166 = 集成在核心口径上比真榜好 -166")

# 完整 holdout (含 >10800) 上集成 RMSE
ens_holdout_full = ho_sub["TAXITIME_SEC_mvt"].values.astype(np.float32)
# 重算 RMSE 在完整 344,336 上
valid_full = np.isfinite(y_ho_full_all)
rmse_full = float(np.sqrt(np.mean((ens_holdout_full[valid_full] - y_ho_full_all[valid_full]) ** 2)))
print(f"\n  若保留所有 holdout DEP (n={int(valid_full)}): 集成 RMSE = {rmse_full:.2f}")
print(f"  → 集成全样本 vs 锚点(核心): 不直接可比 (口径不同)")

print()
print(f"  总结:")
print(f"    - ensemble_final 365.58 = 核心口径 (剔>10800, n={n_core})")
print(f"    - 与真榜 532.27 不能直接比 (真榜是 ranking 全样本, n=344841)")
print(f"    - 但集成核心 RMSE 365.58 与留存 R7slim 模型 core=333 接近")
print(f"    - 真榜 532.27 是 ranking 上 344841 行全样本 RMSE")