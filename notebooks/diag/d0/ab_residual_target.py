"""
Residual Target A/B (LGB, holdout 2025-01+07, train 02~12)
=========================================================
A 版 (基线): y = TAXITIME, predict y.
B 版 (残差): y = TAXITIME - t_unimp_r7, predict residual; final = t_unimp_r7 + pred.
两者用完全一致的 71 特征、相同 LGB 超参、相同 3 seeds (42/123/2024).

评估: 核心RMSE / 全样本RMSE / 长滑行段(>1800) RMSE / 短滑行段(<=1800) RMSE / 机场 RMSE+bias / seed 抖动
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
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
                  "airport_hour", "AIRCRAFT_TYPE_mvt",
                  "AIRCRAFT_OPERATOR_flt"]
SEEDS = [42, 123, 2024]
LGB_PARAMS_BASE = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}

print("=" * 70); print("[1] load + build_features")
print("=" * 70); t0 = time.time()
ho_dfs = [load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet", columns=TRAIN_COLS)
          for m in (1, 7)]
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in train_months:
    fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
          if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[(df_tr["PHASE_mvt"]=="DEP")&(df_tr["TAXITIME_SEC_mvt"]>0)&
                  (df_tr["TAXITIME_SEC_mvt"]<10800)].copy().reset_index(drop=True)
lut = L.build_t_unimp_lut(df_tr_dep)

def build_features(df_raw, ref_dep_for_d, lut, label=""):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(df_basic.copy(),
        include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    d_dict = D.compute_drift_features(df_full, ref_dep=ref_dep_for_d, label=label)
    for k, v in d_dict.items(): df_full[k] = v
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

print("  ... holdout build"); t1 = time.time()
df_ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
t_unimp_ho = df_ho_dep_full["t_unimp_r7"].values.astype(np.float32)
print(f"  holdout DEP {len(df_ho_dep_full):,}, build_features {time.time()-t1:.0f}s")

print("  ... train build"); t1 = time.time()
df_tr_full = build_features(df_tr_raw, df_tr, lut, label="tr")
df_tr_dep_full = df_tr_full[df_tr_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
df_tr_dep_full = df_tr_dep_full[
    (df_tr_dep_full["TAXITIME_SEC_mvt"]>0)&
    (df_tr_dep_full["TAXITIME_SEC_mvt"]<10800)].reset_index(drop=True)
y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
t_unimp_tr = df_tr_dep_full["t_unimp_r7"].values.astype(np.float32)
print(f"  train DEP {len(df_tr_dep_full):,}, build_features {time.time()-t1:.0f}s")

# 类别编码映射
cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())

def encode_X(df):
    X = df[FEATS71].copy()
    for col in HIGH_CARD_CATS:
        m = cat_map.get(col, {})
        X[col] = X[col].astype(str).map(m).fillna(-1).astype("int32")
    for col in LOW_CARD_CATS:
        X[col] = X[col].astype("category")
    return X

X_ho = encode_X(df_ho_dep_full)
X_tr = encode_X(df_tr_dep_full)

# ============================================================
# [2] A 版 (基线)
# ============================================================
print()
print("=" * 70); print("[2] A 版 (基线) train y=TAXITIME")
print("=" * 70); t2 = time.time()

A_ho_preds = []
A_tr_preds_per_seed = {}
for seed in SEEDS:
    params = dict(LGB_PARAMS_BASE, seed=seed)
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=LOW_CARD_CATS)
    booster = lgb.train(params, train_set, num_boost_round=3000)
    p = booster.predict(X_ho).astype(np.float32)
    p = np.clip(p, 30, 10800)
    A_ho_preds.append(p)
    A_tr_preds_per_seed[seed] = p
    rmse_full = float(np.sqrt(np.mean((p - y_ho)**2)))
    print(f"  seed={seed}: full RMSE={rmse_full:.2f}")
A_pred = np.mean(np.stack(A_ho_preds), axis=0)
A_std = float(np.std([float(np.sqrt(np.mean((p - y_ho)**2))) for p in A_ho_preds]))
print(f"  A 版训练 + 推理完成 {time.time()-t2:.0f}s | 3seed full RMSE std={A_std:.2f}")

# ============================================================
# [3] B 版 (残差)
# ============================================================
print()
print("=" * 70); print("[3] B 版 (残差) train y=TAXITIME-t_unimp_r7")
print("=" * 70); t2 = time.time()

resid_tr = (y_tr - t_unimp_tr).astype(np.float32)
print(f"  训练残差样本数: {len(resid_tr):,}")
print(f"  训练残差分布: mean={resid_tr.mean():+.2f}, std={resid_tr.std():.2f}, min={resid_tr.min():.2f}, max={resid_tr.max():.2f}")
print(f"  负残差行: {(resid_tr<0).sum():,} ({(resid_tr<0).mean()*100:.1f}%)")

B_resid_ho_preds = []
B_pred_per_seed = {}
for seed in SEEDS:
    params = dict(LGB_PARAMS_BASE, seed=seed)
    train_set = lgb.Dataset(X_tr, label=resid_tr, categorical_feature=LOW_CARD_CATS)
    booster = lgb.train(params, train_set, num_boost_round=3000)
    p_resid = booster.predict(X_ho).astype(np.float32)
    # final = t_unimp + pred_resid; clip 同 A 版物理区间 (30, 10800)
    p = np.clip(t_unimp_ho + p_resid, 30, 10800)
    B_resid_ho_preds.append(p)
    B_pred_per_seed[seed] = p
    rmse_full = float(np.sqrt(np.mean((p - y_ho)**2)))
    print(f"  seed={seed}: full RMSE={rmse_full:.2f}")
B_pred = np.mean(np.stack(B_resid_ho_preds), axis=0)
B_std = float(np.std([float(np.sqrt(np.mean((p - y_ho)**2))) for p in B_resid_ho_preds]))
print(f"  B 版训练 + 推理完成 {time.time()-t2:.0f}s | 3seed full RMSE std={B_std:.2f}")

# ============================================================
# [4] 评估指标
# ============================================================
print()
print("=" * 70); print("[4] 评估 (完整 holdout, 含 >10800)")
print("=" * 70)

def eval_(pred, label=""):
    m_full = (y_ho > 0)
    m_core = (y_ho > 0) & (y_ho <= 10800)
    m_short = m_core & (y_ho <= 1800)
    m_long = m_core & (y_ho > 1800)
    m_gt10k = m_full & (y_ho > 10800)

    rmse_full = float(np.sqrt(np.mean((pred[m_full] - y_ho[m_full])**2)))
    rmse_core = float(np.sqrt(np.mean((pred[m_core] - y_ho[m_core])**2)))
    rmse_short = float(np.sqrt(np.mean((pred[m_short] - y_ho[m_short])**2)))
    rmse_long = float(np.sqrt(np.mean((pred[m_long] - y_ho[m_long])**2))) if m_long.sum() > 0 else float('nan')
    rmse_gt10k = float(np.sqrt(np.mean((pred[m_gt10k] - y_ho[m_gt10k])**2))) if m_gt10k.sum() > 0 else float('nan')

    bias_full = float(np.mean(pred[m_full] - y_ho[m_full]))
    bias_core = float(np.mean(pred[m_core] - y_ho[m_core]))
    bias_long = float(np.mean(pred[m_long] - y_ho[m_long])) if m_long.sum() > 0 else float('nan')

    out = {
        "n_full": int(m_full.sum()),
        "n_core": int(m_core.sum()),
        "n_short": int(m_short.sum()),
        "n_long": int(m_long.sum()),
        "n_gt10k": int(m_gt10k.sum()),
        "rmse_full": rmse_full,
        "rmse_core": rmse_core,
        "rmse_short": rmse_short,
        "rmse_long": rmse_long,
        "rmse_gt10k": rmse_gt10k,
        "bias_full": bias_full,
        "bias_core": bias_core,
        "bias_long": bias_long,
    }
    print(f"\n  [{label}]")
    print(f"    full (n={out['n_full']:,}):  RMSE={rmse_full:.2f}  bias={bias_full:+.2f}")
    print(f"    core (n={out['n_core']:,}):  RMSE={rmse_core:.2f}  bias={bias_core:+.2f}")
    print(f"    short (≤1800, n={out['n_short']:,}):  RMSE={rmse_short:.2f}")
    print(f"    ★ long (>1800, n={out['n_long']:,}):  RMSE={rmse_long:.2f}  bias={bias_long:+.2f}")
    print(f"    >10800 (n={out['n_gt10k']:,}): RMSE={rmse_gt10k:.2f}")
    return out

A_eval = eval_(A_pred, "A 版 基线")
B_eval = eval_(B_pred, "B 版 残差+t_unimp")

# 3seed individual rmse 抖动
print()
print(f"  seed 抖动 (全样本 RMSE 3 seed 之间标准差):")
print(f"    A: {A_std:.2f}")
print(f"    B: {B_std:.2f}")

# 分机场评估
print()
print("=" * 70); print("[5] 分机场: A vs B (重点 EGLL/LIRF/LFPG/EHAM)")
print("=" * 70)
aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
FOCUS_AP = ["EGLL", "LIRF", "LFPG", "EHAM"]

def per_ap(pred, label):
    print(f"\n  [{label}]")
    print(f"    {'ADEP':6s} {'n':>6s} {'RMSE_full':>9s} {'RMSE_core':>10s} {'RMSE_long':>10s} {'RMSE_short':>11s} {'bias_core':>10s}")
    out = {}
    for ap in sorted(np.unique(aps)):
        m_ap = (aps == ap) & (y_ho > 0)
        m_core = m_ap & (y_ho <= 10800)
        m_long = m_core & (y_ho > 1800)
        m_short = m_core & (y_ho <= 1800)
        n = int(m_ap.sum())
        if n < 100: continue
        r_full = float(np.sqrt(np.mean((pred[m_ap] - y_ho[m_ap])**2)))
        r_core = float(np.sqrt(np.mean((pred[m_core] - y_ho[m_core])**2)))
        r_long = float(np.sqrt(np.mean((pred[m_long] - y_ho[m_long])**2))) if m_long.sum() > 0 else float('nan')
        r_short = float(np.sqrt(np.mean((pred[m_short] - y_ho[m_short])**2)))
        b_core = float(np.mean(pred[m_core] - y_ho[m_core]))
        focus = "★" if ap in FOCUS_AP else " "
        print(f"    {focus}{ap:5s} {n:>6,d} {r_full:>9.2f} {r_core:>10.2f} {r_long:>10.2f} {r_short:>11.2f} {b_core:>+10.2f}")
        out[ap] = {
            "n": n,
            "rmse_full": r_full,
            "rmse_core": r_core,
            "rmse_long": r_long,
            "rmse_short": r_short,
            "bias_core": b_core,
        }
    return out

A_per = per_ap(A_pred, "A 版")
B_per = per_ap(B_pred, "B 版")

# 集中比较
print()
print("=" * 70); print("[6] 重点机场 A vs B 对比")
print("=" * 70)
print(f"  {'ADEP':6s} {'metric':10s} {'A':>10s} {'B':>10s} {'Δ':>10s}  {'verdict':>8s}")
for ap in FOCUS_AP:
    for metric in ["rmse_core", "rmse_long", "rmse_short", "bias_core"]:
        a = A_per.get(ap, {}).get(metric)
        b = B_per.get(ap, {}).get(metric)
        if a is None or b is None: continue
        if metric == "bias_core":
            # bias 看绝对值减小的方向
            delta = abs(b) - abs(a)
            verdict = "✓" if (abs(b) < abs(a)) else " "
        else:
            delta = b - a
            verdict = "✓" if (b < a) else " "
        print(f"  {ap:6s} {metric:10s} {a:>10.2f} {b:>10.2f} {delta:>+10.2f}  {verdict:>8s}")

# ============================================================
# [7] 总判定
# ============================================================
print()
print("=" * 70); print("[7] 总判定 (B vs A)")
print("=" * 70)
delta_core = B_eval['rmse_core'] - A_eval['rmse_core']
delta_long = B_eval['rmse_long'] - A_eval['rmse_long']
delta_short = B_eval['rmse_short'] - A_eval['rmse_short']

# "有效": B 比 A 在核心 OR 长滑行段低 > seed抖动
seed_jitter = max(A_std, B_std)
effective = (delta_core < -seed_jitter) or (delta_long < -seed_jitter)
print(f"    A core RMSE: {A_eval['rmse_core']:.2f}")
print(f"    B core RMSE: {B_eval['rmse_core']:.2f}  Δ={delta_core:+.2f}")
print(f"    A long RMSE: {A_eval['rmse_long']:.2f}")
print(f"    B long RMSE: {B_eval['rmse_long']:.2f}  Δ={delta_long:+.2f}")
print(f"    A short RMSE: {A_eval['rmse_short']:.2f}")
print(f"    B short RMSE: {B_eval['rmse_short']:.2f}  Δ={delta_short:+.2f}")
print(f"    seed 抖动 (max of A/B 全样本 std): {seed_jitter:.2f}")
print()
print(f"  判据: B 版核心 OR 长滑行段 < A 版, 且 |Δ| > seed 抖动 ±{seed_jitter:.1f}s")
if effective:
    print(f"  ✅ 结论: B 版 残差目标 有效")
    if delta_long < delta_core:
        print(f"     → 长滑行段改善更明显 (Δ {delta_long:+.2f})")
    else:
        print(f"     → 核心改善更明显 (Δ {delta_core:+.2f})")
else:
    print(f"  ❌ 结论: B 版 残差目标 无效 (差异 ≤ seed 抖动)")

# 落盘
out = {
    "A_eval": A_eval, "B_eval": B_eval,
    "A_per_ap": A_per, "B_per_ap": B_per,
    "seed_jitter_max": seed_jitter,
    "delta": {
        "core": delta_core, "long": delta_long, "short": delta_short,
    },
    "effective": bool(effective),
    "seeds_used": SEEDS,
    "params": LGB_PARAMS_BASE,
    "n_features": len(FEATS71),
    "n_round": 3000,
    "warning": "训练集剔 TAXITIME>10800, 评估全集 (含 >10800) 与核心 (剔 >10800) 两口径都报",
}
(D0 / "ab_residual_target.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print()
print(f"  → 写入: {D0 / 'ab_residual_target.json'}")
