"""
实际队列特征 A/B v3 (纯 numpy batched, 不 Python loop over airports)
=================================================================
5 特征全部在 airport 级别, 不做 runway 细分:
  1. actl_takeoff_ap_15m
  2. actl_takeoff_ap_30m
  3. actl_congestion_ap
  4. actl_takeoff_ap_5m
  5. actl_takeoff_ap_60m
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
NEW_FEATS = [
    "actl_takeoff_ap_5m",
    "actl_takeoff_ap_15m",
    "actl_takeoff_ap_30m",
    "actl_takeoff_ap_60m",
    "actl_congestion_ap",
]

iNaT = np.iinfo(np.int64).min

def _to_ns(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.values.astype("datetime64[ns]").view("i8")
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts.values.astype("datetime64[ns]").view("i8")

def add_actual_queue_ap_only(df_all, label=""):
    """
    纯机场级特征. 不分跑道. 全 numpy batched (同一个 dataframe 一次性向量化).

    关键高效做法:
      - 按 ADEP 分组: 用 np.argsort 把所有 (anchor, MVT, AOBT) 按 airport index 排序
      - 然后 per-airport 用 searchsorted
    """
    t0 = time.time()
    n = len(df_all)
    dep_mask = (df_all["PHASE_mvt"].values == "DEP")

    eobt_ns = _to_ns(df_all["EOBT_1_flt"])
    sched_ns = _to_ns(df_all["SCHED_TIME_UTC_mvt"])
    mvt_ns = _to_ns(df_all["MVT_TIME_UTC_mvt"])
    aobt_ns = _to_ns(df_all["AOBT_3_flt"])

    # anchor = EOBT 优先, 退 SCHED
    anchor = np.where(eobt_ns != iNaT, eobt_ns, sched_ns).astype("int64")
    aep = df_all["ADEP_mvt"].astype(str).values

    out = {c: np.zeros(n, dtype=np.float32) for c in NEW_FEATS}

    # 用 np.argsort 给 aep 排序, 然后 per-airport 用切片切片, 对排序后的 mvt/aobt/anchor
    aep_order = np.argsort(aep, kind="mergesort")
    aep_sorted = aep[aep_order]
    # 找各机场的起止 index
    unique_ap, starts = np.unique(aep_sorted, return_index=True)
    ends = np.concatenate([starts[1:], [len(aep_sorted)]])

    # 把 anchor/mvt/aobt 按 aep 排序后, 每个机场连续
    anchor_sorted = anchor[aep_order]
    mvt_sorted = mvt_ns[aep_order]
    aobt_sorted = aobt_ns[aep_order]

    # 一次循环每个机场
    n_ap = len(unique_ap)
    for i in range(n_ap):
        s, e = starts[i], ends[i]
        mvt_g = mvt_g_valid = mvt_sorted[s:e]
        aobt_g = aobt_sorted[s:e]
        anchor_g = anchor_sorted[s:e]
        # 排序 (MVT / AOBT)
        valid_m = mvt_g != iNaT
        valid_a = aobt_g != iNaT
        mvt_valid = np.sort(mvt_g[valid_m])
        aobt_valid = np.sort(aobt_g[valid_a])

        # query: anchor_g 数组
        ag = anchor_g
        # takeoff ap: 过去 N 分钟 strict < t
        for win_min, key in [
            (5, "actl_takeoff_ap_5m"),
            (15, "actl_takeoff_ap_15m"),
            (30, "actl_takeoff_ap_30m"),
            (60, "actl_takeoff_ap_60m"),
        ]:
            if len(mvt_valid) == 0:
                out[key][aep_order[s:e]] = 0.0
                continue
            w_ns = win_min * 60 * int(1e9)
            left = np.searchsorted(mvt_valid, ag - w_ns, side="left")
            right = np.searchsorted(mvt_valid, ag, side="left")
            out[key][aep_order[s:e]] = (right - left).astype(np.float32)

        # congestion: AOBT ≤ t 数 - MVT ≤ t 数
        if len(aobt_valid) > 0 and len(mvt_valid) > 0:
            n_a = np.searchsorted(aobt_valid, ag, side="right")
            n_m = np.searchsorted(mvt_valid, ag, side="right")
            out["actl_congestion_ap"][aep_order[s:e]] = np.maximum(n_a - n_m, 0).astype(np.float32)

    print(f"  add_actual_queue_ap_only ({label}): {time.time()-t0:.0f}s, n={n:,}", flush=True)
    return out

# ============================================================
# 1) Load
# ============================================================
print("=" * 70); print("[1] load")
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
df_tr_dep_raw = df_tr[(df_tr["PHASE_mvt"]=="DEP")].copy().reset_index(drop=True)
df_tr_dep_raw = df_tr_dep_raw[
    (df_tr_dep_raw["TAXITIME_SEC_mvt"]>0)&(df_tr_dep_raw["TAXITIME_SEC_mvt"]<10800)].reset_index(drop=True)
lut = L.build_t_unimp_lut(df_tr_dep_raw)
print(f"  load {time.time()-t0:.0f}s")

# ============================================================
# 2) 实际队列特征 (纯机场级)
# ============================================================
print()
print("=" * 70); print("[2] 实际队列特征 (机场级)")
print("=" * 70); t0 = time.time()
df_ho_raw_parsed = parse_time_columns(df_ho_raw.copy(), cols=TIME_COLS)
ho_out = add_actual_queue_ap_only(df_ho_raw_parsed, "ho")
df_tr_raw_parsed = parse_time_columns(df_tr_raw.copy(), cols=TIME_COLS)
tr_out = add_actual_queue_ap_only(df_tr_raw_parsed, "tr")
print(f"  total {time.time()-t0:.0f}s")

df_ho_dep_idx = df_ho_raw_parsed["PHASE_mvt"].values == "DEP"
df_tr_dep_idx = df_tr_raw_parsed["PHASE_mvt"].values == "DEP"
df_ho_dep = df_ho_raw_parsed[df_ho_dep_idx].copy().reset_index(drop=True)
df_tr_dep_pre = df_tr_raw_parsed[df_tr_dep_idx].copy()
mask_tax = (df_tr_dep_pre["TAXITIME_SEC_mvt"].values > 0) & (df_tr_dep_pre["TAXITIME_SEC_mvt"].values < 10800)
df_tr_dep_full = df_tr_dep_pre[mask_tax].reset_index(drop=True)
for c in NEW_FEATS:
    df_ho_dep[c] = ho_out[c][df_ho_dep_idx]
    df_tr_dep_full[c] = tr_out[c][df_tr_dep_idx][mask_tax]

# ============================================================
# 3) Sanity check
# ============================================================
print()
print("=" * 70); print("[3] Sanity check")
print("=" * 70)
from scipy.stats import spearmanr
sanity_rows = []
y_train = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(float)
for c in NEW_FEATS:
    x = df_tr_dep_full[c].astype(float).values
    valid = ~np.isnan(x) & ~np.isnan(y_train)
    n = int(valid.sum())
    if n < 1000:
        sanity_rows.append({"col": c, "rho": None, "n": n,
                            "p50": None, "p99": None, "max": None, "PASS": "FAIL-low-n"})
        continue
    rho, _ = spearmanr(x[valid], y_train[valid])
    p50 = float(np.percentile(x[valid], 50))
    p99 = float(np.percentile(x[valid], 99))
    mx = float(np.max(x[valid]))
    sanity_rows.append({"col": c, "rho": float(rho), "n": n,
                        "p50": p50, "p99": p99, "max": mx,
                        "PASS": "✓" if (rho is not None and rho > 0 and p99 < 1e8) else "✗"})
    print(f"  {c:25s}  n={n:>7,}  rho={rho:+.4f}  p50={p50:>9.2f}  p99={p99:>9.2f}  max={mx:>11.2f}")

all_pass = all(r["PASS"].startswith("✓") for r in sanity_rows)
if not all_pass:
    print()
    print("  ⚠️  Sanity 失败, 停车分析 bug")
    (D0 / "ab_actual_queue.json").write_text(json.dumps({
        "sanity": sanity_rows, "decision": "STOP_bug",
    }, indent=2, ensure_ascii=False))
    print(f"  → 写出: {D0 / 'ab_actual_queue.json'}")
    sys.exit(0)

print("  ✓ 全部 sanity 通过. 训练 A/B.")

# ============================================================
# 4) 构造 71 特征
# ============================================================
print()
print("=" * 70); print("[4] 71 特征")
print("=" * 70); t0 = time.time()

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

ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
ho_dep = ho_full[ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
for c in NEW_FEATS:
    ho_dep[c] = df_ho_dep[c].values

tr_full = build_features(df_tr_raw, df_tr, lut, label="tr")
tr_dep_all = tr_full[tr_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
tr_dep_all = tr_dep_all[
    (tr_dep_all["TAXITIME_SEC_mvt"]>0)&(tr_dep_all["TAXITIME_SEC_mvt"]<10800)].reset_index(drop=True)
for c in NEW_FEATS:
    tr_dep_all[c] = df_tr_dep_full[c].values
print(f"  71 features done {time.time()-t0:.0f}s")

# ============================================================
# 5) Train A vs B
# ============================================================
print()
print("=" * 70); print("[5] A (71) vs B (71+new)")
print("=" * 70)
cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())

def encode_X(df, feats):
    X = df[feats].copy()
    for col in HIGH_CARD_CATS:
        if col in X.columns:
            m = cat_map.get(col, {})
            X[col] = X[col].astype(str).map(m).fillna(-1).astype("int32")
    for col in LOW_CARD_CATS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X

X_tr_A = encode_X(tr_dep_all, FEATS71)
X_tr_B = encode_X(tr_dep_all, FEATS71 + NEW_FEATS)
X_ho_A = encode_X(ho_dep, FEATS71)
X_ho_B = encode_X(ho_dep, FEATS71 + NEW_FEATS)

y_tr = tr_dep_all["TAXITIME_SEC_mvt"].values.astype(np.float32)
y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)

def train_predict(X_tr, X_ho, label):
    preds, boosters = [], []
    for seed in SEEDS:
        params = dict(LGB_PARAMS_BASE, seed=seed)
        ds = lgb.Dataset(X_tr, label=y_tr, categorical_feature=LOW_CARD_CATS)
        b = lgb.train(params, ds, num_boost_round=3000)
        p = np.clip(b.predict(X_ho).astype(np.float32), 30, 10800)
        preds.append(p); boosters.append(b)
        print(f"  [{label}] seed={seed} RMSE={float(np.sqrt(np.mean((p - y_ho)**2))):.2f}", flush=True)
    avg = np.mean(np.stack(preds), axis=0)
    std = float(np.std([float(np.sqrt(np.mean((p - y_ho)**2))) for p in preds]))
    return avg, std, boosters

t0 = time.time()
A_pred, A_std, _ = train_predict(X_tr_A, X_ho_A, "A")
print(f"  A 训练 {time.time()-t0:.0f}s | std={A_std:.2f}")

t0 = time.time()
B_pred, B_std, B_boosters = train_predict(X_tr_B, X_ho_B, "B")
print(f"  B 训练 {time.time()-t0:.0f}s | std={B_std:.2f}")

# importance
imp_b = pd.DataFrame({
    "feat": X_tr_B.columns.tolist(),
    "gain": np.mean([b.feature_importance(importance_type="gain") for b in B_boosters], axis=0),
}).sort_values("gain", ascending=False).reset_index(drop=True)
print("\n  B 版 特征 importance (gain) top 15:")
print(imp_b.head(15).to_string())

# ============================================================
# 6) 评估
# ============================================================
print()
print("=" * 70); print("[6] A vs B 评估")
print("=" * 70)
m_full = y_ho > 0
m_core = (y_ho > 0) & (y_ho <= 10800)
m_short = m_core & (y_ho <= 1800)
m_long = m_core & (y_ho > 1800)
m_gt10k = m_full & (y_ho > 10800)
aps = ho_dep["ADEP_mvt"].astype(str).values

def ev(pred, label):
    out = {
        "n_full": int(m_full.sum()),
        "n_core": int(m_core.sum()),
        "n_short": int(m_short.sum()),
        "n_long": int(m_long.sum()),
        "n_gt10k": int(m_gt10k.sum()),
        "rmse_full": float(np.sqrt(np.mean((pred[m_full] - y_ho[m_full])**2))),
        "rmse_core": float(np.sqrt(np.mean((pred[m_core] - y_ho[m_core])**2))),
        "rmse_short": float(np.sqrt(np.mean((pred[m_short] - y_ho[m_short])**2))),
        "rmse_long": float(np.sqrt(np.mean((pred[m_long] - y_ho[m_long])**2))) if m_long.sum() else None,
        "rmse_gt10k": float(np.sqrt(np.mean((pred[m_gt10k] - y_ho[m_gt10k])**2))) if m_gt10k.sum() else None,
        "bias_full": float(np.mean(pred[m_full] - y_ho[m_full])),
        "bias_core": float(np.mean(pred[m_core] - y_ho[m_core])),
        "bias_short": float(np.mean(pred[m_short] - y_ho[m_short])),
        "bias_long": float(np.mean(pred[m_long] - y_ho[m_long])) if m_long.sum() else None,
    }
    print(f"\n  [{label}]")
    print(f"    full   (n={out['n_full']:,}):   RMSE={out['rmse_full']:.2f}")
    print(f"    core   (n={out['n_core']:,}):   RMSE={out['rmse_core']:.2f}")
    print(f"    ★short (n={out['n_short']:,}):   RMSE={out['rmse_short']:.2f}  bias={out['bias_short']:+.2f}")
    print(f"    long   (n={out['n_long']:,}):   RMSE={out['rmse_long']:.2f}  bias={out['bias_long']:+.2f}")
    return out

A_eval = ev(A_pred, "A 版 71")
B_eval = ev(B_pred, "B 版 71+new")

print()
print("  分机场 (★ 短滑行段):")
print(f"    {'ADEP':6s} {'n':>6s}  {'A_short':>9s} {'B_short':>9s} {'Δ':>9s}  "
      f"{'A_core':>9s} {'B_core':>9s} {'Δ_core':>9s}")
FOCUS_AP = ["EGLL", "LIRF", "LFPG", "EHAM"]
A_per, B_per = {}, {}
for ap in sorted(np.unique(aps)):
    m_ap = (aps == ap) & m_core
    n = int(m_ap.sum())
    if n < 100: continue
    m_short_ap = m_ap & (y_ho <= 1800)
    a_short = float(np.sqrt(np.mean((A_pred[m_short_ap] - y_ho[m_short_ap])**2)))
    b_short = float(np.sqrt(np.mean((B_pred[m_short_ap] - y_ho[m_short_ap])**2)))
    a_core = float(np.sqrt(np.mean((A_pred[m_ap] - y_ho[m_ap])**2)))
    b_core = float(np.sqrt(np.mean((B_pred[m_ap] - y_ho[m_ap])**2)))
    focus = "★" if ap in FOCUS_AP else " "
    print(f"    {focus}{ap:5s} {n:>6,d}  {a_short:>9.2f} {b_short:>9.2f} {b_short - a_short:>+9.2f}  "
          f"{a_core:>9.2f} {b_core:>9.2f} {b_core - a_core:>+9.2f}")
    A_per[ap] = {"n": n, "rmse_short": a_short, "rmse_core": a_core}
    B_per[ap] = {"n": n, "rmse_short": b_short, "rmse_core": b_core}

print()
print("  重点机场 A vs B (短滑行段):")
print(f"    {'ADEP':6s} {'A_short':>9s} {'B_short':>9s} {'Δ':>9s}  verdict")
seed_jit = max(A_std, B_std)
for ap in FOCUS_AP:
    a = A_per.get(ap); b = B_per.get(ap)
    if not a or not b: continue
    d = b["rmse_short"] - a["rmse_short"]
    v = "✓" if d < -seed_jit else " "
    print(f"    {ap:6s} {a['rmse_short']:>9.2f} {b['rmse_short']:>9.2f} {d:>+9.2f}  {v}")

print()
print("=" * 70); print("[7] 总判定")
print("=" * 70)
delta_short = B_eval["rmse_short"] - A_eval["rmse_short"]
delta_core = B_eval["rmse_core"] - A_eval["rmse_core"]
effective = (delta_short < -seed_jit) and (delta_core < 0)
print(f"    A short RMSE: {A_eval['rmse_short']:.2f}")
print(f"    B short RMSE: {B_eval['rmse_short']:.2f}  Δ={delta_short:+.2f}")
print(f"    A core RMSE:  {A_eval['rmse_core']:.2f}")
print(f"    B core RMSE:  {B_eval['rmse_core']:.2f}  Δ={delta_core:+.2f}")
print(f"    seed 抖动: A={A_std:.2f}, B={B_std:.2f}, max={seed_jit:.2f}")
print()
if effective:
    print(f"  ✅ 结论: B 版 实际队列特征 有效")
    ap_short_improved = sum(1 for ap in B_per if B_per[ap]["rmse_short"] < A_per[ap]["rmse_short"])
    ap_total = len(B_per)
    print(f"     分机场同号改善 (短滑行段): {ap_short_improved}/{ap_total}")
else:
    reasons = []
    if delta_short > -seed_jit:
        reasons.append(f"短滑行段 Δ{delta_short:+.2f} 不足 -{seed_jit:.2f}")
    if delta_core >= 0:
        reasons.append(f"核心 RMSE 涨 {delta_core:+.2f}")
    print(f"  ❌ 结论: B 版 实际队列特征 无效")
    for r in reasons:
        print(f"     - {r}")

out = {
    "sanity": sanity_rows,
    "new_feats": NEW_FEATS,
    "A_eval": A_eval, "B_eval": B_eval,
    "A_per_ap": A_per, "B_per_ap": B_per,
    "importance_top": imp_b.head(20).to_dict("records"),
    "seed_jitter_max": seed_jit,
    "delta_short": delta_short, "delta_core": delta_core,
    "effective": bool(effective),
    "seeds": SEEDS,
    "note": "v3: 5 个特征全机场级, heavy_frac 暂取消",
}
(D0 / "ab_actual_queue.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print()
print(f"  → 写出: {D0 / 'ab_actual_queue.json'}")
