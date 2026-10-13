"""
模型体检 v1 (read-only) — 71 特征 LGB holdout (2025-01+07)
==========================================================
输入: models/lgb_full_v15/lgb_seed{42,123,2024}.txt (A版基线 71 特征)
输出: notebooks/diag/d0/model_checkup.json

体检:
  1) 特征重要性: gain / gain占比 / SHAP均值绝对值 (降序), 标 gain占比<0.1% 疑似无用
  2) 冗余: 数值特征两两 Spearman, |ρ|>0.90 标出 + 保留建议
  3) 残差结构: 真值-预测 按
      (分机场, 分跑道配, 分小时, 分星期, 分尾流等级, 分AOBT是否有效)
     切片, 每片报 n/RMSE/bias; |bias|>30 且 n>200 标"系统性偏差"

红线: 不改 label, 不 clip 真值, 不建单机场模型
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

# 重用 v15 的 71 特征列表
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
NUMERIC_FEATS = [c for c in FEATS71 if c not in (HIGH_CARD_CATS + LOW_CARD_CATS)]
SEEDS = [42, 123, 2024]

print("=" * 70); print("[1] load + build holdout feature")
print("=" * 70); t0 = time.time()
ho_dfs = [load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet", columns=TRAIN_COLS)
          for m in (1, 7)]
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in train_months:
    if m < 12:
        fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    else:
        fn = "data/raw/training_2025-12-01_2026-01-01.parquet"
    train_dfs.append(load_month(fn, columns=TRAIN_COLS))
# (note: actual file is training_2025-12-01_2026-01-01.parquet)
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

ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
ho_dep = ho_full[ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
print(f"  build_features done {time.time()-t0:.0f}s, holdout DEP={len(ho_dep):,}")

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

X_ho = encode_X(ho_dep, FEATS71).reset_index(drop=True)
y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)
print(f"  X_ho.shape={X_ho.shape}")

print()
print("=" * 70); print("[2] load 3 boosters + predict")
print("=" * 70); t0 = time.time()
boosters = [lgb.Booster(model_file=str(ROOT / "models/lgb_full_v15" / f"lgb_seed{s}.txt"))
            for s in SEEDS]
preds = [np.clip(b.predict(X_ho).astype(np.float32), 30, 10800) for b in boosters]
pred = np.mean(np.stack(preds), axis=0)
resid = y_ho - pred
rmse_overall = float(np.sqrt(np.mean(resid**2)))
print(f"  加载+预测 {time.time()-t0:.0f}s | 3-seed 均值 RMSE={rmse_overall:.2f}")

# ============================================================
# [3] 特征重要性 (gain, SHAP)
# ============================================================
print()
print("=" * 70); print("[3] 特征重要性: gain + SHAP")
print("=" * 70); t0 = time.time()
# gain: per booster
gain = np.mean([b.feature_importance(importance_type="gain") for b in boosters], axis=0)
split_count = np.mean([b.feature_importance(importance_type="split") for b in boosters], axis=0)
total_gain = float(gain.sum())
imp = pd.DataFrame({
    "feat": X_ho.columns.tolist(),
    "gain": gain,
    "gain_pct": gain / total_gain * 100,
    "split": split_count,
}).sort_values("gain", ascending=False).reset_index(drop=True)

# SHAP: 用 shap.TreeExplainer, 在 holdout subsample 5000 rows (344k 太慢)
shap_means_abs = np.zeros(len(X_ho.columns))
rng = np.random.default_rng(42)
sub_size = min(8000, len(X_ho))
sub_idx = rng.choice(len(X_ho), size=sub_size, replace=False)
X_sub = X_ho.iloc[sub_idx]
# 使用 booster[0] 算 SHAP (单 booster 已经能代表)
explainer = shap.TreeExplainer(boosters[0])
shap_values = explainer.shap_values(X_sub)
shap_means_abs = np.mean(np.abs(shap_values), axis=0)
print(f"  SHAP done {time.time()-t0:.0f}s on sub={sub_size}")

imp["shap_mean_abs"] = shap_means_abs
total_shap = float(imp["shap_mean_abs"].sum())
imp["shap_pct"] = imp["shap_mean_abs"] / total_shap * 100
imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

# 疑似无用 (gain_pct<0.1% AND shap_pct<0.1%)
useless = imp[(imp["gain_pct"] < 0.1) & (imp["shap_pct"] < 0.1)].copy()
print(f"\n  特征 importance top 10 (gain):")
print(imp.head(10)[["feat", "gain", "gain_pct", "shap_mean_abs", "shap_pct"]].to_string(index=False))
print(f"\n  疑似无用特征 (gain<0.1% AND shap<0.1%) 共 {len(useless)} 个:")
for c in useless["feat"].tolist():
    print(f"    - {c}")

# ============================================================
# [4] 冗余 Spearman
# ============================================================
print()
print("=" * 70); print("[4] 冗余 Spearman (数值特征)")
print("=" * 70); t0 = time.time()
# 数值列 (skip categoricals 的 int32 编码也算数值)
NUMERIC_ALL = NUMERIC_FEATS.copy()
sub_num = X_ho[NUMERIC_ALL].astype(float).copy()
nunique = sub_num.nunique()
const_cols = nunique[nunique < 2].index.tolist()
if const_cols:
    print(f"  跳过 const 列: {const_cols}")
NUMERIC_ALL = [c for c in NUMERIC_ALL if c not in const_cols]

# 为效率: holdout 上 5000 采样
sub_idx = rng.choice(len(X_ho), size=min(5000, len(X_ho)), replace=False)
S = X_ho.iloc[sub_idx][NUMERIC_ALL].astype(float)
S = S.fillna(S.median())
rho_mat = S.corr(method="spearman")
pairs = []
for i in range(len(NUMERIC_ALL)):
    for j in range(i+1, len(NUMERIC_ALL)):
        c1, c2 = NUMERIC_ALL[i], NUMERIC_ALL[j]
        r = rho_mat.loc[c1, c2]
        if pd.notna(r) and abs(r) > 0.9:
            pairs.append({
                "feat_a": c1, "feat_b": c2, "rho": float(r),
                "keep_suggest": "保留 gain 较高者",
                "gain_a_pct": float(imp[imp["feat"]==c1]["gain_pct"].values[0]) if len(imp[imp["feat"]==c1]) else 0,
                "gain_b_pct": float(imp[imp["feat"]==c2]["gain_pct"].values[0]) if len(imp[imp["feat"]==c2]) else 0,
            })
pairs.sort(key=lambda x: -abs(x["rho"]))
print(f"  Spearman 数: {len(NUMERIC_ALL)}^2 = {len(NUMERIC_ALL)**2:,} | 冗余 (|ρ|>0.9): {len(pairs)} 对")
for p in pairs[:20]:
    print(f"    {p['feat_a']:25s}  ↔  {p['feat_b']:25s}  ρ={p['rho']:+.4f}  (gain: {p['gain_a_pct']:.2f}% / {p['gain_b_pct']:.2f}%)")

# ============================================================
# [5] 残差结构 (按维度切片)
# ============================================================
print()
print("=" * 70); print("[5] 残差结构")
print("=" * 70)

def slice_summary(mask, label):
    n = int(mask.sum())
    if n < 50:
        return None
    rmse = float(np.sqrt(np.mean(resid[mask]**2)))
    bias = float(np.mean(resid[mask]))
    return {"n": n, "rmse": rmse, "bias": bias, "verdict": None}

def flag_systemic(records, n_thr=200, bias_thr=30):
    """记录带 verdict: systemic if |bias|>30 and n>200"""
    for r in records:
        if r is None: continue
        r["verdict"] = ("⚠ systemic-bias" if (r["n"] > n_thr and abs(r["bias"]) > bias_thr) else "ok")
    return records

# 5a) 分机场
per_ap = []
for ap in sorted(ho_dep["ADEP_mvt"].astype(str).unique()):
    m = (ho_dep["ADEP_mvt"].astype(str).values == ap) & (y_ho > 0)
    r = slice_summary(m, ap)
    if r is not None:
        r["ADEP"] = ap
        per_ap.append(r)
per_ap = [r for r in flag_systemic(per_ap) if r is not None]

# 5b) 分跑道配
ho_dep["runway_num"] = ho_dep["RUNWAY_mvt"].astype(str).str.extract(r'(\d{2})').fillna("")
def rwy_kind(s):
    if not s or not s.isdigit(): return "Other"
    n = int(s)
    r1 = n % 10
    if r1 == 9: return f"RWY {s} (27 family)"
    return f"RWY {s}"
ho_dep["rwy_label"] = ho_dep["runway_num"].apply(rwy_kind)
per_rwy = []
for lab in sorted(ho_dep[ho_dep["rwy_label"]!="RWY ()"]["rwy_label"].unique()):
    m = (ho_dep["rwy_label"].values == lab) & (y_ho > 0)
    r = slice_summary(m, lab)
    if r is not None:
        r["runway"] = lab
        per_rwy.append(r)
per_rwy = [r for r in flag_systemic(per_rwy) if r is not None]

# 5c) 分小时
ho_dep["sched_hour"] = ho_dep["sched_hour"].astype(int)
per_hour = []
for h in sorted(ho_dep["sched_hour"].unique()):
    m = (ho_dep["sched_hour"].values == h) & (y_ho > 0)
    r = slice_summary(m, str(h))
    if r is not None:
        r["hour"] = h
        per_hour.append(r)
per_hour = [r for r in flag_systemic(per_hour) if r is not None]

# 5d) 分星期
ho_dep["sched_dow"] = ho_dep["sched_dow"].astype(int)
per_dow = []
for d in sorted(ho_dep["sched_dow"].unique()):
    m = (ho_dep["sched_dow"].values == d) & (y_ho > 0)
    r = slice_summary(m, str(d))
    if r is not None:
        r["dow"] = d
        per_dow.append(r)
per_dow = [r for r in flag_systemic(per_dow) if r is not None]

# 5e) 分尾流等级 (J+H 是 heavy, M/L 是 light)
ho_dep["wake_class"] = ho_dep["WK_TBL_CAT_flt"].astype(str).map({
    "J": "heavy", "H": "heavy", "M": "medium", "L": "light"
}).fillna("unknown")
per_wake = []
for w in ["heavy", "medium", "light", "unknown"]:
    m = (ho_dep["wake_class"].values == w) & (y_ho > 0)
    r = slice_summary(m, w)
    if r is not None:
        r["wake"] = w
        per_wake.append(r)
per_wake = [r for r in flag_systemic(per_wake) if r is not None]

# 5f) 分 AOBT 是否有效
ho_dep["aobt_valid"] = ho_dep["AOBT_3_flt"].notna()
per_aobt = []
for v, lab in [(True, "AOBT_valid"), (False, "AOBT_NaT")]:
    m = (ho_dep["aobt_valid"].values == v) & (y_ho > 0)
    r = slice_summary(m, lab)
    if r is not None:
        r["aobt"] = lab
        per_aobt.append(r)

# 输出诊断摘要
print("\n  残差结构 (按维度) — ★: |bias|>30 且 n>200")
def show_dim(records, key, name):
    recs = [r for r in records if r is not None]
    recs.sort(key=lambda x: -abs(x.get("bias", 0)))
    print(f"\n  [{name}]")
    for r in recs[:8]:
        flag = r.get("verdict","")
        print(f"    {key}={r[key]:>6}  n={r['n']:>6,}  RMSE={r['rmse']:>7.2f}  bias={r['bias']:+8.2f}  {flag}")
    print(f"      (共 {len(recs)} 片)")

show_dim(per_ap, "ADEP", "分机场")
show_dim(per_rwy, "runway", "分跑道配")
show_dim(per_hour, "hour", "分小时")
show_dim(per_dow, "dow", "分星期")
show_dim(per_wake, "wake", "分尾流等级")
show_dim(per_aobt, "aobt", "分 AOBT 有效性")

systemic = []
for recs in [per_ap, per_rwy, per_hour, per_dow, per_wake, per_aobt]:
    for r in recs:
        if r.get("verdict") == "⚠ systemic-bias":
            systemic.append(r)
print(f"\n  ★ 系统性偏差 (|bias|>30 且 n>200) 共 {len(systemic)} 片:")
for r in systemic:
    key = next((k for k in r if k not in ("n","rmse","bias","verdict")), "?")
    print(f"    [{key}={r[key]}] n={r['n']:,} RMSE={r['rmse']:.1f} bias={r['bias']:+.1f}")

# ============================================================
# 落盘
# ============================================================
out = {
    "n_holdout_dep": int(len(X_ho)),
    "overall_rmse_3seed": rmse_overall,
    "feature_importance": imp.head(50).to_dict("records"),
    "useless_features": useless["feat"].tolist(),
    "n_useless": len(useless),
    "redundant_pairs": pairs,
    "residual_per_ap": per_ap,
    "residual_per_rwy": per_rwy,
    "residual_per_hour": per_hour,
    "residual_per_dow": per_dow,
    "residual_per_wake": per_wake,
    "residual_per_aobt": per_aobt,
    "systemic_bias_records": systemic,
    "n_systemic": len(systemic),
    "model_source": "models/lgb_full_v15/ (A 版 71 特征, 3 seed 集成), holdout 2025-01+07",
    "red_lines": "不改 label / 不 clip 真值 / 不建单机场模型",
    "shap_subsample_n": int(sub_size),
}
(D0 / "model_checkup.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print()
print(f"  → 写出: {D0 / 'model_checkup.json'}")
