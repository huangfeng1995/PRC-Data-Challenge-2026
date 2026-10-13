"""
短段(≤1800)系统性高估归因 (read-only)
=====================================
仅做诊断, 不重训发版, 不上传, 不改真值.

输入: models/lgb_full_v15 (A 版 71 特征 LGB 3-seed 集成)
数据: holdout 2025-01+07 (DEP 344,419, 短段 328,707)

Step1: 短段按真值分 6 档, 看 bias 形态 (平移 vs 回归)
Step2: 短段残差按多维切片 (ADEP/RWY/STAND/TYPE/OPERATOR/t_unimp), 找 |bias|>40 & n>300 的组合
Step3: 查 t_unimp_r7 vs 真值的口径偏差

输出: notebooks/diag/d0/short_bias_diagnose.json + parquet cache (holdout_diag_cache.parquet)
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

# ============================================================
# 0) Load + build_features + 3-seed predict
# ============================================================
print("=" * 70); print("[0] load + build_features + predict")
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
boosters = [lgb.Booster(model_file=str(ROOT / "models/lgb_full_v15" / f"lgb_seed{s}.txt"))
            for s in (42, 123, 2024)]
pred = np.mean(np.stack([np.clip(b.predict(X_ho).astype(np.float32), 30, 10800) for b in boosters]), axis=0)
resid = (y_ho - pred).astype(np.float32)
print(f"  预测 done {time.time()-t0:.0f}s")

# ============================================================
# 缓存: parquet 写入, 后续脚本读
# ============================================================
df_cache = pd.DataFrame({
    "ADEP_mvt": ho_dep["ADEP_mvt"].astype(str).values,
    "ADES_mvt": ho_dep["ADES_mvt"].astype(str).values,
    "RUNWAY_mvt": ho_dep["RUNWAY_mvt"].astype(str).values,
    "STAND_mvt": ho_dep["STAND_mvt"].astype(str).values,
    "AIRCRAFT_TYPE_mvt": ho_dep["AIRCRAFT_TYPE_mvt"].astype(str).values,
    "AIRCRAFT_OPERATOR_flt": ho_dep["AIRCRAFT_OPERATOR_flt"].astype(str).values,
    "t_unimp_r7": ho_dep["t_unimp_r7"].values,
    "y_true": y_ho,
    "y_pred": pred,
    "resid": resid,
    "short_strict": (y_ho > 0) & (y_ho <= 1800),  # 严格 short 段
    "core": (y_ho > 0) & (y_ho <= 10800),
})
cache_path = D0 / "holdout_diag_cache.parquet"
df_cache.to_parquet(cache_path)
print(f"  cache 写入: {cache_path}, 行数={len(df_cache)}")

# ============================================================
# Step1: 短段按真值分 6 档, 看 bias 形态
# ============================================================
print()
print("=" * 70); print("[Step1] 短段按真值分 6 档")
print("=" * 70)
bins = [(30, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1800)]
m_s = df_cache["short_strict"].values  # 严格 short 段 (≤1800)

step1 = []
print(f"\n  严格短段 (n={int(m_s.sum()):,}, y_ho in (0, 1800])")
print(f"  {'bin':15s}  {'n':>7s}  {'pred_mean':>10s}  {'true_mean':>10s}  {'bias':>9s}  {'rmse':>8s}")
for lo, hi in bins:
    m = m_s & (y_ho >= lo) & (y_ho < hi)
    n = int(m.sum())
    if n == 0: continue
    pred_mean = float(pred[m].mean())
    true_mean = float(y_ho[m].mean())
    bias = float(np.mean(pred[m] - y_ho[m]))
    rmse = float(np.sqrt(np.mean((pred[m] - y_ho[m])**2)))
    step1.append({"bin": f"[{lo}, {hi})", "lo": lo, "hi": hi, "n": n,
                   "pred_mean": pred_mean, "true_mean": true_mean,
                   "bias": bias, "rmse_bin": rmse})
    print(f"  [{lo:>4d}, {hi:>4d})  {n:>7,}  {pred_mean:>10.2f}  {true_mean:>10.2f}  {bias:+9.2f}  {rmse:>8.2f}")

# 严格短段 (n=328,707) bias & RMSE (与体检对齐)
rmse_short = float(np.sqrt(np.mean(resid[m_s]**2)))
bias_short = float(np.mean(resid[m_s]))
print(f"\n  严格短段整体 (n={int(m_s.sum()):,}): RMSE={rmse_short:.2f}, bias={bias_short:+.2f}")

step1_summary = {
    "bias_overall": bias_short,
    "rmse_overall": rmse_short,
    "n_short": int(m_s.sum()),
    "judgment": None,
}
# 判断: 平移 vs 回归
biases = [r["bias"] for r in step1]
bias_range = max(biases) - min(biases)
mean_bias = float(np.mean(biases))
first_last_diff = biases[0] - biases[-1]  # 正 → 回归 (首档偏正多, 末档偏负多)
if bias_range < 30:
    step1_summary["judgment"] = (f"平移型 (各档 bias_range={bias_range:.1f}s < 30s, 均值={mean_bias:+.0f}s)")
elif first_last_diff > 100:
    step1_summary["judgment"] = (f"向均值回归型 (首档 bias={biases[0]:+.0f}s 渐降到末档 {biases[-1]:+.0f}s, "
                                  f"首-末差={first_last_diff:+.0f}s, 回归证据显著)")
elif abs(biases[0]) < 50 and abs(biases[-1]) < 50:
    step1_summary["judgment"] = (f"近似平移 (各档偏差均在 ±50s 内, 全段偏高 {mean_bias:+.0f}s)")
else:
    step1_summary["judgment"] = (f"混合: 首-末差 {first_last_diff:+.0f}s, bias_range {bias_range:.0f}s")

# 如果消 bias, 短段 RMSE 改进幅度估算
err2 = resid**2
ss_short_total = float(err2[m_s].sum())
n_short = int(m_s.sum())
current_rmse = np.sqrt(ss_short_total / n_short)
# 若整段减掉 bias, SSE = sum((pred - true - bias)^2) = sum(resid^2 - bias^2 * n_per_g)?
# 简化: 直接按全段减 bias 估算 SSE_new
target_sse = ss_short_total - bias_short**2 * n_short
target_rmse = np.sqrt(max(target_sse, 0) / n_short)
step1_summary["if_bias_canceled_rmse"] = target_rmse
step1_summary["rmse_drop"] = current_rmse - target_rmse
print(f"  若短段 bias 全消除: {current_rmse:.2f} → {target_rmse:.2f}  (Δ={current_rmse - target_rmse:+.2f})")
# 整体核心 RMSE
m_core = (y_ho > 0) & (y_ho <= 10800)
n_core = int(m_core.sum())
ss_core = float(err2[m_core].sum())
rmse_core = np.sqrt(ss_core / n_core)
# 若 short 段 bias 减半 (e.g. 后处理校正 50% of bias)
half_bias_short_corr = bias_short * 0.5
target_ss_short = ss_short_total - bias_short * half_bias_short_corr * n_short  # 不严格
# 直接算: short 段 SSE = n*var(resid) = n*(rmse^2); 减 bias 后的 var 不变; 但 mean 减半后残差偏均值
# 准确: var_unbiased = var(resid) (mean shift 不影响 var)
rmse_short_debiased = float(np.sqrt(np.var(resid[m_s])))
step1_summary["rmse_short_debiased_estimate"] = rmse_short_debiased
step1_summary["rmse_core_debiased_estimate"] = float(np.sqrt((ss_core - 0)/n_core))  # core 不动
# 修正: 短段 bias 减半 → SSE 修正估算
new_ss_short = float(np.sum((resid[m_s] - half_bias_short_corr)**2))
target_core = float(np.sqrt((ss_core + new_ss_short - ss_short_total)/n_core)) if False else None
# 直接算: 假设短段 dev bias 后短段 SSE:
# SSE_new = sum((r-bias_new)^2) = sum((r-bias)^2) + n*bias^2 - 2*sum(r*bias) - n*bias^2 + 2*sum(r)*bias - ...
# 简化: SSE_new = sum((r - bias_new)^2) where bias_new = bias_old/2
# = sum((r-bias_old + bias_old/2)^2) = sum((r-bias_old)^2 + bias_old^2/4 + bias_old*(r-bias_old))
# ≈ SSE_old + n*bias_old^2/4 + 0 (残差项与(bias_old/2)独立)
# 实际: SSE_new ≈ SSE_old - (3/4) * n * bias_old^2 (因为分摊去除 mean shift 后 var 不变)
# 但均值从 bias_old → bias_old/2, var 不变 = rmse^2
# 那 RMSE 还是不变? 不对, 因为 (r - bias/2)^2 = (r-bias)^2 + (3/4)*bias^2 + bias*(r-bias)
# expectation of (r-bias)^2 = var = rmse^2
# exp of bias*(r-bias) = 0 (因为 r-bias 已中心化)
# exp SSE_new = SSE_old + n * (3/4) * bias_old^2
# 但是 — 没取消才对: 真实 (r - 0)^2 = r^2, mean shift 0, var=r^2 的 mean = r^2
# 对: y - pred - bias_new: 求 SSE_new = n * var(r-bias_new) + n * mean(r-bias_new)^2
#                                  = n * var(r) + n * (mean(r)-bias_new)^2
# mean(r) = bias_old, so SSE_new = n*var + n*(bias_old - bias_new)^2 = SSE_old + n*(bias_old - bias_new)^2
# 若 bias_new = 0: SSE_0 = SSE_old + n*bias_old^2 → 大! (反向)
# 所以整体 RMSE 必须按差分公式重算
rmse_core_with_short_debiased = float(np.sqrt((ss_core + new_ss_short)/n_core))
print(f"  短段 bias 减半: short SSE {ss_short_total:.3e} → {new_ss_short:.3e}, 整体核心 RMSE: {rmse_core:.2f} → {rmse_core_with_short_debiased:.2f}")

step1_summary["n_core"] = n_core
step1_summary["rmse_core"] = rmse_core
step1_summary["rmse_core_if_short_debiased_half"] = rmse_core_with_short_debiased

# ============================================================
# Step2: 短段残差按多维切片, 找 |bias|>40 & n>300 组合
# ============================================================
print()
print("=" * 70); print("[Step2] 短段残差 — 多维切片 (|bias|>40 & n>300)")
print("=" * 70)
df_short = df_cache[m_s].copy()

def slice_bias(df, key_label, key_cols):
    g = df.groupby(key_cols).agg(
        n=("resid", "size"),
        bias_mean=("resid", "mean"),
        rmse=("resid", lambda s: float(np.sqrt(np.mean(s**2)))),
    ).reset_index()
    g["abs_bias"] = g["bias_mean"].abs()
    g = g.sort_values(["abs_bias", "n"], ascending=[False, False])
    return g

def report_below(g, n_thr=300, bias_thr=40):
    """报 n>300 且 |bias|>40"""
    sig = g[(g["n"] > n_thr) & (g["abs_bias"] > bias_thr)].copy()
    return sig

# 2a) (ADEP, RUNWAY)
g_ar = slice_bias(df_short, "(ADEP, RUNWAY)", ["ADEP_mvt", "RUNWAY_mvt"])
print("\n  [2a] (ADEP, RUNWAY) - top 20 by |bias|:")
print(g_ar.head(20).to_string(index=False))
sig_ar = report_below(g_ar)

# 2b) (ADEP, STAND)
g_as = slice_bias(df_short, "(ADEP, STAND)", ["ADEP_mvt", "STAND_mvt"])
print("\n  [2b] (ADEP, STAND) - top 20 by |bias|:")
print(g_as.head(20).to_string(index=False))
sig_as = report_below(g_as)

# 2c) AIRCRAFT_TYPE
g_at = slice_bias(df_short, "AIRCRAFT_TYPE", ["AIRCRAFT_TYPE_mvt"])
print("\n  [2c] AIRCRAFT_TYPE - 全 (按 |bias|):")
print(g_at.head(20).to_string(index=False))
sig_at = report_below(g_at)

# 2d) AIRCRAFT_OPERATOR_flt
g_op = slice_bias(df_short, "OPERATOR", ["AIRCRAFT_OPERATOR_flt"])
print("\n  [2d] OPERATOR - top 20 by |bias|:")
print(g_op.head(20).to_string(index=False))
sig_op = report_below(g_op)

# 2e) t_unimp 分档 (10 档)
df_short["t_unimp_q"] = pd.qcut(df_short["t_unimp_r7"], q=10, labels=False, duplicates="drop")
g_tu = slice_bias(df_short, "t_unimp_q", ["t_unimp_q"])
print("\n  [2e] t_unimp_r7 分 10 档 (按 |bias|):")
print(g_tu.sort_values("t_unimp_q").to_string(index=False))

# 汇总: 主要看 (ADEP, RUNWAY) 维度 - 它是最强的结构信号
# 多维度按"a)某个维度总行数 / 总 short 行数" 看集中度
n_total = int(m_s.sum())
print()
print("  各维度 n>300 & |bias|>40 行占比:")
for name, sig_df in [("(ADEP, RUNWAY)", sig_ar), ("(ADEP, STAND)", sig_as),
                     ("AIRCRAFT_TYPE", sig_at), ("OPERATOR", sig_op)]:
    if len(sig_df) > 0:
        sig_rows = int(sig_df["n"].sum())
        print(f"    {name:25s}:  {len(sig_df):>4d} 组合, 占 {sig_rows/n_total*100:>6.2f}% 的短段 ({sig_rows:,} 行), 最大|bias|={float(sig_df.abs_bias.max()):.0f}s")

# 综合判定: 主要看 (ADEP, RUNWAY) 集中度
if len(sig_ar) > 0:
    ar_pct_of_short = float(sig_ar.n.sum()) / n_total * 100
    ar_count = len(sig_ar)
else:
    ar_pct_of_short = 0
    ar_count = 0

# (ADEP, RUNWAY) 层组合
step2 = {
    "n_short_total": n_total,
    "n_sig_combinations_by_dim": {
        "ADEP_RUNWAY": {"n_groups": int(len(sig_ar)), "n_sig_rows_pct": ar_pct_of_short if len(sig_ar) else 0},
        "ADEP_STAND": {"n_groups": int(len(sig_as)), "n_sig_rows_pct": float(sig_as.n.sum())/n_total*100 if len(sig_as) else 0},
        "AIRCRAFT_TYPE": {"n_groups": int(len(sig_at)), "n_sig_rows_pct": float(sig_at.n.sum())/n_total*100 if len(sig_at) else 0},
        "OPERATOR": {"n_groups": int(len(sig_op)), "n_sig_rows_pct": float(sig_op.n.sum())/n_total*100 if len(sig_op) else 0},
    },
    "top_combinations_(ADEP,RUNWAY)": sig_ar.head(20).to_dict("records") if len(sig_ar) else [],
    "t_unimp_q10": g_tu.to_dict("records"),
    "judgment": None,
}
# 均匀散布: std(bias) 在所有 n>300 组中 < X
# OLD 判断块已删除, 用新的 [verdict] 段

# ============================================================
# Step3: t_unimp_r7 vs 短段真值口径核对
# ============================================================
print()
print("=" * 70); print("[Step3] t_unimp_r7 口径核对")
print("=" * 70)

# 训练折 P30 baseline (用于和短段真值中位差)
# 实际上 t_unimp_r7 已经在 LUT 中分位数 0.3 计算, 但实际是按 (ADEP,STAND,RUNWAY) 查询
# 这里检视: 在 holdout 上, t_unimp_r7 vs 短段真值中位数 的差 (按 ADEP×RUNWAY×STAND 分组)

df_with_tu = df_short.dropna(subset=["t_unimp_r7"]).copy()
# t_unimp 偏高的依据: t_unimp_r7 > 短段 TAXITIME 中位数
df_with_tu["bias_tu"] = df_with_tu["t_unimp_r7"] - df_with_tu["y_true"]

g_tu_method = df_with_tu.groupby(["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]).agg(
    n=("y_true", "size"),
    t_unimp_median=("t_unimp_r7", "median"),
    true_median=("y_true", "median"),
    tu_vs_true_median=("y_true", lambda s: float(df_with_tu.loc[s.index, "t_unimp_r7"].median() - s.median())),
    pred_median=("y_pred", "median"),
).reset_index()
g_tu_method = g_tu_method[g_tu_method["n"] >= 50]  # 过滤小样本
g_tu_method["abs_diff"] = g_tu_method["tu_vs_true_median"].abs()
g_tu_method = g_tu_method.sort_values("abs_diff", ascending=False)

# 基准口径偏高判断: t_unimp_r7 - 真值中位 的中位数
median_diff = float(g_tu_method["tu_vs_true_median"].median())
mean_diff = float(g_tu_method["tu_vs_true_median"].mean())
median_abs_diff = float(g_tu_method["abs_diff"].median())
# 若 median_diff > 30 (基准高估短段 → 长段修正可能受益)
print(f"\n  在 holdout 短段, 按 (ADEP, RUNWAY, STAND) 分组:")
print(f"    t_unimp_r7 - 真值中位 的中位数 = {median_diff:+.2f}s (基准偏高为正)")
print(f"    t_unimp_r7 - 真值中位 的均值   = {mean_diff:+.2f}s")
print(f"    |差| 的中位数                 = {median_abs_diff:.2f}s")
print(f"    组数 (n≥50): {len(g_tu_method)}")
if median_diff > 30:
    print(f"  → 基准口径偏高 (median_diff={median_diff:+.0f}s > 30s)")
elif median_diff < -30:
    print(f"  → 基准口径偏低 (median_diff={median_diff:+.0f}s < -30s)")
else:
    print(f"  → 基准无系统性偏差 (|median_diff|<30s)")

# 看 pred_median vs true_median 的差 (验证短段高估是什么因素)
print(f"\n  Pred - True 中位数 (按 (A,R,S) 分组):")
g_tu_method["pred_minus_true"] = g_tu_method["pred_median"] - g_tu_method["true_median"]
print(f"    中位数 = {float(g_tu_method.pred_minus_true.median()):+.2f}s")
print(f"    均值   = {float(g_tu_method.pred_minus_true.mean()):+.2f}s")
print(f"    |差| 中位 = {float(g_tu_method.pred_minus_true.abs().median()):.2f}s")

# 短段真值 ≤ 1800 的全样本 P50 真值 vs t_unimp P50 差:
g_glob = df_short.groupby("ADEP_mvt").agg(
    n=("y_true", "size"),
    true_med=("y_true", "median"),
    tu_med=("t_unimp_r7", "median"),
    pred_med=("y_pred", "median"),
).reset_index()
g_glob = g_glob[g_glob["n"] >= 1000]
print(f"\n  按 ADEP 分组 (n≥1000, {len(g_glob)} 个机场):")
print(f"    tu_med - true_med 中位数 = {float((g_glob.tu_med - g_glob.true_med).median()):+.2f}s")
print(f"    pred_med - true_med 中位数 = {float((g_glob.pred_med - g_glob.true_med).median()):+.2f}s")

step3 = {
    "median_diff_tu_minus_true_median": median_diff,
    "mean_diff_tu_minus_true_median": mean_diff,
    "median_abs_diff": median_abs_diff,
    "n_groups_n>=50": len(g_tu_method),
    "pred_minus_true_median_of_medians": float(g_tu_method.pred_minus_true.median()),
    "pred_minus_true_mean_of_means": float(g_tu_method.pred_minus_true.mean()),
    "tu_minus_true_per_airport_median": float((g_glob.tu_med - g_glob.true_med).median()),
    "pred_minus_true_per_airport_median": float((g_glob.pred_med - g_glob.true_med).median()),
    "judgment": None,
}
if median_diff > 30:
    step3["judgment"] = "基准口径偏高 → 可修 (修法: 训练折重算 t_unimp LUT 分位数或换 baseline 估计)"
elif median_diff < -30:
    step3["judgment"] = "基准口径偏低"
else:
    step3["judgment"] = "基准无系统性偏差, 高估纯来自均值回归 → 后处理修不了, 需更强判别"

# ============================================================
# 总体 verdict
# ============================================================
print()
print("=" * 70); print("[verdict] 短段 bias 归因 总判")
print("=" * 70)
verdicts = []
verdicts.append(f"Step1: {step1_summary['judgment']}")
verdicts.append(f"Step2: {step2['judgment']}")
verdicts.append(f"Step3: {step3['judgment']}")
for v in verdicts:
    print(f"  - {v}")

final_verdict = ""
# Step1 是判据核心: 平移 vs 回归
flat_translation = step1_summary["bias_overall"] > 50  # bias 整体偏正 (>50s, 算"平移型")
is_regression = len(step1) >= 4 and (
    step1[0]["bias"] - step1[-1]["bias"] > 100  # 首档偏正最多, 尾档偏负, 是回归
)
# Step2: 看 (ADEP, RUNWAY) 集中度
ar_pct = step2["n_sig_combinations_by_dim"]["ADEP_RUNWAY"]["n_sig_rows_pct"]
is_concentrated = ar_pct > 30  # > 30% 集中在少数组合 → 可修
# Step3
median_diff_tu = step3["median_diff_tu_minus_true_median"]
tu_is_high = median_diff_tu > 30

if is_regression and not is_concentrated and not tu_is_high:
    final_verdict = ("均值回归 (LGB 不可避免) + 散布均匀 + 基准无偏 → 后处理根本修不了, "
                     "必须用强判别信息 (更多 t_unimp 噪声变量, deep model, 或 sub-model per airport/runway)")
elif is_regression and is_concentrated:
    final_verdict = ("均值回归为主但集中度高 → per-(AP,RWY) bucket 校准 + 训练折偏置对齐可缓解, "
                     "pure post-process 无需改 label/clip 即可做")
elif tu_is_high:
    final_verdict = ("基准口径偏高 → 可修: 训练折 LUT 重建 (其他分位数或更细粒度 baseline)")
elif flat_translation:
    final_verdict = ("平移型 (短段整体偏高), 可能 by per-(AP,RWY) 减 bias → 验算后处理")
else:
    final_verdict = "见 Step1-3 分项判断"
print(f"\n  ★ 总判: {final_verdict}")
step2["judgment"] = final_verdict  # 把总判写到 step2 (overwrite 占位)

# ============================================================
# 落盘
# ============================================================
out = {
    "step1_bins": step1,
    "step1_summary": step1_summary,
    "step2": step2,
    "step3": step3,
    "step3_top_groups_tu_high": g_tu_method.head(20).to_dict("records"),
    "step3_per_ap": g_glob.to_dict("records"),
    "verdict_per_step": verdicts,
    "final_verdict": final_verdict,
    "n_short": int(m_s.sum()),
    "n_core": n_core,
    "red_lines": "不改 label / 不 clip 真值 / 不建单机场模型; 诊断,不拟合常数当发版方案",
    "model_source": "models/lgb_full_v15 (3-seed avg, 71 features)",
    "data": "holdout 2025-01+07 DEP 344,419, short 328,707 (≤1800)",
}
(D0 / "short_bias_diagnose.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print()
print(f"  → 写出: {D0 / 'short_bias_diagnose.json'}")
print(f"  → cache:  {cache_path}")
