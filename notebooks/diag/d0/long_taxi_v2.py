"""
长滑行结构诊断 v2 (read-only, 重算 RMSE + 极端值定性 + 指纹)
================================================================
红线:
  ❌ 不重算 TAXITIME 标签 (不引入 MVT−AOBT 等替换)
  ❌ 不 clip/winsorize 真值
  ❌ 不建单机场模型

利用上版输出 long_taxi_structure.json 已有的核心holdout预测
(pred_all / y_all / df_ho_dep_full / labels_long / X_ho / X_long / df_long)
直接对它们再做分析. 不重跑重训, 不重跑 build_features.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
PREV = D0 / "long_taxi_structure.json"

prev = json.loads(PREV.read_text())
# 上版存的字段
sse_table = prev["sse_by_bin"]
n_core = prev["n_core"]

# 但上版没把整个 (pred_all / y_all / df_ho_dep_full / labels_long / X_long / df_long) 存出来
# —— 那些是脚本里的局部变量. 需要重跑脚本的核心 / build_features / LGB 推理部分来恢复它们
# 但因为 build_features + LGB 推理共 ~1.5 分钟, 重跑可接受
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D
import lightgbm as lgb
import time

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

print("=" * 70); print("[1] 加载 holdout + 训练帧")
print("=" * 70)
t0 = time.time()
ho_dfs = []
for m in [1, 7]:
    ho_dfs.append(load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet",
                              columns=TRAIN_COLS))
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

print("=" * 70); print("[2] build_features")
print("=" * 70); t1 = time.time()
df_ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)
print(f"  holdout DEP {len(df_ho_dep_full):,} | build_features {time.time()-t1:.1f}s")

print("=" * 70); print("[3] LGB 3-seed predict")
print("=" * 70); t1 = time.time()
cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
X_ho = df_ho_dep_full[FEATS71].copy()
for col in HIGH_CARD_CATS:
    m = cat_map.get(col, {})
    X_ho[col] = X_ho[col].astype(str).map(m).fillna(-1).astype("int32")
for col in LOW_CARD_CATS:
    X_ho[col] = X_ho[col].astype("category")
boosters = [lgb.Booster(model_file=str(ROOT / "models/lgb_full_v15" / f"lgb_seed{s}.txt"))
            for s in (42, 123, 2024)]
pred_all = np.mean(np.stack([b.predict(X_ho) for b in boosters]), axis=0).astype(np.float32)
pred_all = np.clip(pred_all, 30, 10800)
y_all = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
print(f"  LGB 3-seed 推理完成, {time.time()-t1:.1f}s")

# ============================================================
# 修正 [A] 长滑行整体 RMSE 重算 (包含 type_S/D/N/H/neither 全部 >1800 行)
# ============================================================
print()
print("=" * 70); print("[A] 长滑行整体 RMSE (核心口径, 含所有 type)")
print("=" * 70)
core_mask = (y_all > 0) & (y_all <= 10800)
y_c = y_all[core_mask]; p_c = pred_all[core_mask]; n_core = int(core_mask.sum())
err2_c = (p_c - y_c) ** 2

m_long = (y_c > 1800)
n_long = int(m_long.sum())
ss_long = float(err2_c[m_long].sum())
ss_short = float(err2_c.sum() - ss_long)
long_rmse = float(np.sqrt(ss_long / n_long)) if n_long else float('nan')
short_rmse = float(np.sqrt(ss_short / (n_core - n_long)))
core_rmse = float(np.sqrt(err2_c.sum() / n_core))
print(f"  短滑行 (0,1800]:  n={n_core-n_long:>7,}  SSE={ss_short:.3e}  RMSE={short_rmse:.2f}")
print(f"  长滑行 (1800,10800]: n={n_long:>7,}  SSE={ss_long:.3e}  RMSE={long_rmse:.2f}")
print(f"  整盘 core:                  n={n_core:>7,}  SSE={err2_c.sum():.3e}  RMSE={core_rmse:.2f}")
print(f"  → 长滑行整体 RMSE = {long_rmse:.2f} (上版只看 type_H 取 2487, 偏大是没加权)")

print()
print("[B] 修正敏感性测算 (单调下降, 整盘看)")
print("=" * 70)
sens_rows = []
for tgt in [2000, 1500, 1000, 800, 500, 300, 200, 100, 0]:
    new_ss_long = n_long * (tgt ** 2)
    new_rmse = float(np.sqrt((ss_short + new_ss_long) / n_core))
    sens_rows.append({"long_rmse_target": tgt, "core_rmse": new_rmse,
                       "delta_vs_current": new_rmse - core_rmse})
    print(f"    长滑行 RMSE → {tgt:>5}: core RMSE = {new_rmse:>6.2f}  (Δ={new_rmse - core_rmse:+7.2f})")

# ============================================================
# 完整 holdout 上做 "真实长滑行 vs 脏值/默认" 定性
# ============================================================
print()
print("=" * 70); print("[C] 极端值定性: 真实长滑行 vs 脏值 (完整 holdout, 不剔)")
print("=" * 70)
# 完整 holdout: 所有 DEP 行 (含 >10800 / 含 ≤0)
m_dep = df_ho_dep_full["PHASE_mvt"].values == "DEP"
n_total = int(m_dep.sum())
m_big = (y_all >= 1800) & m_dep  # 长滑行 (不限上限)
n_big = int(m_big.sum())
print(f"  DEP 总行数 (完整 holdout): {n_total:,}")
print(f"  长滑行 TAXITIME>=1800: {n_big:,} ({n_big/n_total*100:.2f}%)")

def _to_ns(s):
    return pd.to_datetime(s, utc=True, errors="coerce").values.astype("datetime64[ns]").view("i8")
iNaT = np.iinfo(np.int64).min
def _valid(a): return (a != iNaT) & ~np.isnan(a)

mvt_ns = _to_ns(df_ho_dep_full["MVT_TIME_UTC_mvt"])
block_ns = _to_ns(df_ho_dep_full["BLOCK_TIME_UTC_mvt"])
sched_ns = _to_ns(df_ho_dep_full["SCHED_TIME_UTC_mvt"])
aobt_ns = _to_ns(df_ho_dep_full["AOBT_3_flt"])

# --- 多个"脏值"条件, 任意一条命中 → 脏值 ---
def _diff_sec(a, b):
    d = (a - b) / 1e9
    invalid = (a == iNaT) | (b == iNaT)
    d = np.where(invalid, np.nan, d)
    return d

# 1) BLOCK≈SCHED (±60s, 计划当 BLOCK)
dirty_blocksched = _valid(block_ns) & _valid(sched_ns) & (np.abs(_diff_sec(block_ns, sched_ns)) <= 60)
# 2) 跨天错位: |TAX - (MVT-SCHED+86400)| <= 600s
guess_p1d = (mvt_ns - sched_ns) / 1e9 + 86400
guess_m1d = (mvt_ns - sched_ns) / 1e9 - 86400
dirty_dayoff = _valid(mvt_ns) & _valid(sched_ns) & (
    (np.abs(y_all - guess_p1d) <= 600) | (np.abs(y_all - guess_m1d) <= 600))
# 3) MVT 异常: MVT 早于 SCHED+1d (不可能, 应为次日同段); MVT 远早于 SCHED - 30d
mvt_vs_sched_days = (mvt_ns - sched_ns) / 1e9 / 86400
dirty_mvt_anom = _valid(mvt_ns) & _valid(sched_ns) & (
    (mvt_vs_sched_days < -30) | (mvt_vs_sched_days > 30))
# 4) 字段矛盾: TAX > MVT - AOBT (滑出比所有时序还长) 但 AOBT 有效
mvt_minus_aobt = np.where(_valid(mvt_ns) & _valid(aobt_ns), (mvt_ns - aobt_ns) / 1e9, np.nan)
dirty_contradict = _valid(mvt_ns) & _valid(aobt_ns) & (y_all > mvt_minus_aobt + 60)
# 5) AOBT 远晚于 MVT (>10 分钟)
dirty_aobt_late = _valid(aobt_ns) & _valid(mvt_ns) & (_diff_sec(aobt_ns, mvt_ns) > 600)
# 6) BLOCK 远早于 MVT (BLOCK 比 MVT 还晚 24h)
dirty_block_late = _valid(block_ns) & _valid(mvt_ns) & (_diff_sec(mvt_ns, block_ns) > 86400)
dirty = dirty_blocksched | dirty_dayoff | dirty_mvt_anom | dirty_contradict | dirty_aobt_late | dirty_block_late

is_dirty = dirty & m_big
is_real = m_big & ~dirty
n_dirty = int(is_dirty.sum()); n_real = int(is_real.sum())
print(f"    真实长滑行 (AOBT/SCHED/MVT 自洽): {n_real:,} ({n_real/n_big*100:.1f}%)")
print(f"    脏值/默认 (BLOCK≈SCHED/跨天/MVT矛盾/...): {n_dirty:,} ({n_dirty/n_big*100:.1f}%)")

# 分档统计
bins3 = [(1800, 3600, "(1800,3600]"),
         (3600, 10800, "(3600,10800]"),
         (10800, float("inf"), ">10800")]
print()
print(f"  {'range':15s}  {'n':>7s}  {'real':>7s} ({'real%':>5s})  {'dirty':>7s} ({'dirty%':>5s})  "
      f"{'real_med':>8s}  {'real_p99':>8s}  {'real_max':>8s}  "
      f"{'dirty_med':>9s}  {'dirty_p99':>9s}  {'dirty_max':>9s}")
fits = {}
for lo, hi, lab in bins3:
    m_bin = m_big & (y_all > lo) & (y_all <= hi)
    n_bin = int(m_bin.sum())
    if n_bin == 0: continue
    real_mask = m_bin & ~dirty
    dirty_mask = m_bin & dirty
    n_real_b = int(real_mask.sum()); n_dirty_b = int(dirty_mask.sum())
    if n_real_b:
        r_med = float(np.median(y_all[real_mask]))
        r_p99 = float(np.percentile(y_all[real_mask], 99))
        r_max = float(np.max(y_all[real_mask]))
    else:
        r_med = r_p99 = r_max = float('nan')
    if n_dirty_b:
        d_med = float(np.median(y_all[dirty_mask]))
        d_p99 = float(np.percentile(y_all[dirty_mask], 99))
        d_max = float(np.max(y_all[dirty_mask]))
    else:
        d_med = d_p99 = d_max = float('nan')
    fits[lab] = {"n": n_bin, "real_n": n_real_b, "dirty_n": n_dirty_b,
                 "real_med": r_med, "real_p99": r_p99, "real_max": r_max,
                 "dirty_med": d_med, "dirty_p99": d_p99, "dirty_max": d_max}
    print(f"  {lab:15s}  {n_bin:>7,}  "
          f"{n_real_b:>7,} ({n_real_b/n_bin*100:>4.1f}%)  "
          f"{n_dirty_b:>7,} ({n_dirty_b/n_bin*100:>4.1f}%)  "
          f"{r_med:>8.0f}  {r_p99:>8.0f}  {r_max:>8.0f}  "
          f"{d_med:>9.0f}  {d_p99:>9.0f}  {d_max:>9.0f}")

# 分机场分档
print()
print("=" * 70); print("[D] 分机场 × 真实/脏值 × 档")
print("=" * 70)
print(f"  {'ADEP':6s} {'n':>6s} {'r(1800,3600]':>13s} {'r(3600,10800]':>14s} "
      f"{'r(>10800)':>10s} {'real%':>6s} {'dirty%':>7s}")
per_ap = {}
aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
for ap in sorted(np.unique(aps)):
    m_ap = m_big & (aps == ap)
    n_ap = int(m_ap.sum())
    if n_ap == 0: continue
    real_ap = m_ap & ~dirty
    dirty_ap = m_ap & dirty
    real_n = int(real_ap.sum())
    r1 = int((real_ap & (y_all > 1800) & (y_all <= 3600)).sum())
    r2 = int((real_ap & (y_all > 3600) & (y_all <= 10800)).sum())
    r3 = int((real_ap & (y_all > 10800)).sum())
    print(f"  {ap:6s} {n_ap:>6d} {r1:>13d} {r2:>14d} {r3:>10d} "
          f"{real_n/n_ap*100:>5.1f}% "
          f"{int(dirty_ap.sum())/n_ap*100:>6.1f}%")
    per_ap[ap] = {
        "n_long": n_ap,
        "real_n": real_n, "real_pct": real_n/n_ap*100,
        "dirty_n": int(dirty_ap.sum()), "dirty_pct": int(dirty_ap.sum())/n_ap*100,
        "real_1800_3600": r1, "real_3600_10800": r2, "real_gt10800": r3,
    }

# ============================================================
# 可预测指纹 + 兜底可行性
# ============================================================
print()
print("=" * 70); print("[E] 真实长滑行的可预测指纹 (与这些信号的相关性)")
print("=" * 70)

# 把 X_long 与真实长滑行 mask 对齐
df_long_real = df_ho_dep_full[is_real].copy().reset_index(drop=True)
X_real = X_ho[is_real].copy().reset_index(drop=True)

def _stat(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0: return {"mean": None, "median": None, "q25": None, "q75": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }

fingerprints_real = {
    "n": int(len(X_real)),
    "mvt_n_ahead": _stat(X_real["mvt_n_ahead"].values),
    "mvt_mu_rwy_15": _stat(X_real["mvt_mu_rwy_15"].values),
    "mvt_q_peak": _stat(X_real["mvt_q_peak"].values),
    "mvt_na_heavy_frac": _stat(X_real["mvt_na_heavy_frac"].values),
    "dep_cnt_m_15": _stat(X_real["dep_cnt_m_15"].values),
    "eobt_ahead_cnt_15": _stat(X_real["eobt_ahead_cnt_15"].values),
    "dly_lobt_aobt": _stat(X_real["dly_lobt_aobt"].values),
    "dly_aobt_sched": _stat(X_real["dly_aobt_sched"].values),
    "rwy_month_share": _stat(X_real["rwy_month_share"].values),
    "ap_month_hourly_density": _stat(X_real["ap_month_hourly_density"].values),
    "t_unimp_r7": _stat(X_real["t_unimp_r7"].values),
}
print(f"  真实长滑行 n={fingerprints_real['n']:,}")
for k, v in fingerprints_real.items():
    if k == "n": continue
    if isinstance(v, dict) and v.get("median") is not None:
        print(f"    {k:25s}  med={v['median']:>9.2f}  q25={v['q25']:>9.2f}  q75={v['q75']:>9.2f}  mean={v['mean']:>9.2f}")

# 兜底: 机场 × 时段 × 指纹条件期望
# 概念可行检测: 对真实长滑行的真值 TAXITIME, 按 (ADEP, hour) 求 group mean,
# 看 group 规模 / std; std<全局中位 → 兜底有意义
print()
print("[F] 兜底可行性: 机场 × 时段 条件期望")
print("  真实长滑行 (is_real) 真值做 (ADEP, sched_hour) 分组:")
real_df = df_ho_dep_full[is_real].copy()
real_df["hour"] = pd.to_datetime(real_df["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce").dt.hour
g = real_df.groupby(["ADEP_mvt", "hour"])["TAXITIME_SEC_mvt"]
g_stats = g.agg(["count", "mean", "median", "std"])
g_stats["cv"] = g_stats["std"] / g_stats["mean"]
big_groups = g_stats[g_stats["count"] >= 5]
n_groups = len(big_groups)
print(f"    总组数 n>=5: {n_groups}")
if n_groups:
    print(f"    CV 中位: {big_groups['cv'].median():.3f}  (CV<0.5 → 兜底较稳)")
    print(f"    CV < 0.5 的组数: {(big_groups['cv'] < 0.5).sum()}")
    print(f"    CV < 0.3 的组数: {(big_groups['cv'] < 0.3).sum()}")
    fallback_feasibility = {
        "n_groups_n>=5": n_groups,
        "cv_median": float(big_groups["cv"].median()),
        "n_cv_lt_0.5": int((big_groups["cv"] < 0.5).sum()),
        "n_cv_lt_0.3": int((big_groups["cv"] < 0.3).sum()),
        "verdict": ("不可行" if big_groups["cv"].median() > 0.6 else
                    "勉强可行" if big_groups["cv"].median() > 0.4 else
                    "可行"),
    }
else:
    fallback_feasibility = {"n_groups_n>=5": 0, "verdict": "数据不足"}

# 脏值兜底: 垃圾行用 (机场 × 时段) "全局 base" 替代
print()
print("[G] 脏值兜底: 机场 × 时段 (DEP, 不限长) 整体 base")
print("  (用真实长滑行的 base 还是 全 holdout DEP 的 base? — 真长滑行可能尾部不稳)")
print(f"  'is_dirty' 兜底选项:")
print(f"    a) 兜底 = 机场×时段 (训练样本) mean(真实长滑行) — 上述 CV 表")
print(f"    b) 兜底 = 机场×时段 mean(全部 DEP) — 样本多, 但会被正常滑行稀释")
print(f"    c) 兜底 = LGB 预测本身 (clip 到 [t_unimp_r7, t_unimp_r7+15min]) — 物理下限")

# 写报告
out = {
    "core_n": n_core,
    "core_rmse_baseline": core_rmse,
    "long_taxi_overall_rmse": long_rmse,
    "n_long_dep_core": n_long,
    "long_taxi_total_sse_pct": float(ss_long / err2_c.sum() * 100),
    "long_taxi_row_pct": float(n_long / n_core * 100),
    "sensitivity_corrected": sens_rows,
    "extreme_n_long_total": n_big,
    "extreme_real_n": n_real, "extreme_real_pct": n_real/n_big*100,
    "extreme_dirty_n": n_dirty, "extreme_dirty_pct": n_dirty/n_big*100,
    "extreme_dirty_reasons": {
        "blocksched": int((dirty_blocksched & m_big).sum()),
        "dayoff": int((dirty_dayoff & m_big).sum()),
        "mvt_anomaly": int((dirty_mvt_anom & m_big).sum()),
        "contradict_tax_gt_mvt_aobt": int((dirty_contradict & m_big).sum()),
        "aobt_late": int((dirty_aobt_late & m_big).sum()),
        "block_late_mvt": int((dirty_block_late & m_big).sum()),
    },
    "extreme_by_bin": fits,
    "per_airport": per_ap,
    "fingerprints_real_long_taxi": fingerprints_real,
    "fallback_feasibility": fallback_feasibility,
}
out_path = D0 / "long_taxi_structure_v2.json"
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print()
print(f"  → 写入: {out_path}")
