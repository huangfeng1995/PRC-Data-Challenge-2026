"""
长滑行结构诊断 (read-only + 复用现有 LGB 模型)
======================================================
只用现有模型 (models/lgb_full_v15/lgb_seed{42,123,2024}.txt) + 标准 holdout
(2025-01+07) 推断, 不重训、不上传、不生成新提交。

最终输出:
  notebooks/diag/d0/long_taxi_structure.json — 全部诊断结果
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

OUT = {}
print("=" * 70); print("[1] 加载 holdout + 训练帧")
print("=" * 70); t0 = time.time()
ho_dfs = []
for m in [1, 7]:
    fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(fn, columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)

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
print(f"  holdout rows: {len(df_ho_raw):,}, train DEP rows: {len(df_tr_dep):,}, {time.time()-t0:.1f}s")

print()
print("=" * 70); print("[2] build_features (holdout + train)")
print("=" * 70); t0 = time.time()
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
y_all = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
core_mask = (y_all > 0) & (y_all <= 10800)
n_core = int(core_mask.sum())
print(f"  full feature done in {time.time()-t0:.1f}s, holdout DEP {len(df_ho_dep_full):,}, core {n_core:,}")

print()
print("=" * 70); print("[3] 复用 LGB 3-seed 模型推理")
print("=" * 70); t0 = time.time()

cat_map_path = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
cat_map = json.loads(cat_map_path.read_text())

X_ho = df_ho_dep_full[FEATS71].copy()
for col in HIGH_CARD_CATS:
    m = cat_map.get(col, {})
    X_ho[col] = X_ho[col].astype(str).map(m).fillna(-1).astype("int32")
for col in LOW_CARD_CATS:
    X_ho[col] = X_ho[col].astype("category")

boosters = []
for s in [42, 123, 2024]:
    fn = ROOT / "models" / "lgb_full_v15" / f"lgb_seed{s}.txt"
    b = lgb.Booster(model_file=str(fn))
    boosters.append(b)
print(f"  loaded {len(boosters)} LGB boosters from models/lgb_full_v15/")

pred_seed = []
for b in boosters:
    p = b.predict(X_ho)
    pred_seed.append(p)
pred_all = np.mean(np.stack(pred_seed), axis=0)
pred_all = np.clip(pred_all, 30, 10800).astype(np.float32)
print(f"  predict done, {time.time()-t0:.1f}s")

print()
print("=" * 70); print("[4] 误差分档 (核心口径)")
print("=" * 70)
err2 = (pred_all - y_all) ** 2
y_c = y_all[core_mask]; e_c = err2[core_mask]; p_c = pred_all[core_mask]
aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
aps_c = aps[core_mask]
ss_total_c = float(e_c.sum()); rmse_global_c = float(np.sqrt(np.mean(e_c)))

print(f"  baseline (core holdout {n_core:,}) RMSE = {rmse_global_c:.2f}")
print(f"  total SSE = {ss_total_c:.3e}")
print(f"  pred mean = {p_c.mean():.1f}, true mean = {y_c.mean():.1f}, bias = {np.mean(p_c - y_c):+.2f}")

bins = [(0, 1800, "(0,1800]"),
        (1800, 3600, "(1800,3600]"),
        (3600, 7200, "(3600,7200]"),
        (7200, 10800, "(7200,10800]")]
sse_table = []
for lo, hi, lab in bins:
    m = (y_c > lo) & (y_c <= hi)
    n = int(m.sum())
    if n == 0:
        continue
    rmse_b = float(np.sqrt(np.mean(e_c[m])))
    sse = float(e_c[m].sum())
    pct = float(sse / ss_total_c * 100)
    sse_table.append({"range": lab, "n": n, "pct_rows": float(n / n_core * 100),
                      "sse": sse, "pct_sse": pct, "rmse_bin": rmse_b})
    print(f"  {lab:18s}  n={n:>7,} ({n/n_core*100:>5.1f}%) "
          f"SSE={sse:>13.3e} (SSE%={pct:>5.1f}%) binRMSE={rmse_b:.1f}")
OUT["sse_by_bin"] = sse_table
OUT["n_core"] = n_core

# sensitivity: 假设长滑行误差降到 X
print()
print("  敏感性测算 (长滑行误差降到 1500 / 1000 时) →")
mean_y_all = float(np.mean(y_all))
# 把长滑行 (>1800) 视为一个块: 行数 = n_long
m_long = (y_c > 1800)
n_long = int(m_long.sum())
ss_long = float(e_c[m_long].sum())
ss_short = ss_total_c - ss_long
n_short = n_core - n_long

for tgt in [1500, 1000, 800]:
    new_ss_long = n_long * (tgt ** 2)
    new_ss_total = ss_short + new_ss_long
    new_rmse = float(np.sqrt(new_ss_total / n_core))
    print(f"    若长滑行 RMSEP 降到 {tgt}: core RMSE ≈ {new_rmse:.2f} "
          f"(Δ={new_rmse - rmse_global_c:+.2f})")
OUT["sensitivity"] = {"target_rmse": [1500, 1000, 800],
                       "expected_core_rmse": []}
for tgt in [1500, 1000, 800]:
    new_ss_long = n_long * (tgt ** 2)
    new_rmse = float(np.sqrt((ss_short + new_ss_long) / n_core))
    OUT["sensitivity"]["expected_core_rmse"].append(
        {"long_rmse_target": tgt, "core_rmse": new_rmse,
         "delta": new_rmse - rmse_global_c})

print()
print("=" * 70); print("[5] 长滑行 (>1800) 类型打标")
print("=" * 70)
m_long_all = y_all > 1800  # 用完整集, 不仅核心
n_long_all = int(m_long_all.sum())
print(f"  长滑行总行数 (全样本, 不限核心/arr): {n_long_all:,}")
print(f"  其中 ARR 滑行: {int(((df_ho_dep_full['PHASE_mvt'] != 'DEP') & m_long_all).sum()):,} "
      f"→ 仅 DEP 进入打标")
dep_long_mask = (df_ho_dep_full["PHASE_mvt"] == "DEP").values & m_long_all
n_long_dep = int(dep_long_mask.sum())
print(f"  DEP 长滑行: {n_long_dep:,}")

# 构造时间戳
def _to_ns(s):
    return pd.to_datetime(s, utc=True, errors="coerce").values.astype("datetime64[ns]").view("i8")
mvt_ns = _to_ns(df_ho_dep_full["MVT_TIME_UTC_mvt"])
block_ns = _to_ns(df_ho_dep_full["BLOCK_TIME_UTC_mvt"])
sched_ns = _to_ns(df_ho_dep_full["SCHED_TIME_UTC_mvt"])
aobt_ns = _to_ns(df_ho_dep_full["AOBT_3_flt"])
iNaT = np.iinfo(np.int64).min
valid = lambda a: a != iNaT

# type_S: |BLOCK - SCHED| <= 60s
type_S = dep_long_mask & valid(block_ns) & valid(sched_ns) & (np.abs(block_ns - sched_ns) <= 60)
# type_D: 跨天错误: TAXITIME - (MVT - SCHED) 与 86400 差不到 600s
# 数据自检: TAX = MVT - BLOCK (taxi-out), 如果 TAX > 7200 且与 (MVT-SCHED)+86400 接近 → 把 SCHED 当 BLOCK 错位 1 天
mvt_minus_sched = (mvt_ns - sched_ns) / 1e9  # 秒
# 真实 TAXITIME - (MVT-SCHED) 应该是 MVT - BLOCK = TAXITIME 直接差 0；这里检查
# 跨天: |TAXITIME - (MVT - SCHED - 86400)| < 600
guess_minus_1day = (mvt_ns - sched_ns) / 1e9 - 86400
type_D = dep_long_mask & valid(mvt_ns) & valid(sched_ns) & (np.abs(y_all - guess_minus_1day) < 600)
# 也用 +86400 一并检查
guess_plus_1day = (mvt_ns - sched_ns) / 1e9 + 86400
type_D_other = dep_long_mask & valid(mvt_ns) & valid(sched_ns) & (np.abs(y_all - guess_plus_1day) < 600)
type_D = type_D | type_D_other
# type_N: |AOBT - BLOCK| > 30min
type_N = dep_long_mask & valid(aobt_ns) & valid(block_ns) & (np.abs(aobt_ns - block_ns) > 30*60*1e9)
# type_H: 其余长滑行（即 "正常但长"）
type_excl = type_S | type_D | type_N
type_H = dep_long_mask & ~type_excl
# neither: TAXITIME 远大于 MVT - AOBT (说明先 AOBT 推出后才开始算 TAXITIME / 含 AOBT 后等待)
mvt_minus_aobt = np.where(valid(mvt_ns) & valid(aobt_ns), (mvt_ns - aobt_ns) / 1e9, np.nan)
neither = dep_long_mask & valid(mvt_ns) & valid(aobt_ns) & (y_all < (mvt_minus_aobt - 600))

# 多标优先级: 互斥化: S > D > N > neither > H
labels = np.array(["H"] * len(y_all), dtype=object)
labels[type_S] = "S"
labels[type_D & ~type_S] = "D"
labels[type_N & ~type_S & ~type_D] = "N"
labels[neither & ~type_S & ~type_D & ~type_N] = "neither"
labels[type_H & ~type_S & ~type_D & ~type_N & ~neither] = "H"
labels_long = labels[dep_long_mask]

count_S = int((labels_long == "S").sum())
count_D = int((labels_long == "D").sum())
count_N = int((labels_long == "N").sum())
count_neither = int((labels_long == "neither").sum())
count_H = int((labels_long == "H").sum())
print(f"    type_S (BLOCK≈SCHED, ±60s): {count_S:,}  ({count_S/n_long_dep*100:.1f}%)")
print(f"    type_D (跨天错位 ±600s):       {count_D:,}  ({count_D/n_long_dep*100:.1f}%)")
print(f"    type_N (AOBT vs BLOCK >30min): {count_N:,}  ({count_N/n_long_dep*100:.1f}%)")
print(f"    type_H (其余长滑行):           {count_H:,}  ({count_H/n_long_dep*100:.1f}%)")
print(f"    neither (TAX<MVT-AOBT-600):    {count_neither:,}  ({count_neither/n_long_dep*100:.1f}%)")
OUT["type_counts"] = {"S": count_S, "D": count_D, "N": count_N,
                       "H": count_H, "neither": count_neither,
                       "total_long_dep": n_long_dep}

print()
print("=" * 70); print("[6] 分机场类型分布")
print("=" * 70)
df_long = df_ho_dep_full[dep_long_mask].copy().reset_index(drop=True)
df_long["type"] = labels_long
df_long["pred"] = pred_all[dep_long_mask]
df_long["true"] = y_all[dep_long_mask]
df_long["err2"] = (df_long["pred"] - df_long["true"]) ** 2
df_long["airport"] = df_long["ADEP_mvt"].astype(str)

airport_table = []
for ap in sorted(df_long["airport"].unique()):
    sub = df_long[df_long["airport"] == ap]
    n = len(sub)
    cnt_S = int((sub["type"] == "S").sum())
    cnt_D = int((sub["type"] == "D").sum())
    cnt_N = int((sub["type"] == "N").sum())
    cnt_H = int((sub["type"] == "H").sum())
    cnt_neither = int((sub["type"] == "neither").sum())
    median = float(sub["true"].median())
    p95 = float(sub["true"].quantile(0.95))
    p99 = float(sub["true"].quantile(0.99))
    mx = float(sub["true"].max())
    rmse_ap = float(np.sqrt(sub["err2"].mean()))
    airport_table.append({
        "airport": ap, "n": n,
        "S": cnt_S, "D": cnt_D, "N": cnt_N, "H": cnt_H, "neither": cnt_neither,
        "pct_S": cnt_S / n * 100, "pct_D": cnt_D / n * 100,
        "pct_N": cnt_N / n * 100, "pct_H": cnt_H / n * 100,
        "pct_neither": cnt_neither / n * 100,
        "median": median, "p95": p95, "p99": p99, "max": mx,
        "rmse_bin": rmse_ap,
    })
print(f"  {'ADEP':6s} {'n':>6s} {'S%':>6s} {'D%':>6s} {'N%':>6s} {'H%':>6s} {'oth%':>6s} "
      f"{'med':>6s} {'p95':>6s} {'p99':>6s} {'max':>6s} {'RMSE':>7s}")
for r in airport_table:
    print(f"  {r['airport']:6s} {r['n']:>6d} "
          f"{r['pct_S']:>5.1f}% {r['pct_D']:>5.1f}% {r['pct_N']:>5.1f}% "
          f"{r['pct_H']:>5.1f}% {r['pct_neither']:>5.1f}% "
          f"{r['median']:>6.0f} {r['p95']:>6.0f} {r['p99']:>6.0f} {r['max']:>6.0f} "
          f"{r['rmse_bin']:>7.1f}")
OUT["per_airport_long_taxi"] = airport_table

print()
print("=" * 70); print("[7] 可预测性初判")
print("=" * 70)

# 给每个类型与信号的关系: 算「同跑道起飞间隔」/「active runways (近似)」/
# 「流量调控 (LOBT/EOBT 后移)」/「除冰天气 (雪)」
# 由于 LOBT/EOBT/IOBT/SCHED 已构造在特征里, 取对应特征值
# 这里用各类型样本在 X 上的分布 vs 全样本长滑行的分布 → 比对差
def _stat_col(col, arr):
    return {"mean": float(np.nanmean(arr)), "median": float(np.nanmedian(arr)),
            "q25": float(np.nanpercentile(arr, 25)), "q75": float(np.nanpercentile(arr, 75))}

# 取部分关键特征 (在 X_ho 上对长滑行取切片)
X_long = X_ho[dep_long_mask].copy().reset_index(drop=True)
df_long["mvt_n_ahead"] = X_long["mvt_n_ahead"].values
df_long["dep_cnt_m_15"] = X_long["dep_cnt_m_15"].values
df_long["dly_lobt_aobt"] = X_long["dly_lobt_aobt"].values
df_long["dly_aobt_eobt"] = X_long["dly_aobt_eobt"].values
df_long["rwy_month_share"] = X_long["rwy_month_share"].values
df_long["mvt_q_peak"] = X_long["mvt_q_peak"].values
df_long["mvt_na_heavy_frac"] = X_long["mvt_na_heavy_frac"].values

predictability = {}
for t in ["S", "D", "N", "H", "neither"]:
    sub = df_long[df_long["type"] == t]
    n = len(sub)
    if n == 0:
        predictability[t] = {"n": 0}
        continue
    pp = {"n": n,
          "mvt_n_ahead": _stat_col("n", sub["mvt_n_ahead"].values),
          "dep_cnt_m_15": _stat_col("d", sub["dep_cnt_m_15"].values),
          "dly_lobt_aobt": _stat_col("dla", sub["dly_lobt_aobt"].values),
          "dly_aobt_eobt": _stat_col("dla", sub["dly_aobt_eobt"].values),
          "rwy_month_share": _stat_col("rwy", sub["rwy_month_share"].values),
          "mvt_q_peak": _stat_col("qp", sub["mvt_q_peak"].values),
          "mvt_na_heavy_frac": _stat_col("hf", sub["mvt_na_heavy_frac"].values),
          "rmse_bin": float(np.sqrt(sub["err2"].mean())),
          "bin_tax_median": float(sub["true"].median()),
          "bin_tax_p95": float(sub["true"].quantile(0.95)),
          }
    print(f"  type={t:7s} n={n:>5d}  RMSE={pp['rmse_bin']:.0f}  "
          f"n_ahead med={pp['mvt_n_ahead']['median']:.1f}  "
          f"queue_peak med={pp['mvt_q_peak']['median']:.2f}  "
          f"dly_l_a med={pp['dly_lobt_aobt']['median']:.0f}s")
predictability["all_long"] = {"n": n_long_dep,
                               "mvt_n_ahead": _stat_col("n", df_long["mvt_n_ahead"].values),
                               "mvt_q_peak": _stat_col("qp", df_long["mvt_q_peak"].values),
                               "rmse_bin": float(np.sqrt(df_long["err2"].mean()))}
OUT["predictability"] = predictability

# === 简评 ===
print()
print("  评审:")
assessments = {
    "S": "BLOCK≈SCHED（计划当真实）：包含大量实际滑行时间被官方TAXITIME字段记为长滑行，"
         "但实际上属于『准点起飞、计划即真』——可建模为机场-跑道-时段基线 + 流量修正；预测应接近 t_unimp_r7",
    "D": "跨天错位（+86400 / SCHED错位）：TAXITIME 字段本身是错位的，预测误差基本是数据 bug；"
         "要么截断到 ≤10800，要么重新解析 MVT-AOBT；可控性取决于数据修正",
    "N": "AOBT - BLOCK > 30min：推出远晚于计划，典型场景是地面延误 / 拖车 / 等舱单 / 等推出；"
         "预测可结合 dep_cnt_m / eobt_ahead_cnt / rwy_month_share / ap_month_hourly_density，"
         "但真正的『何时推出』是难信号",
    "H": "其余长滑行（计划起算 + 推出正常）：核心难点；可建模线索 mvt_q_peak / heavy_frac / 流量/天气；"
         "在 1月+7月 holdout 上大部分 TAXITIME 集中在 30-60 min 这部分预测应还行，但 >30min 的尾部才是分数战场",
    "neither": "TAX < MVT - AOBT - 600 (含 AOBT 后等起飞的等待): 与 type_N 类似 + 等待空地，"
         "实际是 TAXITIME 字段被低估；同样属于『特征缺陷』层面",
}
for t, msg in assessments.items():
    print(f"    {t}: {msg}")
OUT["type_assessment"] = assessments

# === 灵敏度与建议 ===
print()
print("=" * 70)
print("[8] 结论摘要")
print("=" * 70)
top_sse = sorted(sse_table, key=lambda x: -x["pct_sse"])[:3]
print(f"  TOP-3 SSE 贡献分档:")
for r in top_sse:
    print(f"    {r['range']:18s}  SSE%={r['pct_sse']:>5.1f}%  n%={r['pct_rows']:>5.1f}%  binRMSE={r['rmse_bin']:.1f}")
print()
print(f"  长滑行 ({n_long_dep:,} 行, 占 {n_long_dep/n_core*100:.2f}%) 贡献了 {(sum(r['sse'] for r in sse_table if r['range'] != '(0,1800]'))/ss_total_c*100:.1f}% 的 SSE")
print(f"  若长滑行 RMSE 降到 1500, core RMSE = {OUT['sensitivity']['expected_core_rmse'][0]['core_rmse']:.2f}")
print(f"  若长滑行 RMSE 降到 1000, core RMSE = {OUT['sensitivity']['expected_core_rmse'][1]['core_rmse']:.2f}")
print(f"  若长滑行 RMSE 降到  800, core RMSE = {OUT['sensitivity']['expected_core_rmse'][2]['core_rmse']:.2f}")

OUT["base_rmse_core"] = rmse_global_c
OUT["n_long_dep"] = n_long_dep
OUT["summary"] = {
    "base_rmse_core": rmse_global_c,
    "n_long_dep": n_long_dep,
    "pct_long_dep": n_long_dep / n_core * 100,
    "long_taxi_total_sse_pct": (sum(r["sse"] for r in sse_table if r["range"] != "(0,1800]")) / ss_total_c * 100,
    "rmse_if_long_1500": OUT["sensitivity"]["expected_core_rmse"][0]["core_rmse"],
    "rmse_if_long_1000": OUT["sensitivity"]["expected_core_rmse"][1]["core_rmse"],
    "rmse_if_long_0800": OUT["sensitivity"]["expected_core_rmse"][2]["core_rmse"],
}

out_path = D0 / "long_taxi_structure.json"
out_path.write_text(json.dumps(OUT, indent=2, ensure_ascii=False))
print()
print(f"  → 写入: {out_path}")
