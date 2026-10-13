"""
AOBT 物理证据: 长滑行 v4 定性 (read-only)
===========================================
红线: 不改 label / 不 clip 真值 / 不建单机场模型

AOBT (Actual Off-Block Time) = 飞机实际离开登机口的时间，是 OPCDM 里物理时间戳质量最好的字段之一。
本节用 AOBT 把长滑行 (>=1800s) 划三档:
  - real_v4: AOBT 有效 AND |AOBT-BLOCK| <= 300s (5 min, 物理推出窗口) AND MVT-AOBT>0
              → 可建模
  - dirty_v4: (AOBT 缺失) OR (|AOBT-BLOCK| > 1800s = 30 min, 推出严重偏离 BLOCK → 物理不可能 → 字段冲突/污染)
              → 不兜底也不建模
  - uncertain_v4: 介于两者之间 (AOBT 缺失但其他时间自洽 或 |AOBT-BLOCK| 在 5-30min 之间)
              → 等补充数据再决定

另: 核对 LIRF v3 BLOCK==SCHED (within 5s) & MVT-SCHED>45min 行数, 与 v3 LIRF 脏值数对照.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
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

print("=" * 70); print("[1] load + build_features + LGB predict")
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

df_ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"]=="DEP"].copy().reset_index(drop=True)

cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
X_ho = df_ho_dep_full[FEATS71].copy()
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt",
                  "AIRCRAFT_OPERATOR_flt"]
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
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
print(f"  全流程 {time.time()-t0:.1f}s | DEP {len(df_ho_dep_full):,}")

# ============================================================
# 派生
# ============================================================
def _to_ns(s):
    return pd.to_datetime(s, utc=True, errors="coerce").values.astype("datetime64[ns]").view("i8")
iNaT = np.iinfo(np.int64).min
def _v(a): return (a != iNaT) & ~np.isnan(a)
def _diff(a, b):
    d = (a - b) / 1e9
    invalid = (a == iNaT) | (b == iNaT)
    return np.where(invalid, np.nan, d)

mvt_ns = _to_ns(df_ho_dep_full["MVT_TIME_UTC_mvt"])
block_ns = _to_ns(df_ho_dep_full["BLOCK_TIME_UTC_mvt"])
sched_ns = _to_ns(df_ho_dep_full["SCHED_TIME_UTC_mvt"])
aobt_ns = _to_ns(df_ho_dep_full["AOBT_3_flt"])

aobt_block = _diff(aobt_ns, block_ns)
mvt_aobt = np.where(_v(mvt_ns) & _v(aobt_ns), (mvt_ns - aobt_ns) / 1e9, np.nan)
mvt_sched = _diff(mvt_ns, sched_ns)
block_sched = _diff(block_ns, sched_ns)

aobt_valid = _v(aobt_ns)
mvt_valid = _v(mvt_ns)
block_valid = _v(block_ns)
sched_valid = _v(sched_ns)

# ============================================================
# [2] AOBT 三分类 (核心)
# ============================================================
print()
print("=" * 70); print("[2] AOBT 三分类 (完整 holdout, 含 >10800)")
print("=" * 70)
# 长滑行定义
m_long = y_all >= 1800
m_dep = df_ho_dep_full["PHASE_mvt"].values == "DEP"
m_long_dep = m_long & m_dep
n_long = int(m_long_dep.sum())
print(f"  长滑行 DEP 总数 (>=1800): {n_long:,}")

# 三分类
# real_v4: AOBT有效 AND |AOBT-BLOCK|<=300s (5 min 物理推出窗口) AND MVT-AOBT>0 AND MVT>AOBT
real_v4 = m_long_dep & aobt_valid & block_valid & (
    np.abs(aobt_block) <= 300) & _v(mvt_aobt) & (mvt_aobt > 0)
# dirty_v4: AOBT 缺失 OR |AOBT-BLOCK|>1800s
dirty_v4 = m_long_dep & ((~aobt_valid) | (block_valid & _v(aobt_block) & (np.abs(aobt_block) > 1800)))
# 不确定: 介于两者之间 (AOBT 有效但 |AOBT-BLOCK| 介于 5-30 min, 或 MVT-AOBT<=0)
uncertain_v4 = m_long_dep & ~real_v4 & ~dirty_v4

n_real = int(real_v4.sum())
n_dirty = int(dirty_v4.sum())
n_unc = int(uncertain_v4.sum())
print(f"    real_v4 (AOBT 有效 + 物理推出窗口 ±5min + MVT>AOBT): {n_real:,} ({n_real/n_long*100:.1f}%)")
print(f"    dirty_v4 (AOBT 缺失 或 |AOBT-BLOCK|>30min):           {n_dirty:,} ({n_dirty/n_long*100:.1f}%)")
print(f"    uncertain_v4 (介于两者):                             {n_unc:,} ({n_unc/n_long*100:.1f}%)")

# ============================================================
# [3] 分机场 × 三档 × 三分类
# ============================================================
print()
print("=" * 70); print("[3] 分机场 × 三档 × AOBT 三分类")
print("=" * 70)
aps = df_ho_dep_full["ADEP_mvt"].astype(str).values

bins3 = [(1800, 3600, "(1800,3600]"),
         (3600, 10800, "(3600,10800]"),
         (10800, float("inf"), ">10800")]

# 1) 全机场汇总
print()
print(f"  整体 (全 holdout DEP) 分档:")
print(f"    {'range':15s}  {'n':>7s}  {'real':>7s}  {'dirty':>7s}  {'unc':>6s}  "
      f"{'real%':>6s}  {'dirty%':>7s}")
totals = {}
for lo, hi, lab in bins3:
    m_bin = m_long_dep & (y_all > lo) & (y_all <= hi)
    n = int(m_bin.sum())
    r = int((m_bin & real_v4).sum())
    d = int((m_bin & dirty_v4).sum())
    u = int((m_bin & uncertain_v4).sum())
    totals[lab] = {"n": n, "real": r, "dirty": d, "unc": u,
                   "real_pct": r/n*100 if n else 0,
                   "dirty_pct": d/n*100 if n else 0}
    print(f"    {lab:15s}  {n:>7,}  {r:>7,}  {d:>7,}  {u:>6,}  "
          f"{r/n*100:>5.1f}%  {d/n*100:>6.1f}%")

# 2) 分机场汇总
print()
print(f"  分机场 (按长滑行总数从大到小):")
print(f"    {'ADEP':6s} {'n':>6s}  {'real':>7s}  {'dirty':>7s}  {'unc':>6s}  "
      f"{'real%':>6s}  {'dirty%':>7s}")
per_ap = {}
for ap in sorted(np.unique(aps)):
    m_ap = m_long_dep & (aps == ap)
    n = int(m_ap.sum())
    if n == 0: continue
    r = int((m_ap & real_v4).sum())
    d = int((m_ap & dirty_v4).sum())
    u = int((m_ap & uncertain_v4).sum())
    per_ap[ap] = {"n": n, "real": r, "dirty": d, "uncertain": u,
                   "real_pct": r/n*100, "dirty_pct": d/n*100, "unc_pct": u/n*100}
    print(f"    {ap:6s} {n:>6,}  {r:>7,}  {d:>7,}  {u:>6,}  "
          f"{r/n*100:>5.1f}%  {d/n*100:>6.1f}%")

# 3) 分机场 × 三档
print()
print(f"  分机场 × 三档 (real数 / 总数):")
print(f"    {'ADEP':6s} {'n_total':>7s} {'r18_36':>9s}  {'r36_108':>9s}  {'r>108':>8s}")
for ap in sorted(np.unique(aps)):
    m_ap = m_long_dep & (aps == ap)
    n = int(m_ap.sum())
    if n == 0: continue
    r1 = int((m_ap & real_v4 & (y_all > 1800) & (y_all <= 3600)).sum())
    n1 = int((m_ap & (y_all > 1800) & (y_all <= 3600)).sum())
    r2 = int((m_ap & real_v4 & (y_all > 3600) & (y_all <= 10800)).sum())
    n2 = int((m_ap & (y_all > 3600) & (y_all <= 10800)).sum())
    r3 = int((m_ap & real_v4 & (y_all > 10800)).sum())
    n3 = int((m_ap & (y_all > 10800)).sum())
    print(f"    {ap:6s} {n:>7,} {r1:>4d}/{n1:<4d}  {r2:>4d}/{n2:<4d}  "
          f"{r3:>3d}/{n3:<3d}")

# ============================================================
# [4] LIRF 专项: BLOCK==SCHED(±5s) & MVT-SCHED>45min
# ============================================================
print()
print("=" * 70); print("[4] LIRF 专项: BLOCK==SCHED(±5s) & MVT-SCHED>45min")
print("=" * 70)
m_lirf = (aps == "LIRF") & m_long_dep
n_lirf_long = int(m_lirf.sum())
print(f"  LIRF 长滑行总数: {n_lirf_long:,}")

# 严格: |BLOCK-SCHED| ≤ 5s
R_blocksched_strict = m_lirf & block_valid & sched_valid & (np.abs(block_sched) <= 5)
n_strict_bs = int(R_blocksched_strict.sum())
print(f"  BLOCK-SCHED ±5s 命中: {n_strict_bs:,}")

# 在这些里, MVT-SCHED > 45min
R_full = R_blocksched_strict & mvt_valid & sched_valid & (mvt_sched > 45*60)
n_full = int(R_full.sum())
print(f"  + MVT-SCHED > 45min: {n_full:,}")

# 与 v3 LIRF 脏值比较 (从 long_taxi_structure_v3.json 中读)
v3 = json.loads((D0 / "long_taxi_structure_v3.json").read_text())
lirf_v3 = v3["v3_by_airport"].get("LIRF", {})
print(f"  v3 LIRF 脏值数: {lirf_v3.get('dirty_n', '?')} / {lirf_v3.get('n', '?')}")
print(f"  v4 LIRF dirty_v4: {per_ap.get('LIRF', {}).get('dirty', '?')}")

# 抽样 20 行打印 AOBT 是否有效/合理性
print(f"\n  LIRF 全限定 (R_full) 抽样 20 行:")
rng = np.random.default_rng(99)
lirf_strict_idx = np.where(R_full)[0]
if len(lirf_strict_idx) > 0:
    pick = rng.choice(lirf_strict_idx, size=min(20, len(lirf_strict_idx)), replace=False)
    pick_sorted = sorted(pick.tolist())
    def _fmt(v):
        if pd.isna(v): return "(NaT)"
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")
    print(f"    {'idx':>6s}  {'TAX':>7s}  {'MVT':>16s}  {'BLOCK':>16s}  "
          f"{'SCHED':>16s}  {'AOBT':>16s}  {'MVT-SCHED':>10s}  {'AOBT-BLOCK':>10s}  {'AOBT valid':>11s}")
    for v in pick_sorted:
        tax = int(y_all[v])
        mvt = df_ho_dep_full["MVT_TIME_UTC_mvt"].iloc[v]
        blk = df_ho_dep_full["BLOCK_TIME_UTC_mvt"].iloc[v]
        sch = df_ho_dep_full["SCHED_TIME_UTC_mvt"].iloc[v]
        aob = df_ho_dep_full["AOBT_3_flt"].iloc[v]
        ms = mvt_sched[v] if not np.isnan(mvt_sched[v]) else float('nan')
        ab = aobt_block[v] if not np.isnan(aobt_block[v]) else float('nan')
        av = "Y" if aobt_valid[v] else "N"
        ms_str = f"{ms:.0f}s" if not np.isnan(ms) else "(NaN)"
        ab_str = f"{ab:.0f}s" if not np.isnan(ab) else "(NaN)"
        print(f"    {v:>6d}  {tax:>7d}  {_fmt(mvt):>16s}  {_fmt(blk):>16s}  "
              f"{_fmt(sch):>16s}  {_fmt(aob):>16s}  {ms_str:>10s}  {ab_str:>10s}  {av:>11s}")

# v3 LIRF 脏值抽样 (用于对比)
print(f"\n  v3 LIRF 脏值 抽样 10 行:")
v3_lirf_dirty_idx = np.where(m_lirf & (R_blocksched_strict | np.isnan(aobt_block))) [0]
# 用 v3 dirty_v3 mask
lirf_v3_dirty = m_lirf & ~v3["v3_by_airport"].get("LIRF", {}).get("real_n", 0) * np.ones(len(y_all), dtype=bool)
# 简单近似: 抽 dirty_v4 全部 35+ 行
lirf_dirty_idx = np.where(m_lirf & dirty_v4)[0]
print(f"    dirty_v4 LIRF 总数: {len(lirf_dirty_idx)}")
if len(lirf_dirty_idx) > 0:
    pick = rng.choice(lirf_dirty_idx, size=min(10, len(lirf_dirty_idx)), replace=False)
    pick_sorted = sorted(pick.tolist())
    print(f"    {'idx':>6s}  {'TAX':>7s}  {'MVT':>16s}  {'BLOCK':>16s}  "
          f"{'SCHED':>16s}  {'AOBT':>16s}  {'AOBT-BLOCK':>10s}  {'AOBT valid':>11s}")
    for v in pick_sorted:
        tax = int(y_all[v])
        mvt = df_ho_dep_full["MVT_TIME_UTC_mvt"].iloc[v]
        blk = df_ho_dep_full["BLOCK_TIME_UTC_mvt"].iloc[v]
        sch = df_ho_dep_full["SCHED_TIME_UTC_mvt"].iloc[v]
        aob = df_ho_dep_full["AOBT_3_flt"].iloc[v]
        ab = aobt_block[v] if not np.isnan(aobt_block[v]) else float('nan')
        av = "Y" if aobt_valid[v] else "N"
        ab_str = f"{ab:.0f}s" if not np.isnan(ab) else "(NaN)"
        print(f"    {v:>6d}  {tax:>7d}  {_fmt(mvt):>16s}  {_fmt(blk):>16s}  "
              f"{_fmt(sch):>16s}  {_fmt(aob):>16s}  {ab_str:>10s}  {av:>11s}")

# 解释差异
print(f"\n  差异解释:")
print(f"    v3 LIRF 脏值数: {lirf_v3.get('dirty_n', '?')}")
print(f"    v4 LIRF dirty_v4 数: {per_ap.get('LIRF', {}).get('dirty', '?')}")
print(f"    LIRF 长滑行总: {n_lirf_long}")
print(f"    v3 R1 (BLOCK≈SCHED±60s & MVT-SCHED>30min): {int((m_lirf & block_valid & sched_valid & (np.abs(block_sched) <= 60) & mvt_valid & sched_valid & (mvt_sched > 30*60)).sum())} 行")
print(f"    v4 dirty_v4 (AOBT 缺失 或 |AOBT-BLOCK|>30min): {per_ap.get('LIRF', {}).get('dirty', '?')} 行")
print(f"    v4 不确定 (AOBT-BLOCK 在 5-30min 之间): {per_ap.get('LIRF', {}).get('uncertain', '?')} 行")
print(f"    v4 real_v4 (AOBT 有效 + |AOBT-BLOCK| ≤5min + MVT>AOBT): {per_ap.get('LIRF', {}).get('real', '?')} 行")

# ============================================================
# [5] 可建模样本确切数
# ============================================================
print()
print("=" * 70); print("[5] 可建模样本 (real_v4) 确切数量")
print("=" * 70)
n_real_v4 = int(real_v4.sum())
print(f"  real_v4 总样本数: {n_real_v4:,} (即 sub-model 可学样本)")
print(f"  分机场可学样本 (real_v4):")
for ap in sorted(np.unique(aps)):
    n = int((m_long_dep & real_v4 & (aps == ap)).sum())
    if n >= 50:
        print(f"    {ap}: {n} 行")

# ============================================================
# [6] 抽样: 每机场三类各 5 行
# ============================================================
print()
print("=" * 70); print("[6] 抽样: 每机场 三类各 5 行")
print("=" * 70)
rng = np.random.default_rng(2026)
samples_table = []
for ap in sorted(np.unique(aps)):
    m_ap = m_long_dep & (aps == ap)
    if m_ap.sum() < 10: continue

    print(f"\n  ----- {ap} -----")
    print(f"  {'vidx':>6s} {'TAX':>7s} {'MVT':>16s} {'BLOCK':>16s} {'SCHED':>16s} "
          f"{'AOBT':>16s} {'MVT-AOBT':>9s} {'|AOBT-BLOCK|':>13s} {'verdict':>10s}")

    def _fmt(v):
        if pd.isna(v): return "(NaT)"
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")
    def _show(v, verdict):
        tax = int(y_all[v])
        mvt = df_ho_dep_full["MVT_TIME_UTC_mvt"].iloc[v]
        blk = df_ho_dep_full["BLOCK_TIME_UTC_mvt"].iloc[v]
        sch = df_ho_dep_full["SCHED_TIME_UTC_mvt"].iloc[v]
        aob = df_ho_dep_full["AOBT_3_flt"].iloc[v]
        ma = mvt_aobt[v] if not np.isnan(mvt_aobt[v]) else float('nan')
        ab = aobt_block[v] if not np.isnan(aobt_block[v]) else float('nan')
        ma_str = f"{ma:.0f}s" if not np.isnan(ma) else "(NaN)"
        ab_str = f"{ab:.0f}s" if not np.isnan(ab) else "(NaN)"
        print(f"  {v:>6d} {tax:>7d} {_fmt(mvt):>16s} {_fmt(blk):>16s} {_fmt(sch):>16s} "
              f"{_fmt(aob):>16s} {ma_str:>9s} {ab_str:>13s} {verdict:>10s}")
        samples_table.append({
            "airport": ap, "vidx": v, "TAX": int(y_all[v]),
            "MVT_UTC": str(mvt), "BLOCK_UTC": str(blk), "SCHED_UTC": str(sch),
            "AOBT_UTC": str(aob),
            "MVT_minus_AOBT": float(ma) if not np.isnan(ma) else None,
            "AOBT_minus_BLOCK": float(ab) if not np.isnan(ab) else None,
            "verdict_v4": verdict,
        })

    real_idx = np.where(m_ap & real_v4)[0]
    if len(real_idx) > 0:
        pick = rng.choice(real_idx, size=min(5, len(real_idx)), replace=False)
        for v in sorted(pick.tolist()): _show(v, "REAL_v4")

    dirty_idx = np.where(m_ap & dirty_v4)[0]
    if len(dirty_idx) > 0:
        pick = rng.choice(dirty_idx, size=min(5, len(dirty_idx)), replace=False)
        for v in sorted(pick.tolist()): _show(v, "DIRTY_v4")

    unc_idx = np.where(m_ap & uncertain_v4)[0]
    if len(unc_idx) > 0:
        pick = rng.choice(unc_idx, size=min(5, len(unc_idx)), replace=False)
        for v in sorted(pick.tolist()): _show(v, "UNCERTAIN")

# ============================================================
# 落盘
# ============================================================
out = {
    "n_long_total": n_long,
    "v4_classification": {
        "real_v4_n": n_real, "real_v4_pct": n_real/n_long*100,
        "dirty_v4_n": n_dirty, "dirty_v4_pct": n_dirty/n_long*100,
        "uncertain_v4_n": n_unc, "uncertain_v4_pct": n_unc/n_long*100,
    },
    "by_bin": totals,
    "per_airport": per_ap,
    "lirf_specific": {
        "lirf_long_n": n_lirf_long,
        "blocksched_strict_5s": n_strict_bs,
        "blocksched_strict_5s_AND_mvt_sched_gt_45min": n_full,
        "v3_lirf_dirty_n": lirf_v3.get("dirty_n"),
        "v4_lirf_dirty_n": per_ap.get("LIRF", {}).get("dirty"),
        "v4_lirf_real_n": per_ap.get("LIRF", {}).get("real"),
        "v4_lirf_uncertain_n": per_ap.get("LIRF", {}).get("uncertain"),
    },
    "submodel_trainable_samples": n_real_v4,
    "submodel_per_airport": {
        ap: int((m_long_dep & real_v4 & (aps == ap)).sum())
        for ap in sorted(np.unique(aps))
    },
    "samples": samples_table,
}
(D0 / "long_taxi_structure_v4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print()
print(f"  → 写入: {D0 / 'long_taxi_structure_v4.json'}")
