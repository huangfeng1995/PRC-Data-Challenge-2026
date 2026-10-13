"""
脏值判据核验 v3 (read-only)
=============================
红线: 不改 label / 不 clip 真值 / 不建单机场模型

1. 逐条复核脏值规则 + 加交叉条件
   - BLOCK≈SCHED(±60s) 单独不够脏; 用 |MVT−SCHED| 拆
   - TAX>MVT−AOBT 单独不够脏; 用 BLOCK−AOBT 拆
   - 跨天 ±86400 / MVT 日期差>30d / AOBT 晚于 MVT 三条保留, 逐条报命中
2. 用修正后判据重统计 (机场, 三档)
3. 每个机场抽 10 行 (5 real + 5 dirty 候选), 打印 MVT/BLOCK/SCHED/AOBT + TAXITIME + MVT−AOBT + 命中规则
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
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt",
                  "AIRCRAFT_OPERATOR_flt"]

print("=" * 70); print("[1] 加载 holdout + 训练 + build_features + LGB 推理")
print("=" * 70)
t0 = time.time()
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
print(f"  全流程 {time.time()-t0:.1f}s | holdout DEP {len(df_ho_dep_full):,}")

# ============================================================
# 派生特征 (不动 label)
# ============================================================
def _to_ns(s):
    return pd.to_datetime(s, utc=True, errors="coerce").values.astype("datetime64[ns]").view("i8")
iNaT = np.iinfo(np.int64).min
def _v(a): return (a != iNaT) & ~np.isnan(a)

mvt_ns = _to_ns(df_ho_dep_full["MVT_TIME_UTC_mvt"])
block_ns = _to_ns(df_ho_dep_full["BLOCK_TIME_UTC_mvt"])
sched_ns = _to_ns(df_ho_dep_full["SCHED_TIME_UTC_mvt"])
aobt_ns = _to_ns(df_ho_dep_full["AOBT_3_flt"])

def _diff(a, b):
    """a - b 秒, 缺失 NaN"""
    d = (a - b) / 1e9
    invalid = (a == iNaT) | (b == iNaT)
    return np.where(invalid, np.nan, d)

diff_block_sched = _diff(block_ns, sched_ns)        # BLOCK - SCHED (秒)
diff_mvt_sched = _diff(mvt_ns, sched_ns)            # MVT - SCHED
diff_aobt_block = _diff(aobt_ns, block_ns)          # AOBT - BLOCK  (正常是 -5 to +5 min)
diff_tax_mvtaobt = np.where(
    _v(mvt_ns) & _v(aobt_ns),
    (mvt_ns - aobt_ns) / 1e9 - y_all,
    np.nan
)                                                    # (MVT-AOBT) - TAX  → ≈0 正常
diff_mvt_aobt = np.where(
    _v(mvt_ns) & _v(aobt_ns), (mvt_ns - aobt_ns) / 1e9, np.nan
)
diff_block_aobt = _diff(block_ns, aobt_ns)          # BLOCK - AOBT (推后于 BLOCK 的时间)

# ============================================================
# [2] 逐条复核: 修正前/后命中数
# ============================================================
print()
print("=" * 70); print("[2] 逐条规则修正前后命中数 (完整 holdout DEP)")
print("=" * 70)

m_dep = df_ho_dep_full["PHASE_mvt"].values == "DEP"
m_long_v2 = (y_all >= 1800) & m_dep  # V2 的"长滑行"定义 (不 clip 上限, 含 >10800)
n_long_v2 = int(m_long_v2.sum())

# --- 规则 (R1) BLOCK≈SCHED ±60s ---
R1 = _v(block_ns) & _v(sched_ns) & (np.abs(diff_block_sched) <= 60)
n_r1 = int((R1 & m_long_v2).sum())
print(f"\n  R1 (BLOCK≈SCHED ±60s): v2命中 (长滑行)={n_r1:,}")

# 修正 R1: 加交叉 |MVT−SCHED| 阈值
#   - 准点:    |MVT−SCHED| ≤ 30min (即可接受 = 实际滑出是从准点起飞时间算起)
#   - 矛盾真脏值: BLOCK≈SCHED 但 MVT−SCHED>30min (=撤轮挡似准点, 起飞却晚很多, label 错)
for thr_min in [10, 20, 30, 60, 120]:
    R1_kept_dirty = R1 & _v(mvt_ns) & _v(sched_ns) & (diff_mvt_sched > thr_min * 60)
    n = int((R1_kept_dirty & m_long_v2).sum())
    print(f"    + 交叉: BLOCK≈SCHED & MVT-SCHED>{thr_min}min → {n:,} (long-tail)")

# 用 30min 阈值的 R1_mod (修正版)
R1_mod = R1 & _v(mvt_ns) & _v(sched_ns) & (diff_mvt_sched > 30 * 60)
n_r1_mod = int((R1_mod & m_long_v2).sum())
print(f"    = R1_modified (保留 dirty 判定): {n_r1_mod:,}")
print(f"    误判恢复 (R1 → R1_mod): {n_r1 - n_r1_mod:,} 行原被判脏但实际可能是准点 ({100*(n_r1-n_r1_mod)/max(n_r1,1):.1f}%)")

# --- 规则 (R2) TAX > MVT - AOBT (无对照) ---
# 直接 (MVT-AOBT) - TAX ≈0; 正常：(MVT-AOBT) - TAX ∈ [-30s, +30s] (含 AOBT/起飞的传输误差)
# 矛盾真脏值：(MVT-AOBT) - TAX > 30min (即 label 错, 物理上 TAX 应该是 MVT-AOBT 不能多 30min)
# 但 TAX 字段又可能被低估, 所以同时考虑另一种矛盾: TAX 远 > (MVT-AOBT)  (label 包含等待时间)
# "logically impossible" 方向: TAX >> (MVT-AOBT) 时, 是 TAX 字段把等待时间也算进去
# 我们这里单独看 (MVT-AOBT) - TAX:
#   - 如果 (MVT-AOBT) - TAX < -1800: TAX 字段比 (MVT-AOBT) 大很多 → "TAX 字段含等待时间", 不算脏(只是定义模糊)
#   - 如果 (MVT-AOBT) - TAX > +300: TAX 比 (MVT-AOBT) 小很多 → "label 错位"
# 修正: R2 单独不够, 需 + 另一条错位证据 (BLOCK-AOBT 异常 / 等)
print(f"\n  R2 (TAX>MVT-AOBT): v2未直接使用此条, 因为 (MVT-AOBT)-TAX 分布: ")
print(f"    全部 DEP: (MVT-AOBT)-TAX 分布:")
mask_a = _v(mvt_ns) & _v(aobt_ns) & (y_all >= 0)
print(f"      < -1800s (TAX 含等待时间): {int((diff_tax_mvtaobt < -1800)[mask_a].sum()):,}")
print(f"      -1800 ~ -300s (TAX 多 5-30min): {int(((diff_tax_mvtaobt >= -1800) & (diff_tax_mvtaobt < -300))[mask_a].sum()):,}")
print(f"      -300 ~ +30s (正常): {int(((diff_tax_mvtaobt >= -300) & (diff_tax_mvtaobt <= 30))[mask_a].sum()):,}")
print(f"      +30 ~ +1800s (异常小): {int(((diff_tax_mvtaobt > 30) & (diff_tax_mvtaobt <= 1800))[mask_a].sum()):,}")
print(f"      > +1800s (矛盾极大): {int((diff_tax_mvtaobt > 1800)[mask_a].sum()):,}")

# 先统计 block_aobt 分布 (评估 R2 交叉条件)
print(f"\n    BLOCK-AOBT 分布 (推后于 BLOCK 的时间, 正常 < 60s):")
mask_ba = _v(block_ns) & _v(aobt_ns)
block_aobt_arr = diff_block_aobt[mask_ba]
print(f"      < 60s (正常): {int((block_aobt_arr < 60).sum()):,}")
print(f"      60~600s (10min, 异常): {int(((block_aobt_arr >= 60) & (block_aobt_arr < 600)).sum()):,}")
print(f"      600~1800s (30min, 真脏): {int(((block_aobt_arr >= 600) & (block_aobt_arr < 1800)).sum()):,}")
print(f"      > 1800s (>30min, 严重脏): {int((block_aobt_arr >= 1800).sum()):,}")

# R2_mod: TAX 大于 (MVT-AOBT) 但 BLOCK-AOBT 也异常 (>600s)
# 这才是真矛盾: 飞机明明 30min 前未推出, 现在 TAX 字段却是 5min
R2_mod = (diff_tax_mvtaobt > 1800) & _v(block_ns) & _v(aobt_ns) & (diff_block_aobt > 600)
n_r2_mod = int((R2_mod & m_long_v2).sum())
print(f"    R2_modified (TAX>MVT-AOBT+30min & BLOCK-AOBT>10min): {n_r2_mod:,} (long-tail)")

# R2b: 用 BLOCK-AOBT 单独异常 >30min 当 R2 (label 字段矛盾代理)
R2b = _v(block_ns) & _v(aobt_ns) & (diff_block_aobt > 1800)
n_r2b = int((R2b & m_long_v2).sum())
print(f"    R2b (BLOCK-AOBT >30min 单独): {n_r2b:,} (long-tail)")

# --- 规则 (R3) 跨天 ±86400 ---
guess_p1d = (mvt_ns - sched_ns) / 1e9 + 86400
guess_m1d = (mvt_ns - sched_ns) / 1e9 - 86400
R3 = _v(mvt_ns) & _v(sched_ns) & (
    (np.abs(y_all - guess_p1d) <= 600) | (np.abs(y_all - guess_m1d) <= 600))
n_r3 = int((R3 & m_long_v2).sum())
print(f"\n  R3 (跨天 ±86400 ±600s): {n_r3:,} (long-tail)")

# --- 规则 (R4) MVT日期差 >30d (跨年/月错位) ---
mvt_vs_sched_days = (mvt_ns - sched_ns) / 1e9 / 86400
R4 = _v(mvt_ns) & _v(sched_ns) & (
    (mvt_vs_sched_days < -30) | (mvt_vs_sched_days > 30))
n_r4 = int((R4 & m_long_v2).sum())
print(f"  R4 (|MVT-SCHED|>30d): {n_r4:,} (long-tail)")

# --- 规则 (R5) AOBT 远晚于 MVT (>10min) ---
R5 = _v(aobt_ns) & _v(mvt_ns) & (_diff(aobt_ns, mvt_ns) > 600)
n_r5 = int((R5 & m_long_v2).sum())
print(f"  R5 (AOBT-MVT>10min): {n_r5:,} (long-tail)")

# --- V2 旧判定 (组合 OR) ---
m_dirty_v2 = (R1 | R3 | R4 | R5) & m_long_v2
n_dirty_v2 = int(m_dirty_v2.sum())
# --- V3 修正判定 (R1_mod OR (R2_mod OR R2b) OR R3 OR R4 OR R5) ---
m_dirty_v3 = (R1_mod | R2_mod | R2b | R3 | R4 | R5) & m_long_v2
n_dirty_v3 = int(m_dirty_v3.sum())
n_real_v2 = n_long_v2 - n_dirty_v2
n_real_v3 = n_long_v2 - n_dirty_v3

print(f"\n  --- 汇总 (v2 vs v3) ---")
print(f"    长滑行 (>=1800):       {n_long_v2:,}")
print(f"    V2 旧判据 脏值:        {n_dirty_v2:,} ({n_dirty_v2/n_long_v2*100:.1f}%)")
print(f"    V3 修正判据 脏值:      {n_dirty_v3:,} ({n_dirty_v3/n_long_v2*100:.1f}%)")
print(f"    V3 真实 (修正后):      {n_real_v3:,} ({n_real_v3/n_long_v2*100:.1f}%)")
print(f"    V2 → V3 误判恢复 (误杀): {n_dirty_v2 - n_dirty_v3:,}")

# ============================================================
# [3] 用 V3 重统计: 机场 × 档 × 真实/脏值
# ============================================================
print()
print("=" * 70); print("[3] V3 分机场 × 三档 重统计")
print("=" * 70)
aps = df_ho_dep_full["ADEP_mvt"].astype(str).values

bins3 = [(1800, 3600), (3600, 10800), (10800, float('inf'))]
bin_labels = ["(1800,3600]", "(3600,10800]", ">10800"]

print(f"  V2 (旧):")
print(f"  {'ADEP':6s} {'n':>6s} {'r18_36':>7s} {'r36_108':>8s} {'r>108':>6s} "
      f"{'real%':>6s} {'dirty%':>7s}")
v2_per = {}
for ap in sorted(np.unique(aps)):
    m_ap = m_long_v2 & (aps == ap)
    n_ap = int(m_ap.sum())
    if n_ap == 0: continue
    n_r = int((m_ap & ~m_dirty_v2).sum())
    r1 = int((m_ap & ~m_dirty_v2 & (y_all > 1800) & (y_all <= 3600)).sum())
    r2 = int((m_ap & ~m_dirty_v2 & (y_all > 3600) & (y_all <= 10800)).sum())
    r3 = int((m_ap & ~m_dirty_v2 & (y_all > 10800)).sum())
    print(f"  {ap:6s} {n_ap:>6d} {r1:>7d} {r2:>8d} {r3:>6d} "
          f"{n_r/n_ap*100:>5.1f}% {(n_ap-n_r)/n_ap*100:>6.1f}%")
    v2_per[ap] = {"n": n_ap, "real_n": n_r, "real_pct": n_r/n_ap*100,
                  "dirty_n": n_ap-n_r, "dirty_pct": (n_ap-n_r)/n_ap*100}

print(f"\n  V3 (修正):")
print(f"  {'ADEP':6s} {'n':>6s} {'r18_36':>7s} {'r36_108':>8s} {'r>108':>6s} "
      f"{'real%':>6s} {'dirty%':>7s}  {'Δreal%':>7s}")
v3_per = {}
for ap in sorted(np.unique(aps)):
    m_ap = m_long_v2 & (aps == ap)
    n_ap = int(m_ap.sum())
    if n_ap == 0: continue
    n_r = int((m_ap & ~m_dirty_v3).sum())
    r1 = int((m_ap & ~m_dirty_v3 & (y_all > 1800) & (y_all <= 3600)).sum())
    r2 = int((m_ap & ~m_dirty_v3 & (y_all > 3600) & (y_all <= 10800)).sum())
    r3 = int((m_ap & ~m_dirty_v3 & (y_all > 10800)).sum())
    real_pct = n_r/n_ap*100
    prev_real_pct = v2_per.get(ap, {}).get("real_pct", 0)
    delta = real_pct - prev_real_pct
    print(f"  {ap:6s} {n_ap:>6d} {r1:>7d} {r2:>8d} {r3:>6d} "
          f"{real_pct:>5.1f}% {(n_ap-n_r)/n_ap*100:>6.1f}%  "
          f"{delta:>+6.1f}%")
    v3_per[ap] = {"n": n_ap, "real_n": n_r, "real_pct": real_pct,
                  "dirty_n": n_ap-n_r, "dirty_pct": (n_ap-n_r)/n_ap*100,
                  "real_1800_3600": r1, "real_3600_10800": r2, "real_gt10800": r3,
                  "delta_real_pct_vs_v2": delta}

# 分档 V3 整体
print(f"\n  V3 分档整体:")
print(f"    {'range':15s}  {'n':>7s}  {'real':>7s}  {'real%':>6s}  {'dirty':>7s}  {'dirty%':>7s}")
v3_bin = {}
for (lo, hi), lab in zip(bins3, bin_labels):
    m_bin = m_long_v2 & (y_all > lo) & (y_all <= hi)
    n_bin = int(m_bin.sum())
    n_r = int((m_bin & ~m_dirty_v3).sum())
    v3_bin[lab] = {"n": n_bin, "real_n": n_r,
                   "real_pct": n_r/n_bin*100 if n_bin else 0}
    print(f"    {lab:15s}  {n_bin:>7,}  {n_r:>7,}  {n_r/n_bin*100:>5.1f}%  "
          f"{n_bin-n_r:>7,}  {(n_bin-n_r)/n_bin*100:>5.1f}%")

# ============================================================
# [4] 抽样人工核对 (每机场 10 行: 5 real + 5 dirty)
# ============================================================
print()
print("=" * 70); print("[4] 抽样人工核对 (每机场 10 行: 5 real + 5 dirty V3)")
print("=" * 70)

rng = np.random.default_rng(42)
samples_table = []
for ap in sorted(v3_per.keys()):
    m_ap = m_long_v2 & (aps == ap)
    if m_ap.sum() < 10: continue

    # 5 real
    real_idx = np.where(m_ap & ~m_dirty_v3)[0]
    if len(real_idx) >= 5:
        real_idx = rng.choice(real_idx, size=min(5, len(real_idx)), replace=False)
    # 5 dirty (V3) - 若 v3 dirty < 5, 拿 v2-only 标 dirty 的补足
    dirty_v3_idx = np.where(m_ap & m_dirty_v3)[0]
    if len(dirty_v3_idx) >= 5:
        dirty_v3_idx = rng.choice(dirty_v3_idx, size=5, replace=False)
    else:
        # 拿全部
        pass

    print(f"\n  ----- {ap} -----")
    print(f"  {'vidx':>5} {'TAX':>7} {'MVT_UTC':>20} {'BLOCK_UTC':>20} "
          f"{'SCHED_UTC':>20} {'AOBT_UTC':>20} {'MVT-AOBT':>9} {'R1':>3} {'R2b':>3} {'R3':>3} {'R4':>3} {'R5':>3} {'verdict':>10}")

    def _fmt(v):
        if pd.isna(v): return "(NaT)"
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")

    def _show(vidx, verdict):
        v = vidx
        tax = int(y_all[v])
        mvt = df_ho_dep_full["MVT_TIME_UTC_mvt"].iloc[v]
        blk = df_ho_dep_full["BLOCK_TIME_UTC_mvt"].iloc[v]
        sch = df_ho_dep_full["SCHED_TIME_UTC_mvt"].iloc[v]
        aob = df_ho_dep_full["AOBT_3_flt"].iloc[v]
        mvt_aobt = diff_mvt_aobt[v]
        mvt_aobt_str = f"{mvt_aobt:.0f}s" if not np.isnan(mvt_aobt) else "(NaN)"
        hit = lambda R: "Y" if R[v] else "."
        print(f"  {v:>5d} {tax:>7d} {_fmt(mvt):>20} {_fmt(blk):>20} "
              f"{_fmt(sch):>20} {_fmt(aob):>20} {mvt_aobt_str:>9} "
              f"{hit(R1_mod):>3} {hit(R2b):>3} {hit(R3):>3} {hit(R4):>3} {hit(R5):>3} {verdict:>10}")
        samples_table.append({
            "airport": ap, "vidx": v, "TAX": int(y_all[v]),
            "MVT_UTC": str(mvt), "BLOCK_UTC": str(blk),
            "SCHED_UTC": str(sch), "AOBT_UTC": str(aob),
            "MVT_minus_AOBT": float(mvt_aobt) if not np.isnan(mvt_aobt) else None,
            "R1_mod_hit": bool(R1_mod[v]),
            "R2b_hit": bool(R2b[v]),
            "R3_hit": bool(R3[v]),
            "R4_hit": bool(R4[v]),
            "R5_hit": bool(R5[v]),
            "verdict_v3": verdict,
        })

    for v in real_idx: _show(v, "REAL")
    for v in dirty_v3_idx: _show(v, "DIRTY")

# ============================================================
# 落盘
# ============================================================
out = {
    "n_long": n_long_v2,
    "rules_v2_dirty_n": n_dirty_v2,
    "rules_v2_real_n": n_real_v2,
    "rules_v3_dirty_n": n_dirty_v3,
    "rules_v3_real_n": n_real_v3,
    "rules_misjudged_recovered": n_dirty_v2 - n_dirty_v3,
    "rules_per_rule": {
        "R1_blocksched_only": int((R1 & m_long_v2).sum()),
        "R1_mod_blocksched_plus_mvt_sched_threshold_30min": n_r1_mod,
        "R2_mod_tax_gt_mvtaobt_plus_block_aobt_anomaly": int((R2_mod & m_long_v2).sum()),
        "R2b_block_aobt_gt_30min_alone": n_r2b,
        "R3_dayoff": n_r3,
        "R4_mvt_sched_30d": n_r4,
        "R5_aobt_late_after_mvt": n_r5,
    },
    "v3_by_airport": v3_per,
    "v3_by_bin": v3_bin,
    "v2_by_airport": v2_per,
    "samples": samples_table,
}
(D0 / "long_taxi_structure_v3.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False, default=str)
)
print()
print(f"  → 写入: {D0 / 'long_taxi_structure_v3.json'}")
