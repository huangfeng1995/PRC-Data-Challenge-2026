"""
诊断 1: holdout 上 t_unimp < 306 / > 2752 的航班
统计数量, ADEP/STAND/RUNWAY 组合, LUT 命中层级
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

import _r7_lib as L
from src.data_loader import load_month, parse_time_columns

# ============================================================
# 1. 加载 holdout (1+7)
# ============================================================
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
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    df = load_month(f"data/raw/{fname}", columns=TRAIN_COLS)
    ho_dfs.append(df)
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)

# ============================================================
# 2. 加载训练 (2-12), 构建 LUT
# ============================================================
months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months:
    fname = (f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
             if m < 12 else "training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)

# 用 TAXITIME>0&<10800 筛 DEP
df_tr_dep = df_tr[
    (df_tr["PHASE_mvt"] == "DEP") &
    (df_tr["TAXITIME_SEC_mvt"] > 0) &
    (df_tr["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)

print(f"训练 DEP: {len(df_tr_dep):,}")

# ============================================================
# 3. LUT 诊断: 看看 q30 在训练数据上的分布
# ============================================================
print("\n" + "=" * 70)
print("[LUT 构建诊断] q30 在训练数据上的分布")
print("=" * 70)

dep = df_tr_dep
tt = dep["TAXITIME_SEC_mvt"].values

print(f"\n全 TAXITIME_SEC 分布:")
print(f"  n={len(tt)}, min={tt.min():.1f}, p1={np.percentile(tt,1):.1f}, p5={np.percentile(tt,5):.1f}, "
      f"q30={np.quantile(tt,0.3):.1f}, p50={np.median(tt):.1f}, max={tt.max():.1f}")

# < 60s 的"短滑行"
short = tt < 60
print(f"\n< 60s (短滑行): {short.sum()} ({100*short.sum()/len(tt):.2f}%)")
print(f"  < 30s: {(tt<30).sum()}, < 10s: {(tt<10).sum()}, < 5s: {(tt<5).sum()}")

# 按 (ADEP, STAND, RUNWAY) group 内的 q30 — 看有多少 group 的 q30 异常小
g_l1 = dep.groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])["TAXITIME_SEC_mvt"].quantile(0.3)
print(f"\nL1 (ADEP,STAND,RUNWAY) q30:")
print(f"  n_group={len(g_l1)}, n_valid={g_l1.notna().sum()}")
print(f"  min={g_l1.min():.1f}, p1={g_l1.quantile(0.01):.1f}, "
      f"p10={g_l1.quantile(0.1):.1f}, p50={g_l1.median():.1f}, "
      f"max={g_l1.max():.1f}")
print(f"  < 60s: {(g_l1 < 60).sum()}, < 100s: {(g_l1 < 100).sum()}, "
      f"< 200s: {(g_l1 < 200).sum()}")

# 看 L1 q30 < 60 的 group 长啥样
small_groups = g_l1[g_l1 < 60]
print(f"\nL1 q30 < 60s 的 group 例子 (前 10):")
for (a, s, r), v in small_groups.head(10).items():
    print(f"  {a}/{s}/{r}: q30={v:.1f}s, n_in_group={len(dep[(dep['ADEP_mvt']==a)&(dep['STAND_mvt']==s)&(dep['RUNWAY_mvt']==r)])}")

# ============================================================
# 4. 在 holdout 上查 LUT, 看 < 306 / > 2752 的航班
# ============================================================
print("\n" + "=" * 70)
print("[Holdout t_unimp 诊断] 查 LUT 后异常值")
print("=" * 70)

lut = L.build_t_unimp_lut(df_tr_dep)
print(f"  LUT: L1={len(lut['L1'])}, L2={len(lut['L2'])}, L3={len(lut['L3'])}")
print(f"  L1 min={lut['L1'].min():.1f}, max={lut['L1'].max():.1f}")
print(f"  L2 min={lut['L2'].min():.1f}, max={lut['L2'].max():.1f}")
print(f"  L3 min={lut['L3'].min():.1f}, max={lut['L3'].max():.1f}")

# Holdout DEP
df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_ho_dep = df_ho_dep[
    (df_ho_dep["TAXITIME_SEC_mvt"] > 0) &
    (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)
].reset_index(drop=True)
print(f"  Holdout DEP: {len(df_ho_dep):,}")

t_vals, t_src = L.lookup_t_unimp(df_ho_dep, lut)
df_ho_dep["t_unimp_r7"] = t_vals
df_ho_dep["t_unimp_level"] = t_src

# 异常低 (<306)
low = df_ho_dep[df_ho_dep["t_unimp_r7"] < 306]
print(f"\nt_unimp < 306: {len(low)} ({100*len(low)/len(df_ho_dep):.2f}%)")
print(f"  按 level 分布:")
print(low["t_unimp_level"].value_counts().to_string())
print(f"  按 ADDEP 分布 (top 10):")
print(low["ADEP_mvt"].value_counts().head(10).to_string())

# 异常高 (>2752)
high = df_ho_dep[df_ho_dep["t_unimp_r7"] > 2752]
print(f"\nt_unimp > 2752: {len(high)} ({100*len(high)/len(df_ho_dep):.2f}%)")
print(f"  按 level 分布:")
print(high["t_unimp_level"].value_counts().to_string())
print(f"  按 ADEP 分布 (top 10):")
print(high["ADEP_mvt"].value_counts().head(10).to_string())

# < 100 的极小值 (脏数据信号)
very_low = df_ho_dep[df_ho_dep["t_unimp_r7"] < 100]
print(f"\nt_unimp < 100 (极小, 物理可疑): {len(very_low)} ({100*len(very_low)/len(df_ho_dep):.2f}%)")
print(f"  按 level 分布:")
print(very_low["t_unimp_level"].value_counts().to_string())
print(f"  按 ADEP 分布:")
print(very_low["ADEP_mvt"].value_counts().head(5).to_string())

# 这些极小值航班对应的训练 LUT group 长啥样
print(f"\n=== 极小值航班的 LUT 来源诊断 ===")
for level in ["L1", "L2", "L3"]:
    sub = very_low[very_low["t_unimp_level"] == level]
    if len(sub) == 0:
        continue
    print(f"\n[{level}] {len(sub)} 航班:")
    print(f"  t_unimp 分布: min={sub['t_unimp_r7'].min():.1f}, p50={sub['t_unimp_r7'].median():.1f}, max={sub['t_unimp_r7'].max():.1f}")
    # 举 3 个具体例子
    for _, r in sub.head(3).iterrows():
        a, s, rw = r["ADEP_mvt"], r["STAND_mvt"], r["RUNWAY_mvt"]
        n_grp = len(dep[(dep["ADEP_mvt"]==a) & (dep["STAND_mvt"]==s) & (dep["RUNWAY_mvt"]==rw)])
        if level == "L1":
            v_l1 = lut["L1"].get((a, s, rw), np.nan)
            print(f"  {a}/{s}/{rw}: t_unimp={r['t_unimp_r7']:.1f}s, "
                  f"训练 L1 group 大小={n_grp}, q30={v_l1:.1f}s" if not np.isnan(v_l1) else
                  f"  {a}/{s}/{rw}: t_unimp={r['t_unimp_r7']:.1f}s, L1 NaN")
        elif level == "L2":
            v_l2 = lut["L2"].get((a, s), np.nan)
            print(f"  {a}/{s}/{rw}: t_unimp={r['t_unimp_r7']:.1f}s, "
                  f"L2 group 大小={len(dep[(dep['ADEP_mvt']==a)&(dep['STAND_mvt']==s)])}, q30={v_l2:.1f}s")
        else:
            v_l3 = lut["L3"].get(a, np.nan)
            print(f"  {a}: t_unimp={r['t_unimp_r7']:.1f}s, "
                  f"L3 group 大小={len(dep[dep['ADEP_mvt']==a])}, q30={v_l3:.1f}s")