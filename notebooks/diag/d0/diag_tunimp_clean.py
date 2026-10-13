"""
干净版 LUT: 过滤物理不可能值, 限定合理范围, L1 最小样本数
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

import _r7_lib as L
from src.data_loader import load_month, parse_time_columns

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


def build_t_unimp_lut_clean(df_train, min_sec=60, max_sec=3600,
                              min_obs=5, quantile=0.3):
    """干净版 LUT:
      - 只用 TAXITIME in [min_sec, max_sec] 的样本（过滤脏数据）
      - L1 group 内 obs < min_obs 时回退到 L2
      - L2 obs < min_obs 时回退到 L3
      - L3 也是用干净样本算
    """
    dep = df_train[df_train["PHASE_mvt"] == "DEP"].copy()
    # 过滤物理不可能值
    dep = dep[(dep["TAXITIME_SEC_mvt"] >= min_sec) & (dep["TAXITIME_SEC_mvt"] <= max_sec)]

    # L1: (ADEP, STAND, RUNWAY)
    g1 = dep.groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")
    # L2: (ADEP, STAND)
    g2 = dep.groupby(["ADEP_mvt", "STAND_mvt"])["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")
    # L3: (ADEP,)
    g3 = dep.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")

    # 把小样本的 L1 标 NaN, 让 lookup 回退
    g1.loc[g1["n"] < min_obs, "q"] = np.nan

    lut = {
        "quantile": quantile,
        "min_sec": min_sec, "max_sec": max_sec, "min_obs": min_obs,
        "L1_q": g1["q"], "L1_n": g1["n"],
        "L2_q": g2["q"], "L2_n": g2["n"],
        "L3_q": g3["q"], "L3_n": g3["n"],
    }
    return lut


def lookup_t_unimp_clean(df, lut):
    """用干净 LUT 查 t_unimp."""
    n = len(df)
    vals = np.full(n, np.nan, dtype=np.float32)
    levels = np.full(n, "", dtype=object)

    dep_mask = (df["PHASE_mvt"] == "DEP").values
    dep_idx = np.where(dep_mask)[0]
    sub = df.iloc[dep_idx]

    ap = sub["ADEP_mvt"].astype(str).values
    st = sub["STAND_mvt"].astype(str).values
    rw = sub["RUNWAY_mvt"].astype(str).values

    result = np.full(len(dep_idx), np.nan, dtype=np.float32)
    used = np.full(len(dep_idx), "", dtype=object)

    # L1
    keys1 = pd.MultiIndex.from_arrays([ap, st, rw])
    v1 = lut["L1_q"].reindex(keys1).values
    filled = ~np.isnan(v1)
    result[filled] = v1[filled]
    used[filled] = "L1"

    # L2
    need = np.isnan(result)
    if need.sum() > 0:
        keys2 = pd.MultiIndex.from_arrays([ap[need], st[need]])
        v2 = lut["L2_q"].reindex(keys2).values
        filled2 = ~np.isnan(v2)
        idx_need = np.where(need)[0]
        result[idx_need[filled2]] = v2[filled2]
        used[idx_need[filled2]] = "L2"

    # L3
    need = np.isnan(result)
    if need.sum() > 0:
        idx_need = np.where(need)[0]
        v3 = lut["L3_q"].reindex(ap[need]).values
        filled3 = ~np.isnan(v3)
        result[idx_need[filled3]] = v3[filled3]
        used[idx_need[filled3]] = "L3"

    # 全局回退 (L3 也查不到的冷机场): 用整个训练集干净 q30
    need = np.isnan(result)
    if need.sum() > 0:
        global_q = lut.get("global_q30", 768.0)
        idx_need = np.where(need)[0]
        result[idx_need] = global_q
        used[idx_need] = "GLOBAL"

    vals[dep_idx] = result
    levels[dep_idx] = used
    return vals, levels


# ============================================================
# 1. 加载训练/holdout
# ============================================================
print("=" * 70)
print("[1] 加载 holdout + 训练数据")
print("=" * 70)

ho_dfs = []
for m in [1, 7]:
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)

months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months:
    fname = (f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
             if m < 12 else "training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)

# 训练 DEP (用于构建 LUT)
df_tr_dep = df_tr[
    (df_tr["PHASE_mvt"] == "DEP") &
    (df_tr["TAXITIME_SEC_mvt"] > 0) &
    (df_tr["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  训练 DEP: {len(df_tr_dep):,}")

# ============================================================
# 2. 构建干净 LUT
# ============================================================
print()
print("=" * 70)
print("[2] 构建干净 LUT (min=120, max=3600, min_obs=10)")
print("=" * 70)
lut_clean = build_t_unimp_lut_clean(df_tr_dep, min_sec=120, max_sec=3600, min_obs=10)
# 全局 q30
clean_data = df_tr_dep[(df_tr_dep["TAXITIME_SEC_mvt"] >= 60) & (df_tr_dep["TAXITIME_SEC_mvt"] <= 3600)]
lut_clean["global_q30"] = clean_data["TAXITIME_SEC_mvt"].quantile(0.3)
print(f"  LUT clean: L1有效={lut_clean['L1_q'].notna().sum()}/{len(lut_clean['L1_q'])}, "
      f"L2有效={lut_clean['L2_q'].notna().sum()}/{len(lut_clean['L2_q'])}, "
      f"L3有效={lut_clean['L3_q'].notna().sum()}/{len(lut_clean['L3_q'])}")
print(f"  L1 q30 范围: min={lut_clean['L1_q'].min():.0f}, p10={lut_clean['L1_q'].quantile(0.1):.0f}, "
      f"p50={lut_clean['L1_q'].median():.0f}, max={lut_clean['L1_q'].max():.0f}")
print(f"  L2 q30 范围: min={lut_clean['L2_q'].min():.0f}, p50={lut_clean['L2_q'].median():.0f}, max={lut_clean['L2_q'].max():.0f}")
print(f"  L3 q30 范围: min={lut_clean['L3_q'].min():.0f}, p50={lut_clean['L3_q'].median():.0f}, max={lut_clean['L3_q'].max():.0f}")

# ============================================================
# 3. 在 holdout 上查询干净 t_unimp
# ============================================================
print()
print("=" * 70)
print("[3] Holdout 干净 t_unimp 查询")
print("=" * 70)
df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_ho_dep = df_ho_dep[
    (df_ho_dep["TAXITIME_SEC_mvt"] > 0) &
    (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)
].reset_index(drop=True)
print(f"  Holdout DEP: {len(df_ho_dep):,}")

t_vals, t_src = lookup_t_unimp_clean(df_ho_dep, lut_clean)
print(f"\n干净 t_unimp 分布:")
v = ~np.isnan(t_vals)
print(f"  min={np.nanmin(t_vals[v]):.0f}, p1={np.percentile(t_vals[v],1):.0f}, "
      f"p10={np.percentile(t_vals[v],10):.0f}, p50={np.nanmedian(t_vals[v]):.0f}, "
      f"p99={np.percentile(t_vals[v],99):.0f}, max={np.nanmax(t_vals[v]):.0f}")

print(f"\n异常值检查:")
print(f"  < 306: {(t_vals < 306).sum()} ({100*(t_vals<306).sum()/len(t_vals):.2f}%)")
print(f"  < 100: {(t_vals < 100).sum()}")
print(f"  > 2752: {(t_vals > 2752).sum()}")

print(f"\n按 level 分布:")
import collections
ctr = collections.Counter(t_src)
for k, v in ctr.most_common():
    print(f"  {k}: {v}")

# ============================================================
# 4. 对比: 旧 LUT (脏) vs 新 LUT (干净)
# ============================================================
print()
print("=" * 70)
print("[4] 旧 vs 新 LUT 对比")
print("=" * 70)
lut_old = L.build_t_unimp_lut(df_tr_dep)
t_old, _ = L.lookup_t_unimp(df_ho_dep, lut_old)
print(f"\n{'指标':30s} {'旧 LUT':>15s} {'新 LUT':>15s}")
print("-" * 65)
print(f"{'min':30s} {np.nanmin(t_old):>15.0f} {np.nanmin(t_vals):>15.0f}")
print(f"{'p50':30s} {np.nanmedian(t_old):>15.0f} {np.nanmedian(t_vals):>15.0f}")
print(f"{'max':30s} {np.nanmax(t_old):>15.0f} {np.nanmax(t_vals):>15.0f}")
print(f"{'< 306 占比':30s} {100*(t_old<306).sum()/len(t_old):>14.2f}% {100*(t_vals<306).sum()/len(t_vals):>14.2f}%")
print(f"{'< 100 个数':30s} {(t_old<100).sum():>15d} {(t_vals<100).sum():>15d}")
print(f"{'匹配分裂 min=306':30s} {'❌':>15s} {'✅' if np.nanmin(t_vals)>=300 else '❌':>15s}")