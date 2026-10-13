"""
干净+截断版 LUT:
- 过滤物理不可能值 (min_sec, max_sec)
- min_obs 阈值
- L1/L2/L3 q30 限制在 [phys_min, phys_max] 内, 超出则视为 missing
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

import _r7_lib as L
from src.data_loader import load_month, parse_time_columns
from src.train_xgb import _encode_categoricals

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


def build_t_unimp_lut_clean(df_train, min_sec=120, max_sec=3600,
                              min_obs=10, quantile=0.3,
                              clamp_lo=300, clamp_hi=2800):
    """干净版 LUT + 截断
      - 训练过滤: TAXITIME in [min_sec, max_sec]
      - obs 阈值: L1 group < min_obs 时 q30 标 NaN
      - 截断: q30 < clamp_lo 或 > clamp_hi 标 NaN (让样本走 missing)
      - 全局兜底: 用干净 q30 中位数
    """
    dep = df_train[df_train["PHASE_mvt"] == "DEP"].copy()
    dep_clean = dep[(dep["TAXITIME_SEC_mvt"] >= min_sec) & (dep["TAXITIME_SEC_mvt"] <= max_sec)]

    g1 = dep_clean.groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")
    g2 = dep_clean.groupby(["ADEP_mvt", "STAND_mvt"])["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")
    g3 = dep_clean.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].agg(
        q=lambda x: x.quantile(quantile), n="count")

    # 小样本 → NaN (让 lookup 回退)
    g1.loc[g1["n"] < min_obs, "q"] = np.nan
    g2.loc[g2["n"] < min_obs, "q"] = np.nan
    g3.loc[g3["n"] < min_obs, "q"] = np.nan

    # 截断到物理范围
    for g in [g1, g2, g3]:
        g.loc[(g["q"] < clamp_lo) | (g["q"] > clamp_hi), "q"] = np.nan

    lut = {
        "quantile": quantile, "min_sec": min_sec, "max_sec": max_sec,
        "min_obs": min_obs, "clamp_lo": clamp_lo, "clamp_hi": clamp_hi,
        "L1_q": g1["q"], "L1_n": g1["n"],
        "L2_q": g2["q"], "L2_n": g2["n"],
        "L3_q": g3["q"], "L3_n": g3["n"],
        "global_q30": float(dep_clean["TAXITIME_SEC_mvt"].quantile(quantile)),
    }
    return lut


def lookup_t_unimp_clean(df, lut):
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

    keys1 = pd.MultiIndex.from_arrays([ap, st, rw])
    v1 = lut["L1_q"].reindex(keys1).values
    filled = ~np.isnan(v1)
    result[filled] = v1[filled]
    used[filled] = "L1"

    need = np.isnan(result)
    if need.sum() > 0:
        keys2 = pd.MultiIndex.from_arrays([ap[need], st[need]])
        v2 = lut["L2_q"].reindex(keys2).values
        filled2 = ~np.isnan(v2)
        idx_need = np.where(need)[0]
        result[idx_need[filled2]] = v2[filled2]
        used[idx_need[filled2]] = "L2"

    need = np.isnan(result)
    if need.sum() > 0:
        idx_need = np.where(need)[0]
        v3 = lut["L3_q"].reindex(ap[need]).values
        filled3 = ~np.isnan(v3)
        result[idx_need[filled3]] = v3[filled3]
        used[idx_need[filled3]] = "L3"

    need = np.isnan(result)
    if need.sum() > 0:
        idx_need = np.where(need)[0]
        result[idx_need] = lut["global_q30"]
        used[idx_need] = "GLOBAL"

    vals[dep_idx] = result
    levels[dep_idx] = used
    return vals, levels


# ============================================================
# 1. 加载 holdout + 训练
# ============================================================
print("=" * 70)
print("[1] 加载数据")
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
df_tr_dep = df_tr[
    (df_tr["PHASE_mvt"] == "DEP") &
    (df_tr["TAXITIME_SEC_mvt"] > 0) &
    (df_tr["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  训练 DEP: {len(df_tr_dep):,}")

# ============================================================
# 2. 构建干净+截断 LUT
# ============================================================
print()
print("=" * 70)
print("[2] 构建干净+截断 LUT")
print("=" * 70)
lut_clean = build_t_unimp_lut_clean(df_tr_dep, min_sec=120, max_sec=3600,
                                      min_obs=10, quantile=0.3,
                                      clamp_lo=80, clamp_hi=4000)
print(f"  L1 有效: {lut_clean['L1_q'].notna().sum()}/{len(lut_clean['L1_q'])}")
print(f"  L2 有效: {lut_clean['L2_q'].notna().sum()}/{len(lut_clean['L2_q'])}")
print(f"  L3 有效: {lut_clean['L3_q'].notna().sum()}/{len(lut_clean['L3_q'])}")
print(f"  L1 q30 范围: min={lut_clean['L1_q'].min():.0f}, p50={lut_clean['L1_q'].median():.0f}, max={lut_clean['L1_q'].max():.0f}")
print(f"  Global q30: {lut_clean['global_q30']:.0f}s")

# ============================================================
# 3. holdout 查询
# ============================================================
print()
print("=" * 70)
print("[3] Holdout 查询 + 分裂对比")
print("=" * 70)

# 模型 + 分裂阈值
bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
trees = bst.trees_to_dataframe()
sp = trees[trees["Feature"] == "t_unimp_r7"]["Split"].values
sp_min, sp_max = float(sp.min()), float(sp.max())
print(f"  模型 t_unimp 分裂: min={sp_min:.0f}, p50={np.median(sp):.0f}, max={sp_max:.0f}")

df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_ho_dep = df_ho_dep[
    (df_ho_dep["TAXITIME_SEC_mvt"] > 0) &
    (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)
].reset_index(drop=True)
t_vals, t_src = lookup_t_unimp_clean(df_ho_dep, lut_clean)
df_ho_dep["t_unimp_r7"] = t_vals

v = ~np.isnan(t_vals)
print(f"\n  Holdout t_unimp 分布:")
print(f"    min={np.nanmin(t_vals[v]):.0f}, p1={np.percentile(t_vals[v],1):.0f}, "
      f"p50={np.nanmedian(t_vals[v]):.0f}, p99={np.percentile(t_vals[v],99):.0f}, max={np.nanmax(t_vals[v]):.0f}")
print(f"  < sp_min: {(t_vals < sp_min).sum()} ({100*(t_vals<sp_min).sum()/len(t_vals):.2f}%)")
print(f"  > sp_max: {(t_vals > sp_max).sum()}")
print(f"\n  按 level:")
import collections
ctr = collections.Counter(t_src)
for k, v in ctr.most_common():
    print(f"    {k}: {v}")

# ============================================================
# 4. 推理 — 只替换 t_unimp, 其他特征用旧管线
# ============================================================
print()
print("=" * 70)
print("[4] 推理 RMSE 对比 (只换 t_unimp)")
print("=" * 70)

import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)

# 取旧 holdout 全套特征 (用旧 LUT)
# 把新 t_unimp 替换到 hr7.df_ho_full
dep_mask_full = (hr7.df_ho_full["PHASE_mvt"] == "DEP").values

# hr7.df_ho_full 经过 TAXITIME 0<x<10800 过滤, 但我们的 df_ho_dep 也过滤过, 应当一致
# 但保险起见用 MVT_ID 对齐
new_t_map = dict(zip(df_ho_dep["MVT_ID_mvt"].values, t_vals))
hr7.df_ho_full["t_unimp_r7"] = hr7.df_ho_full["MVT_ID_mvt"].map(new_t_map).astype(np.float32)
# ARR 行无 t_unimp, 留 NaN

# cat 编码
tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

FEATS69 = bst.feature_names
df_ho_dep_full = hr7.df_ho_full[dep_mask_full].copy().reset_index(drop=True)
X = df_ho_dep_full[FEATS69].copy()
X_enc = _encode_categoricals(X, fit=False).values.astype(np.float32)

y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
valid = (y > 0) & (y < 10800) & np.isfinite(y)
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
rmse_new = float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2)))
print(f"  新 LUT RMSE = {rmse_new:.2f}")
print(f"  旧 LUT RMSE = 332.79 (基线)")
print(f"  Δ = {rmse_new - 332.79:+.2f}")

# ============================================================
# 5. 也对比旧 LUT 在 holdout 上的 max/min
# ============================================================
print()
print("=" * 70)
print("[5] 旧 vs 新 t_unimp 分布对比")
print("=" * 70)
lut_old = L.build_t_unimp_lut(df_tr_dep)
t_old, _ = L.lookup_t_unimp(df_ho_dep, lut_old)
print(f"{'指标':30s} {'旧 LUT':>15s} {'新 LUT (干净+截断)':>20s}")
print("-" * 70)
print(f"{'min (秒)':30s} {np.nanmin(t_old):>15.0f} {np.nanmin(t_vals):>20.0f}")
print(f"{'p50 (秒)':30s} {np.nanmedian(t_old):>15.0f} {np.nanmedian(t_vals):>20.0f}")
print(f"{'max (秒)':30s} {np.nanmax(t_old):>15.0f} {np.nanmax(t_vals):>20.0f}")
print(f"{'< 306 (sp_min) 占比':30s} {100*(t_old<sp_min).sum()/len(t_old):>14.2f}% {100*(t_vals<sp_min).sum()/len(t_vals):>19.2f}%")
print(f"{'> 2752 (sp_max) 占比':30s} {100*(t_old>sp_max).sum()/len(t_old):>14.2f}% {100*(t_vals>sp_max).sum()/len(t_vals):>19.2f}%")
print(f"{'< 100 个数':30s} {(t_old<100).sum():>15d} {(t_vals<100).sum():>20d}")
print(f"{'匹配 sp 范围':30s} {'❌':>15s} {'✅' if (np.nanmin(t_vals)>=sp_min and np.nanmax(t_vals)<=sp_max) else '⚠️':>20s}")