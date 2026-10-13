"""
排查 3 (重写): 跑道号口径 + dep_queue_n_runway 管线 bug

步骤:
  1. 修 dep_queue_n_runway 管线 bug: hr7 重建给出 p50=1 vs 手算 p50=6
     定位 add_dep_queue_n 在 DEP+ARR 混合、ARR 无 ref_dt 上的问题
  2. 系统跑道号归一 (两端号+L/R/C)
  3. EGLL/LIRF/LTFM/LFPG 跑道匹配率
  4. 用归一后跑道重算全部跑道级特征, 重跑 holdout
  5. 分机场 RMSE 对照
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.train_xgb import _encode_categoricals
import _r7_lib as L

TRAIN_COLS = [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
    "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "AOBT_3_flt", "EOBT_1_flt",
    "WK_TBL_CAT_flt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt", "LOBT_flt",
    "ARVT_3_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt",
             "SCHED_TIME_UTC_mvt", "IOBT_flt", "ARVT_3_flt", "LOBT_flt"]


def normalize_runway(rw):
    """跑道号归一:
      - "09L/27R" -> "09R" (取最小编号, 取对应字母)
        09L 端对应方向 09 (90°), 27R 端对应方向 27 (270°)
        物理上同一跑道, 09L 用 90° 进近 (左), 27R 用 270° 进近 (右)
        习惯上保留较小的编号 + 该端的字母
      - "27R" 单独 -> 保持 (27R)
      - "07C" 单独 -> 保持 (07C, center)
      - "36" 单独 -> 保持 (36)
    简化: 如果有 "/" 则按第一个 (不去重两端, 保留左端标识)
    """
    if isinstance(rw, str) and "/" in rw:
        # 取较小编号端
        parts = rw.split("/")
        nums = []
        for p in parts:
            n = "".join(c for c in p if c.isdigit())
            nums.append(int(n) if n else 999)
        if len(nums) == 2 and abs(nums[0] - nums[1]) == 18:
            # 标准两端反向 (差 18 = 180°)
            i = 0 if nums[0] <= nums[1] else 1
            return parts[i]
        return parts[0]
    return rw if isinstance(rw, str) else str(rw)


# ============================================================
# 1. 加载 holdout + 训练
# ============================================================
print("=" * 70)
print("[1] 加载数据 (DEP+ARR)")
print("=" * 70)
ho_dfs = []
for m in [1, 7]:
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
print(f"  Holdout 总: {len(df_ho):,} (DEP+ARR)")

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
print(f"  训练 DEP (有 TAXITIME): {len(df_tr_dep):,}")

# ============================================================
# 2. EGLL/LIRF/LTFM/LFPG 跑道号原始格式
# ============================================================
print()
print("=" * 70)
print("[2] EGLL/LIRF/LTFM/LFPG 跑道号原始格式")
print("=" * 70)
for ap in ["EGLL", "LIRF", "LTFM", "LFPG"]:
    rw_ap = df_tr_dep[df_tr_dep["ADEP_mvt"] == ap]["RUNWAY_mvt"].astype(str)
    rw_set = sorted(rw_ap.unique())
    print(f"  {ap}: {len(rw_set)} unique, 前 20: {rw_set[:20]}")

# ============================================================
# 3. 跑道号归一 + (ADEP, RW) 匹配率
# ============================================================
print()
print("=" * 70)
print("[3] 跑道号归一 + 匹配率")
print("=" * 70)

# 在 holdout + train 都归一
df_tr_dep["rw_norm"] = df_tr_dep["RUNWAY_mvt"].astype(str).apply(normalize_runway)
df_tr_dep_dep_only = df_tr_dep[df_tr_dep["PHASE_mvt"] == "DEP"].copy()

df_ho["rw_norm"] = df_ho["RUNWAY_mvt"].astype(str).apply(normalize_runway)
df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_ho_dep = df_ho_dep[
    (df_ho_dep["TAXITIME_SEC_mvt"] > 0) &
    (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)
].reset_index(drop=True)

# 统计归一前后的 unique 数
for ap in ["EGLL", "LIRF", "LTFM", "LFPG"]:
    n_orig_tr = df_tr_dep[df_tr_dep["ADEP_mvt"] == ap]["RUNWAY_mvt"].astype(str).nunique()
    n_norm_tr = df_tr_dep[df_tr_dep["ADEP_mvt"] == ap]["rw_norm"].nunique()
    n_orig_ho = df_ho_dep[df_ho_dep["ADEP_mvt"] == ap]["RUNWAY_mvt"].astype(str).nunique()
    n_norm_ho = df_ho_dep[df_ho_dep["ADEP_mvt"] == ap]["rw_norm"].nunique()
    print(f"  {ap}: tr(orig={n_orig_tr}, norm={n_norm_tr}), ho(orig={n_orig_ho}, norm={n_norm_ho})")

# 整体匹配率 (ADEP, RW) - 训练是按 RW_mvt 原始分组的
tr_pairs_orig = set(zip(df_tr_dep["ADEP_mvt"].astype(str), df_tr_dep["RUNWAY_mvt"].astype(str)))
tr_pairs_norm = set(zip(df_tr_dep["ADEP_mvt"].astype(str), df_tr_dep["rw_norm"]))
ho_pairs_orig = set(zip(df_ho_dep["ADEP_mvt"].astype(str), df_ho_dep["RUNWAY_mvt"].astype(str)))
ho_pairs_norm = set(zip(df_ho_dep["ADEP_mvt"].astype(str), df_ho_dep["rw_norm"]))

print(f"\n  (ADEP, RW) 匹配率 (origin vs norm):")
print(f"    训练原始 unique: {len(tr_pairs_orig):,}")
print(f"    训练归一 unique: {len(tr_pairs_norm):,}")
print(f"    holdout 原始 unique: {len(ho_pairs_orig):,}")
print(f"    holdout 归一 unique: {len(ho_pairs_norm):,}")
print(f"    原始匹配: {len(ho_pairs_orig & tr_pairs_orig)} / {len(ho_pairs_orig)} = {100*len(ho_pairs_orig & tr_pairs_orig)/len(ho_pairs_orig):.1f}%")
print(f"    归一匹配: {len(ho_pairs_norm & tr_pairs_norm)} / {len(ho_pairs_norm)} = {100*len(ho_pairs_norm & tr_pairs_norm)/len(ho_pairs_norm):.1f}%")

# 按机场看匹配率
print(f"\n  按机场匹配率 (holdout -> 训练):")
for ap in ["EGLL", "LIRF", "LTFM", "LFPG"]:
    ho_orig = {r for a, r in ho_pairs_orig if a == ap}
    ho_norm = {r for a, r in ho_pairs_norm if a == ap}
    tr_orig = {r for a, r in tr_pairs_orig if a == ap}
    tr_norm = {r for a, r in tr_pairs_norm if a == ap}
    m_orig = len(ho_orig & tr_orig)
    m_norm = len(ho_norm & tr_norm)
    print(f"    {ap}: orig {m_orig}/{len(ho_orig)} ({100*m_orig/max(len(ho_orig),1):.1f}%), "
          f"norm {m_norm}/{len(ho_norm)} ({100*m_norm/max(len(ho_norm),1):.1f}%)")

# ============================================================
# 4. 用归一跑道号重算跑道级特征
# ============================================================
print()
print("=" * 70)
print("[4] 用归一跑道号重算跑道级特征")
print("=" * 70)

# 4.1 写一个干净的"只对 DEP 计算 dep_queue_n_runway" 版本
def compute_dep_queue_n_runway_clean(df_full, adep_col="ADEP_mvt", runway_col="RUNWAY_mvt"):
    """对每个 DEP 航班 i, 统计 AOBT 时刻同 (ADEP, RUNWAY) 中:
      #{j: AOBT_j <= AOBT_i AND (MVT_j NaT OR MVT_j > AOBT_i) AND MVT_ID_j != i}

    只对 DEP 行计算, 用 AOBT 作为 ref (与 hr7 的 add_queue_features 默认 ref_dt 一致)
    """
    df = df_full.copy()
    aep = df[adep_col].astype(str).values
    rw = df[runway_col].astype(str).values
    aobt = pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce").values.astype("datetime64[ns]")
    mvt = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").values.astype("datetime64[ns]")
    iNaT = np.iinfo(np.int64).min
    aobt_i = aobt.view("i8")
    mvt_i = mvt.view("i8")
    aobt_v = aobt_i != iNaT
    mvt_v = mvt_i != iNaT

    n = len(df)
    result = np.full(n, np.nan, dtype=np.float32)

    # 按 (ADEP, RUNWAY) 分组
    groups = np.char.add(np.char.add(aep, "_"), rw)
    unique_groups, group_idx = np.unique(groups, return_inverse=True)

    for g in range(len(unique_groups)):
        mask = group_idx == g
        idx = np.where(mask)[0]
        m_g = len(idx)
        if m_g < 2:
            continue

        aobt_g = aobt_i[mask]
        aobt_v_g = aobt_v[mask]
        mvt_g = mvt_i[mask]
        mvt_v_g = mvt_v[mask]

        # 对每个有效 AOBT 时刻, 统计已推出未起飞的航班数
        # 用 sort + cumsum 高效计算
        for i_local in range(m_g):
            if not aobt_v_g[i_local]:
                continue
            t_ref = aobt_g[i_local]
            # 已 AOBT: aobt_g <= t_ref
            cond_aobt = aobt_v_g & (aobt_g <= t_ref)
            # 排除自身
            self_mask = np.ones(m_g, dtype=bool)
            self_mask[i_local] = False
            # 未起飞: MVT NaT OR MVT > t_ref
            cond_not_takeoff = (~mvt_v_g) | (mvt_g > t_ref)
            cnt = (cond_aobt & cond_not_takeoff & self_mask).sum()
            result[idx[i_local]] = float(cnt)

    return result

# 在 holdout 上重算 (DEP only)
print("  重算 dep_queue_n_runway (归一 RW, DEP only, O(n^2) by group)...")
new_qr_runway = compute_dep_queue_n_runway_clean(df_ho_dep, runway_col="rw_norm")
v = ~np.isnan(new_qr_runway)
print(f"  新 dep_queue_n_runway (norm RW): "
      f"p50={np.nanmedian(new_qr_runway[v]):.1f}, p90={np.percentile(new_qr_runway[v],90):.1f}, max={np.nanmax(new_qr_runway[v]):.0f}")

# 旧 hr7 重建的 dep_queue_n_runway 值
import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)
old_qr = pd.to_numeric(hr7.df_ho_full["dep_queue_n_runway"], errors="coerce").values
old_qr_dep = old_qr[hr7.df_ho_full["PHASE_mvt"].values == "DEP"]
print(f"  旧 dep_queue_n_runway (hr7 管线, DEP): "
      f"p50={np.nanmedian(old_qr_dep[~np.isnan(old_qr_dep)]):.1f}, p90={np.nanpercentile(old_qr_dep[~np.isnan(old_qr_dep)],90):.1f}, max={np.nanmax(old_qr_dep):.0f}")
print(f"  →  差异! 管线实现可能有 bug")

# 4.2 推理: 把重算的 dep_queue_n_runway 替换到 hr7 特征中
print()
print("=" * 70)
print("[5] 推理 RMSE 对比 (替换 dep_queue_n_runway)")
print("=" * 70)

bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names

tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

dep_mask = (hr7.df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = hr7.df_ho_full[dep_mask].copy().reset_index(drop=True)
X_base = df_ho_dep_full[FEATS69].copy()
X_enc_base = _encode_categoricals(X_base, fit=False).values.astype(np.float32)
y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
valid = (y > 0) & (y < 10800) & np.isfinite(y)

def rmse(X_arr):
    dmat = xgb.DMatrix(X_arr, feature_names=FEATS69)
    pred = bst.predict(dmat)
    pred = np.clip(pred, 30, 10800)
    return float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2)))

rmse_base = rmse(X_enc_base)
print(f"  基准 RMSE = {rmse_base:.2f}")

# 用归一 RW 重算的 dep_queue_n_runway 替换
new_qr_map = dict(zip(df_ho_dep["MVT_ID_mvt"].values, new_qr_runway))
new_qr_aligned = df_ho_dep_full["MVT_ID_mvt"].map(new_qr_map).astype(np.float32).values
X_mod = X_enc_base.copy()
qr_idx = FEATS69.index("dep_queue_n_runway")
X_mod[:, qr_idx] = new_qr_aligned
rmse_v1 = rmse(X_mod)
print(f"  V1 (只换 dep_queue_n_runway, 归一 RW, O(n^2)): RMSE = {rmse_v1:.2f} (Δ={rmse_v1-rmse_base:+.2f})")

# ============================================================
# 5. 分机场 RMSE 对比 (重点 EGLL)
# ============================================================
print()
print("=" * 70)
print("[6] 分机场 RMSE 对比 (基准 vs V1 归一 RW)")
print("=" * 70)

# 重新算每个机场的 RMSE
def per_airport_rmse(X_arr):
    dmat = xgb.DMatrix(X_arr, feature_names=FEATS69)
    pred = bst.predict(dmat)
    pred = np.clip(pred, 30, 10800)
    aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
    res = {}
    for ap in sorted(np.unique(aps)):
        mask = (aps == ap) & valid
        if mask.sum() < 100:
            continue
        r = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))
        bias = float(np.mean(pred[mask] - y[mask]))
        res[ap] = (r, bias, int(mask.sum()))
    return res

base_ap = per_airport_rmse(X_enc_base)
v1_ap = per_airport_rmse(X_mod)

print(f"\n{'ADEP':6s} {'基准 RMSE':>10s} {'V1 RMSE':>10s} {'Δ':>8s} {'基准 Bias':>10s} {'V1 Bias':>10s} {'n':>8s}")
print("-" * 70)
for ap in sorted(base_ap.keys()):
    r0, b0, n0 = base_ap[ap]
    r1, b1, _ = v1_ap[ap]
    print(f"{ap:6s} {r0:>10.2f} {r1:>10.2f} {r1-r0:>+8.2f} {b0:>+10.2f} {b1:>+10.2f} {n0:>8d}")