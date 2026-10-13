"""
测试: mvt_q_peak 改成"机场级 + ARR + DEP + AOBT 之前 enter"
看 RMSE 变化
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

import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)

bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names
feat_idx = {f: i for i, f in enumerate(FEATS69)}

tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

dep_mask = (hr7.df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep = hr7.df_ho_full[dep_mask].copy().reset_index(drop=True)
X = df_ho_dep[FEATS69].copy()
X_enc = tx._encode_categoricals(X, fit=False).values.astype(np.float32)

y = df_ho_dep["TAXITIME_SEC_mvt"].values
valid = (y > 0) & (y < 10800) & np.isfinite(y)

def rmse(X_mod):
    dmat = xgb.DMatrix(X_mod, feature_names=FEATS69)
    pred = bst.predict(dmat)
    pred = np.clip(pred, 30, 10800)
    return float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2)))

base = rmse(X_enc)
print(f"基准 RMSE = {base:.2f}")

# 用 ranking 相同的全数据 (DEP+ARR) 重新计算 q_peak/q_mean
df_full = hr7.df_ho.copy()  # DEP+ARR
df_full = df_full[df_full["PHASE_mvt"].isin(["DEP", "ARR"])].copy().reset_index(drop=True)

aobt_all = pd.to_datetime(df_full["AOBT_3_flt"], utc=True, errors="coerce").values
mvt_all = pd.to_datetime(df_full["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").values
aep_all = df_full["ADEP_mvt"].astype(str).values
phase_all = df_full["PHASE_mvt"].astype(str).values

aobt_i = aobt_all.astype("datetime64[ns]").view("i8")
mvt_i = mvt_all.astype("datetime64[ns]").view("i8")
iNaT = np.iinfo(np.int64).min
aobt_v = aobt_i != iNaT
mvt_v = mvt_i != iNaT
mvt_arr = mvt_i.copy()
mvt_arr[~mvt_v] = np.iinfo(np.int64).max
n_full = len(df_full)
new_q_peak = np.full(n_full, np.nan, dtype=np.float32)
new_q_mean = np.full(n_full, np.nan, dtype=np.float32)

# 机场级分组 (含全部 DEP+ARR)
print("计算 q_peak/q_mean (含全部 DEP+ARR)...")
for ap in np.unique(aep_all):
    mask = aep_all == ap
    idx = np.where(mask)[0]
    if len(idx) < 10:
        continue
    aobt_ap = aobt_i[mask]; mvt_ap = mvt_arr[mask]
    aobt_v_ap = aobt_v[mask]; mvt_v_ap = mvt_v[mask]
    phase_ap = phase_all[mask]
    both = aobt_v_ap & mvt_v_ap
    n_ent = int(aobt_v_ap.sum()); n_lev = int(both.sum())
    if n_ent == 0:
        continue
    ev_t = np.concatenate([aobt_ap[aobt_v_ap], mvt_ap[both]])
    ev_k = np.concatenate([np.ones(n_ent, dtype="int8"), -np.ones(n_lev, dtype="int8")])
    order = np.lexsort((-ev_k.astype(int), ev_t))
    times_sorted = ev_t[order]; cum = np.cumsum(ev_k[order].astype("int32"))
    dt = np.diff(times_sorted.astype("int64"))
    if len(dt) > 0:
        q_vals = cum[:-1].astype(np.float64)
        int_vals = dt.astype(np.float64) * q_vals
        prefix_int = np.concatenate([[0.0], np.cumsum(int_vals)])
    else:
        prefix_int = np.array([0.0])
    dep_pos = np.where(phase_ap == "DEP")[0]
    for i in dep_pos:
        t0 = aobt_ap[i] if aobt_v_ap[i] else mvt_ap[i]
        t1 = mvt_ap[i]
        if t0 >= t1 or not mvt_v_ap[i]:
            continue
        left = np.searchsorted(times_sorted, t0, side="left")
        right = np.searchsorted(times_sorted, t1, side="right") - 1
        if left > right:
            qs = cum[left-1] if left > 0 else 0
            if aobt_v_ap[i]:
                qs = max(qs - 1, 0)
            new_q_peak[idx[i]] = float(qs); new_q_mean[idx[i]] = float(qs)
            continue
        qs = cum[left-1] if left > 0 else 0
        peak = max(int(qs), int(cum[left: right+1].max()))
        int_t = 0.0
        if right > left:
            int_t += (prefix_int[right] - prefix_int[left])
        int_t += (times_sorted[left] - t0) * qs
        int_t += (t1 - times_sorted[right]) * cum[right]
        dur = t1 - t0
        mean = int_t / dur if dur > 0 else float(qs)
        new_q_peak[idx[i]] = float(max(peak - 1, 0))
        new_q_mean[idx[i]] = float(max(mean - 1, 0.0))

# 把结果对齐到 holdout DEP
dep_full_mask = phase_all == "DEP"
mvt_dep_ids = df_full.loc[dep_full_mask, "MVT_ID_mvt"].values
mvt_dep_ids_in_ho = df_ho_dep["MVT_ID_mvt"].values
new_qp_map = dict(zip(mvt_dep_ids, new_q_peak[dep_full_mask]))
new_qm_map = dict(zip(mvt_dep_ids, new_q_mean[dep_full_mask]))
new_qp = np.array([new_qp_map.get(mid, np.nan) for mid in mvt_dep_ids_in_ho])
new_qm = np.array([new_qm_map.get(mid, np.nan) for mid in mvt_dep_ids_in_ho])

v = ~np.isnan(new_qp)
print(f"  新 q_peak (含 ARR): p50={np.nanmedian(new_qp[v]):.1f}, max={np.nanmax(new_qp[v]):.1f}")
print(f"  新 q_mean (含 ARR): p50={np.nanmedian(new_qm[v]):.1f}, max={np.nanmax(new_qm[v]):.1f}")

X_mod = X_enc.copy()
X_mod[:, feat_idx["mvt_q_peak"]] = np.nan_to_num(new_qp, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_q_mean"]] = np.nan_to_num(new_qm, nan=-1).astype(np.float32)
r = rmse(X_mod)
print(f"\n替换 q_peak+q_mean 后 RMSE = {r:.2f} (Δ={r-base:+.2f})")