"""
在 holdout 上测试 pos_rw 的几个候选定义, 看哪个能让 RMSE 下降最多
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

# 复用 holdout 特征重建
import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)

# 模型 + 编码
bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names
feat_idx = {f: i for i, f in enumerate(FEATS69)}

tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

# 取 DEP + 编码
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
print(f"基准 RMSE = {base:.2f}\n")

# ============================================================
# 候选 1: pos_rw = 机场级队列积分 (AOBT→MVT, 只DEP)
# ============================================================
print("[候选 1] pos_rw = 机场级队列积分 (AOBT→MVT, 只DEP)")

aobt = pd.to_datetime(df_ho_dep["AOBT_3_flt"], utc=True, errors="coerce").values
mvt = pd.to_datetime(df_ho_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").values
aep = df_ho_dep["ADEP_mvt"].astype(str).values
aobt_i = aobt.astype("datetime64[ns]").view("i8")
mvt_i = mvt.astype("datetime64[ns]").view("i8")
iNaT = np.iinfo(np.int64).min
aobt_v = aobt_i != iNaT
mvt_v = mvt_i != iNaT
mvt_arr = mvt_i.copy()
mvt_arr[~mvt_v] = np.iinfo(np.int64).max
n = len(df_ho_dep)
integral = np.full(n, np.nan, dtype=np.float32)

for ap in np.unique(aep):
    mask = aep == ap
    idx = np.where(mask)[0]
    if len(idx) < 10:
        continue
    aobt_ap = aobt_i[mask]; mvt_ap = mvt_arr[mask]
    aobt_v_ap = aobt_v[mask]; mvt_v_ap = mvt_v[mask]
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
    for i in range(len(idx)):
        if not aobt_v_ap[i] or not mvt_v_ap[i]:
            continue
        t0 = aobt_ap[i]; t1 = mvt_ap[i]
        if t0 >= t1:
            integral[idx[i]] = 0.0; continue
        left = np.searchsorted(times_sorted, t0, side="left")
        right = np.searchsorted(times_sorted, t1, side="right") - 1
        if left > right:
            qs = cum[left-1] if left > 0 else 0
            integral[idx[i]] = float(qs * (t1 - t0) / 1e9); continue
        qs = cum[left-1] if left > 0 else 0
        int_t = 0.0
        if right > left:
            int_t += (prefix_int[right] - prefix_int[left])
        int_t += (times_sorted[left] - t0) * qs
        int_t += (t1 - times_sorted[right]) * cum[right]
        integral[idx[i]] = float(int_t / 1e9)

v = ~np.isnan(integral) & (integral > 0)
print(f"  积分量: p50={np.nanmedian(integral[v]):.0f}, p90={np.percentile(integral[v],90):.0f}, max={np.nanmax(integral[v]):.0f}")

X_mod = X_enc.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(integral, nan=-1).astype(np.float32)
hf = X_mod[:, feat_idx["mvt_na_heavy_frac"]]
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(integral, nan=0).astype(np.float32)
r = rmse(X_mod)
print(f"  RMSE = {r:.2f} (Δ={r-base:+.2f})")

# ============================================================
# 候选 2: pos_rw = dep_queue_n_airport × taxi_time
# ============================================================
print("\n[候选 2] pos_rw = dep_queue_n_airport × taxi_time")
q_ap = pd.to_numeric(df_ho_dep["dep_queue_n_airport"], errors="coerce").values
taxi = (mvt_i - aobt_i) / 1e9
taxi[~(aobt_v & mvt_v)] = np.nan
prod = q_ap * taxi
v = ~np.isnan(prod) & (prod > 0)
print(f"  乘积: p50={np.nanmedian(prod[v]):.0f}, max={np.nanmax(prod[v]):.0f}")

X_mod = X_enc.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(prod, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(prod, nan=0).astype(np.float32)
r = rmse(X_mod)
print(f"  RMSE = {r:.2f} (Δ={r-base:+.2f})")

# ============================================================
# 候选 3: pos_rw = 跑道级积分 (enter = AOBT + t_unimp)
# ============================================================
print("\n[候选 3] pos_rw = 跑道级积分 (enter = AOBT + t_unimp)")
rw = df_ho_dep["RUNWAY_mvt"].astype(str).values
t_unimp = hr7.t_vals[dep_mask]
t_unimp_ns = (t_unimp * 1e9).astype("int64")
enter = aobt_i + t_unimp_ns
enter_v = aobt_v & ~np.isnan(t_unimp) & mvt_v
groups = np.char.add(np.char.add(aep.astype(str), "_"), rw.astype(str))
int_rw = np.full(n, np.nan, dtype=np.float32)

for g in np.unique(groups):
    mask = groups == g
    idx = np.where(mask)[0]
    if len(idx) < 2:
        continue
    ent_g = enter[mask]; mvt_g = mvt_arr[mask]
    ent_vg = enter_v[mask]; mvt_vg = mvt_v[mask]
    both = ent_vg & mvt_vg
    n_ent = int(ent_vg.sum()); n_lev = int(both.sum())
    if n_ent == 0:
        continue
    ev_t = np.concatenate([ent_g[ent_vg], mvt_g[both]])
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
    for i in range(len(idx)):
        if not ent_vg[i] or not mvt_vg[i]:
            continue
        t0 = ent_g[i]; t1 = mvt_g[i]
        if t0 >= t1:
            int_rw[idx[i]] = 0.0; continue
        left = np.searchsorted(times_sorted, t0, side="left")
        right = np.searchsorted(times_sorted, t1, side="right") - 1
        if left > right:
            qs = cum[left-1] if left > 0 else 0
            int_rw[idx[i]] = float(qs * (t1 - t0) / 1e9); continue
        qs = cum[left-1] if left > 0 else 0
        int_t = 0.0
        if right > left:
            int_t += (prefix_int[right] - prefix_int[left])
        int_t += (times_sorted[left] - t0) * qs
        int_t += (t1 - times_sorted[right]) * cum[right]
        int_rw[idx[i]] = float(int_t / 1e9)

v = ~np.isnan(int_rw) & (int_rw > 0)
print(f"  跑道积分: p50={np.nanmedian(int_rw[v]):.0f}, max={np.nanmax(int_rw[v]):.0f}")

X_mod = X_enc.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(int_rw, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(int_rw, nan=0).astype(np.float32)
r = rmse(X_mod)
print(f"  RMSE = {r:.2f} (Δ={r-base:+.2f})")

# ============================================================
# 候选 4: pos_rw = mvt_q_peak × t_unimp
# ============================================================
print("\n[候选 4] pos_rw = mvt_q_peak × t_unimp")
qp = pd.to_numeric(df_ho_dep["mvt_q_peak"], errors="coerce").values
prod4 = qp * t_unimp
v = ~np.isnan(prod4) & (prod4 > 0)
print(f"  乘积: p50={np.nanmedian(prod4[v]):.0f}, max={np.nanmax(prod4[v]):.0f}")

X_mod = X_enc.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(prod4, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(prod4, nan=0).astype(np.float32)
r = rmse(X_mod)
print(f"  RMSE = {r:.2f} (Δ={r-base:+.2f})")