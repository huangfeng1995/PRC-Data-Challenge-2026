"""
测试多个 pos_rw 候选定义, 找 RMSE 最低的
"""
import sys
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

spec = importlib.util.spec_from_file_location(
    "ab_module", ROOT / "notebooks/diag/holdout_v1_queue_ab.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)
import _r7_lib as L
import src.train_xgb as tx

bst_r7 = xgb.Booster()
bst_r7.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst_r7.feature_names
feat_idx = {f: i for i, f in enumerate(FEATS69)}

tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = json.loads(
    (ROOT / "models/feature_lists/xgb_v9_Gd_ABD_runway_seed0_val_cat_mappings.json").read_text())

# 构建 LUT
print("构建 LUT...")
train_files = [
    ROOT / "data/raw/training_2025-01-01_2025-02-01.parquet",
    ROOT / "data/raw/training_2025-02-01_2025-03-01.parquet",
    ROOT / "data/raw/training_2025-03-01_2025-04-01.parquet",
]
df_train_list = []
for f in train_files:
    df = pd.read_parquet(f)
    df = ab.parse_time_columns(df, cols=ab.TIME_COLS)
    df_train_list.append(df)
df_train = pd.concat(df_train_list, ignore_index=True)
dep_train = df_train[
    (df_train["PHASE_mvt"] == "DEP") &
    (df_train["TAXITIME_SEC_mvt"] > 0) &
    (df_train["TAXITIME_SEC_mvt"] < 10800)
].copy()
lut = L.build_t_unimp_lut(dep_train)

# 构建 ranking 特征
print("构建 ranking 特征...")
df_rank = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
df_rank = ab.parse_time_columns(df_rank, cols=ab.TIME_COLS)
df_full = ab.build_full_features(df_rank.copy())
dep_mask = df_full["PHASE_mvt"] == "DEP"
df_dep = df_full[dep_mask].copy().reset_index(drop=True)

t_vals, t_src = L.lookup_t_unimp(df_dep, lut)
df_dep["t_unimp_r7"] = t_vals
extra_dict = L.compute_extra_features(df_dep, label="rank")
for k, v in extra_dict.items():
    df_dep[k] = v
mvt_dict = L.compute_mvt_anchor_features(df_dep, t_unimp_vals=t_vals, label="rank")
for k, v in mvt_dict.items():
    df_dep[k] = v

sub_r7 = pd.read_parquet(ROOT / "submissions" / "jubilant-puzzle_r7_v1.parquet")
dep_ids = df_dep["MVT_ID_mvt"].values
sub_map = dict(zip(sub_r7["MVT_ID_mvt"].values, sub_r7["TAXITIME_SEC_mvt"].values))
pred_sub = np.array([sub_map.get(mid, np.nan) for mid in dep_ids])
valid = ~np.isnan(pred_sub)

X_base = df_dep[FEATS69].copy()
X_enc_base = tx._encode_categoricals(X_base.copy(), fit=False).values.astype(np.float32)

def predict_rmse(X_arr):
    dmat = xgb.DMatrix(X_arr, feature_names=FEATS69)
    pred = bst_r7.predict(dmat)
    pred = np.clip(pred, 30, 10800)
    diff = pred[valid] - pred_sub[valid]
    return np.sqrt(np.mean(diff**2)), np.mean(diff)

base_rmse, _ = predict_rmse(X_enc_base)
print(f"基准 RMSE = {base_rmse:.1f}")

# 公共数据
aobt = pd.to_datetime(df_dep["AOBT_3_flt"], utc=True, errors="coerce")
mvt = pd.to_datetime(df_dep["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
aep = df_dep["ADEP_mvt"].astype(str).values
rw = df_dep["RUNWAY_mvt"].astype(str).values

aobt_i = aobt.values.astype("datetime64[ns]").view("i8")
mvt_i = mvt.values.astype("datetime64[ns]").view("i8")
iNaT = np.iinfo(np.int64).min
aobt_v = aobt_i != iNaT
mvt_v = mvt_i != iNaT
mvt_arr = mvt_i.copy()
mvt_arr[~mvt_v] = np.iinfo(np.int64).max
n = len(df_dep)

t_unimp_ns = (t_vals * 1e9).astype("int64")
enter_rw_ns = aobt_i + t_unimp_ns  # 跑道入口时间

def compute_integral(enter_times, enter_valid, mvt_times, mvt_valid, group_keys):
    """计算各组内 [enter_i, mvt_i] 窗口内的队列积分 (架次*秒)"""
    n = len(enter_times)
    result = np.full(n, np.nan, dtype=np.float32)
    
    unique_groups, group_indices = np.unique(group_keys, return_inverse=True)
    
    for g in range(len(unique_groups)):
        mask = group_indices == g
        idx = np.where(mask)[0]
        m_g = len(idx)
        if m_g < 2:
            continue
        
        ent_g = enter_times[mask]
        lev_g = mvt_times[mask]
        ent_v_g = enter_valid[mask]
        lev_v_g = mvt_valid[mask]
        
        both_v = ent_v_g & lev_v_g
        n_ent = int(ent_v_g.sum())
        n_lev = int(both_v.sum())
        
        ev_t = np.concatenate([ent_g[ent_v_g], lev_g[both_v]])
        ev_k = np.concatenate([
            np.ones(n_ent, dtype="int8"),
            -np.ones(n_lev, dtype="int8"),
        ])
        
        order = np.lexsort((-ev_k.astype(int), ev_t))
        times_sorted = ev_t[order]
        cum = np.cumsum(ev_k[order].astype("int32"))
        
        dt = np.diff(times_sorted.astype("int64"))
        if len(dt) > 0:
            q_vals = cum[:-1].astype(np.float64)
            int_vals = dt.astype(np.float64) * q_vals
            prefix_int = np.concatenate([[0.0], np.cumsum(int_vals)])
        else:
            prefix_int = np.array([0.0])
        
        for i in range(m_g):
            if not ent_v_g[i] or not lev_v_g[i]:
                continue
            
            t_start = ent_g[i]
            t_end = lev_g[i]
            if t_start >= t_end:
                result[idx[i]] = 0.0
                continue
            
            left = np.searchsorted(times_sorted, t_start, side="left")
            right = np.searchsorted(times_sorted, t_end, side="right") - 1
            
            if left > right:
                q_start = cum[left - 1] if left > 0 else 0
                result[idx[i]] = float(q_start * (t_end - t_start) / 1e9)
                continue
            
            q_start = cum[left - 1] if left > 0 else 0
            int_total = 0.0
            if right > left:
                int_total += (prefix_int[right] - prefix_int[left])
            int_total += (times_sorted[left] - t_start) * q_start
            int_total += (t_end - times_sorted[right]) * cum[right]
            
            result[idx[i]] = float(int_total / 1e9)
    
    return result

# ============================================================
# 候选 1: 跑道级积分 (enter = AOBT + t_unimp)
# ============================================================
print(f"\n候选 1: 跑道级积分 (enter = AOBT + t_unimp)")
groups_rw = np.char.add(np.char.add(aep.astype(str), "_"), rw.astype(str))
int_rw = compute_integral(enter_rw_ns, aobt_v, mvt_arr, mvt_v, groups_rw)
v = ~np.isnan(int_rw) & (int_rw > 0)
print(f"  p50={np.nanmedian(int_rw[v]):.0f}, p90={np.percentile(int_rw[v], 90):.0f}, max={np.nanmax(int_rw[v]):.0f}")

X_mod = X_enc_base.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(int_rw, nan=-1).astype(np.float32)
hf = X_mod[:, feat_idx["mvt_na_heavy_frac"]]
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(int_rw, nan=0).astype(np.float32)
rmse, _ = predict_rmse(X_mod)
print(f"  RMSE = {rmse:.1f} (Δ={rmse-base_rmse:+.1f})")

# ============================================================
# 候选 2: 跑道级积分 (enter = AOBT)
# ============================================================
print(f"\n候选 2: 跑道级积分 (enter = AOBT)")
int_rw2 = compute_integral(aobt_i, aobt_v, mvt_arr, mvt_v, groups_rw)
v = ~np.isnan(int_rw2) & (int_rw2 > 0)
print(f"  p50={np.nanmedian(int_rw2[v]):.0f}, p90={np.percentile(int_rw2[v], 90):.0f}, max={np.nanmax(int_rw2[v]):.0f}")

X_mod = X_enc_base.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(int_rw2, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(int_rw2, nan=0).astype(np.float32)
rmse, _ = predict_rmse(X_mod)
print(f"  RMSE = {rmse:.1f} (Δ={rmse-base_rmse:+.1f})")

# ============================================================
# 候选 3: 机场级积分 (enter = AOBT, 只DEP)
# ============================================================
print(f"\n候选 3: 机场级积分 (enter = AOBT, 只DEP)")
int_ap = compute_integral(aobt_i, aobt_v, mvt_arr, mvt_v, aep.astype(str))
v = ~np.isnan(int_ap) & (int_ap > 0)
print(f"  p50={np.nanmedian(int_ap[v]):.0f}, p90={np.percentile(int_ap[v], 90):.0f}, max={np.nanmax(int_ap[v]):.0f}")

X_mod = X_enc_base.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(int_ap, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(int_ap, nan=0).astype(np.float32)
rmse, _ = predict_rmse(X_mod)
print(f"  RMSE = {rmse:.1f} (Δ={rmse-base_rmse:+.1f})")

# ============================================================
# 候选 4: dep_queue_n_airport * t_unimp (近似积分)
# ============================================================
print(f"\n候选 4: dep_queue_n_airport * t_unimp")
q_ap = pd.to_numeric(df_dep["dep_queue_n_airport"], errors="coerce").values
prod1 = q_ap * t_vals
v = ~np.isnan(prod1) & (prod1 > 0)
print(f"  p50={np.nanmedian(prod1[v]):.0f}, p90={np.percentile(prod1[v], 90):.0f}, max={np.nanmax(prod1[v]):.0f}")

X_mod = X_enc_base.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(prod1, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(prod1, nan=0).astype(np.float32)
rmse, _ = predict_rmse(X_mod)
print(f"  RMSE = {rmse:.1f} (Δ={rmse-base_rmse:+.1f})")

# ============================================================
# 候选 5: dep_queue_n_airport * (MVT - AOBT)
# ============================================================
print(f"\n候选 5: dep_queue_n_airport * (MVT - AOBT) 秒")
taxi_sec = (mvt_i - aobt_i) / 1e9
taxi_sec[~(aobt_v & mvt_v)] = np.nan
prod2 = q_ap * taxi_sec
v = ~np.isnan(prod2) & (prod2 > 0)
print(f"  p50={np.nanmedian(prod2[v]):.0f}, p90={np.percentile(prod2[v], 90):.0f}, max={np.nanmax(prod2[v]):.0f}")

X_mod = X_enc_base.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(prod2, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(prod2, nan=0).astype(np.float32)
rmse, _ = predict_rmse(X_mod)
print(f"  RMSE = {rmse:.1f} (Δ={rmse-base_rmse:+.1f})")

# ============================================================
# 候选 6: mvt_q_peak * t_unimp
# ============================================================
print(f"\n候选 6: mvt_q_peak * t_unimp")
qp = pd.to_numeric(df_dep["mvt_q_peak"], errors="coerce").values
prod3 = qp * t_vals
v = ~np.isnan(prod3) & (prod3 > 0)
print(f"  p50={np.nanmedian(prod3[v]):.0f}, p90={np.percentile(prod3[v], 90):.0f}, max={np.nanmax(prod3[v]):.0f}")

X_mod = X_enc_base.copy()
X_mod[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(prod3, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(prod3, nan=0).astype(np.float32)
rmse, _ = predict_rmse(X_mod)
print(f"  RMSE = {rmse:.1f} (Δ={rmse-base_rmse:+.1f})")

print(f"\n✅ 测试完成")
