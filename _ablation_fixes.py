"""
消融实验: 逐个修正可疑特征, 看 RMSE 改善
1. 基准 (已修复 t_unimp LUT)
2. q_peak/q_mean 包含 ARR
3. pos_rw 的各种候选定义
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

# 提交值
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

# ============================================================
# 基准
# ============================================================
base_rmse, base_mean = predict_rmse(X_enc_base)
print(f"\n基准 RMSE = {base_rmse:.1f}")

# ============================================================
# 修正 1: q_peak / q_mean 包含 DEP + ARR 全部航班
# ============================================================
print(f"\n{'='*60}")
print("修正 1: q_peak/q_mean 包含全部航班 (DEP+ARR)")

aobt_all = pd.to_datetime(df_rank["AOBT_3_flt"], utc=True, errors="coerce")
mvt_all = pd.to_datetime(df_rank["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
aep_all = df_rank["ADEP_mvt"].astype(str).values
phase_all = df_rank["PHASE_mvt"].astype(str).values

aobt_i_all = aobt_all.values.astype("datetime64[ns]").view("i8")
mvt_i_all = mvt_all.values.astype("datetime64[ns]").view("i8")
iNaT = np.iinfo(np.int64).min
aobt_v_all = aobt_i_all != iNaT
mvt_v_all = mvt_i_all != iNaT

mvt_arr_all = mvt_i_all.copy()
mvt_arr_all[~mvt_v_all] = np.iinfo(np.int64).max

n_all = len(df_rank)
new_q_peak = np.full(n_all, np.nan, dtype=np.float32)
new_q_mean = np.full(n_all, np.nan, dtype=np.float32)

for ap in np.unique(aep_all):
    mask = aep_all == ap
    idx = np.where(mask)[0]
    m_ap = len(idx)
    if m_ap < 10:
        continue
    
    aobt_ap = aobt_i_all[mask]
    mvt_ap = mvt_arr_all[mask]
    aobt_v_ap = aobt_v_all[mask]
    mvt_v_ap = mvt_v_all[mask]
    phase_ap = phase_all[mask]
    
    both_v = aobt_v_ap & mvt_v_ap
    n_ent = int(aobt_v_ap.sum())
    n_lev = int(both_v.sum())
    
    ev_t = np.concatenate([aobt_ap[aobt_v_ap], mvt_ap[both_v]])
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
    
    dep_pos = np.where(phase_ap == "DEP")[0]
    
    for i in dep_pos:
        t_start = aobt_ap[i] if aobt_v_ap[i] else mvt_ap[i]
        t_end = mvt_ap[i]
        if t_start >= t_end or not mvt_v_ap[i]:
            continue
        
        left = np.searchsorted(times_sorted, t_start, side="left")
        right = np.searchsorted(times_sorted, t_end, side="right") - 1
        
        if left > right:
            q_start = cum[left - 1] if left > 0 else 0
            if aobt_v_ap[i]:
                q_start = max(q_start - 1, 0)
            new_q_peak[idx[i]] = float(q_start)
            new_q_mean[idx[i]] = float(q_start)
            continue
        
        q_start = cum[left - 1] if left > 0 else 0
        peak = max(int(q_start), int(cum[left: right + 1].max()))
        
        int_total = 0.0
        if right > left:
            int_total += (prefix_int[right] - prefix_int[left])
        int_total += (times_sorted[left] - t_start) * q_start
        int_total += (t_end - times_sorted[right]) * cum[right]
        
        duration_ns = t_end - t_start
        mean_q = int_total / duration_ns if duration_ns > 0 else float(q_start)
        
        new_q_peak[idx[i]] = float(max(peak - 1, 0))
        new_q_mean[idx[i]] = float(max(mean_q - 1, 0.0))

# 取 DEP 行
dep_mask_all = phase_all == "DEP"
qp_dep = new_q_peak[dep_mask_all]
qm_dep = new_q_mean[dep_mask_all]

print(f"新 q_peak: p50={np.nanmedian(qp_dep):.1f}, max={np.nanmax(qp_dep):.1f}")
print(f"新 q_mean: p50={np.nanmedian(qm_dep):.1f}, max={np.nanmax(qm_dep):.1f}")

X_mod = X_enc_base.copy()
X_mod[:, feat_idx["mvt_q_peak"]] = np.nan_to_num(qp_dep, nan=-1).astype(np.float32)
X_mod[:, feat_idx["mvt_q_mean"]] = np.nan_to_num(qm_dep, nan=-1).astype(np.float32)
rmse1, _ = predict_rmse(X_mod)
print(f"修正后 RMSE = {rmse1:.1f} (Δ={rmse1-base_rmse:+.1f})")

# ============================================================
# 修正 2: 测试 pos_rw = 机场级队列积分 (AOBT→MVT, 含全部航班)
# ============================================================
print(f"\n{'='*60}")
print("修正 2: pos_rw = 机场级队列积分 (含全部 DEP+ARR)")

# 用刚才计算的 q_mean * 时长 ≈ 积分，但我们需要精确积分
# 重新计算机场级积分
integral_ap = np.full(n_all, np.nan, dtype=np.float32)

for ap in np.unique(aep_all):
    mask = aep_all == ap
    idx = np.where(mask)[0]
    m_ap = len(idx)
    if m_ap < 10:
        continue
    
    aobt_ap = aobt_i_all[mask]
    mvt_ap = mvt_arr_all[mask]
    aobt_v_ap = aobt_v_all[mask]
    mvt_v_ap = mvt_v_all[mask]
    phase_ap = phase_all[mask]
    
    both_v = aobt_v_ap & mvt_v_ap
    n_ent = int(aobt_v_ap.sum())
    n_lev = int(both_v.sum())
    
    ev_t = np.concatenate([aobt_ap[aobt_v_ap], mvt_ap[both_v]])
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
    
    dep_pos = np.where(phase_ap == "DEP")[0]
    for i in dep_pos:
        if not aobt_v_ap[i] or not mvt_v_ap[i]:
            continue
        
        t_start = aobt_ap[i]
        t_end = mvt_ap[i]
        if t_start >= t_end:
            integral_ap[idx[i]] = 0.0
            continue
        
        left = np.searchsorted(times_sorted, t_start, side="left")
        right = np.searchsorted(times_sorted, t_end, side="right") - 1
        
        if left > right:
            q_start = cum[left - 1] if left > 0 else 0
            integral_ap[idx[i]] = float(q_start * (t_end - t_start) / 1e9)
            continue
        
        q_start = cum[left - 1] if left > 0 else 0
        int_total = 0.0
        if right > left:
            int_total += (prefix_int[right] - prefix_int[left])
        int_total += (times_sorted[left] - t_start) * q_start
        int_total += (t_end - times_sorted[right]) * cum[right]
        
        integral_ap[idx[i]] = float(int_total / 1e9)

int_ap_dep = integral_ap[dep_mask_all]
valid_int = ~np.isnan(int_ap_dep) & (int_ap_dep > 0)
print(f"机场级积分 (含全部):")
print(f"  p50={np.nanmedian(int_ap_dep[valid_int]):.0f}, "
      f"p90={np.percentile(int_ap_dep[valid_int], 90):.0f}, "
      f"max={np.nanmax(int_ap_dep[valid_int]):.0f}")

X_mod2 = X_enc_base.copy()
X_mod2[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(int_ap_dep, nan=-1).astype(np.float32)
# 同时更新 heavy_x_pos = heavy_frac * pos_rw
old_hxp = X_mod2[:, feat_idx["mvt_na_heavy_x_pos"]]
hf = X_mod2[:, feat_idx["mvt_na_heavy_frac"]]
X_mod2[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(int_ap_dep, nan=0).astype(np.float32)

rmse2, _ = predict_rmse(X_mod2)
print(f"替换 pos_rw + heavy_x_pos 后 RMSE = {rmse2:.1f} (Δ={rmse2-base_rmse:+.1f})")

# ============================================================
# 修正 3: 两个修正叠加
# ============================================================
print(f"\n{'='*60}")
print("修正 3: q_peak+q_mean + pos_rw 全部修正")

X_mod3 = X_enc_base.copy()
X_mod3[:, feat_idx["mvt_q_peak"]] = np.nan_to_num(qp_dep, nan=-1).astype(np.float32)
X_mod3[:, feat_idx["mvt_q_mean"]] = np.nan_to_num(qm_dep, nan=-1).astype(np.float32)
X_mod3[:, feat_idx["mvt_pos_rw"]] = np.nan_to_num(int_ap_dep, nan=-1).astype(np.float32)
X_mod3[:, feat_idx["mvt_na_heavy_x_pos"]] = hf * np.nan_to_num(int_ap_dep, nan=0).astype(np.float32)

rmse3, _ = predict_rmse(X_mod3)
print(f"叠加修正后 RMSE = {rmse3:.1f} (Δ={rmse3-base_rmse:+.1f})")

print(f"\n✅ 消融实验完成")
