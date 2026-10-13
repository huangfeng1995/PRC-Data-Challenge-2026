"""
特征错位表: 从 R7slim 模型 dump 全部分裂阈值, 对照 holdout 上重建特征的分布
定位错位特征, 输出报告

不需要训练时特征 DataFrame —— 用 holdout 重建分布即可
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

from src.train_xgb import _encode_categoricals

# ============================================================
# 1. 加载留存模型, dump 全部 69 特征分裂阈值
# ============================================================
bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names
trees = bst.trees_to_dataframe()

print(f"模型: {len(FEATS69)} 特征, {len(trees)} 行树结构")

# 按特征聚合分裂阈值
splits_per_feat = {}
for f in FEATS69:
    s = trees[trees["Feature"] == f]["Split"].values
    if len(s) > 0:
        splits_per_feat[f] = np.sort(s)

print(f"有分裂的特征数: {len(splits_per_feat)}")
print(f"无分裂的特征数: {len([f for f in FEATS69 if f not in splits_per_feat])}")
no_split = [f for f in FEATS69 if f not in splits_per_feat]
if no_split:
    print(f"  无分裂: {no_split[:10]}...")

# ============================================================
# 2. 重建 holdout 特征 (复用脚本)
# ============================================================
print()
print("=" * 70)
print("[重构 holdout 特征]")
print("=" * 70)

# 直接复用脚本的设置, exec 到 FEATS69 之后
import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)

# 取 holdout DEP + 编码
dep_mask_full = (hr7.df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = hr7.df_ho_full[dep_mask_full].copy().reset_index(drop=True)

# 编码
tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

# === 编码后用 _encode_categoricals 出来的整型/浮点混合值 ===
X = df_ho_dep_full[FEATS69].copy()
X_enc = _encode_categoricals(X, fit=False)
X_arr = X_enc.values.astype(np.float64)  # 用于分布统计
print(f"  X 形状: {X_arr.shape}")

# ============================================================
# 3. 逐特征对照：分布 vs 分裂阈值
# ============================================================
print()
print("=" * 70)
print("[特征错位表 - 重建分布 vs 模型分裂阈值]")
print("=" * 70)
print(f"{'#':>3s} {'特征':28s} {'n_valid':>8s} {'unique':>8s} "
      f"{'min':>10s} {'p25':>10s} {'p50':>10s} {'p75':>10s} {'max':>10s} "
      f"{'splits':>7s} {'sp_min':>10s} {'sp_max':>10s} {'错位':>6s}")
print("-" * 130)

mismatch_table = []
for i, f in enumerate(FEATS69):
    col = X_arr[:, i]
    valid = ~np.isnan(col) & np.isfinite(col)
    n_valid = int(valid.sum())
    if n_valid == 0:
        print(f"{i+1:3d} {f:28s} {'NaN':>8s} {'NaN':>8s}")
        continue
    
    v = col[valid]
    n_unique = int(len(np.unique(v)))
    c_min = float(np.min(v))
    c_p25 = float(np.percentile(v, 25))
    c_p50 = float(np.median(v))
    c_p75 = float(np.percentile(v, 75))
    c_max = float(np.max(v))
    
    # 分裂阈值
    sp = splits_per_feat.get(f, np.array([]))
    n_sp = len(sp)
    sp_min = float(np.min(sp)) if n_sp > 0 else np.nan
    sp_max = float(np.max(sp)) if n_sp > 0 else np.nan
    
    # 错位判断:
    #   1. 重建值域全在分裂阈值之外
    #   2. 重建值无变化 (unique == 1)
    #   3. 分裂阈值与重建值尺度差 > 100 倍
    flag = ""
    if n_sp > 0:
        # 检查重建值域是否覆盖分裂阈值范围
        overlap = not (c_max < sp_min or c_min > sp_max)
        if not overlap and n_sp > 0:
            flag = "❌ OFF"
        elif n_unique == 1:
            flag = "⚠️ CONST"
        elif c_max > 0 and sp_max > 0:
            ratio = max(c_max / sp_max, sp_max / c_max) if sp_max > 0 and c_max > 0 else 1
            if ratio > 100:
                flag = f"⚠️ ×{ratio:.0f}"
    
    if flag or n_unique <= 5:
        print(f"{i+1:3d} {f:28s} {n_valid:>8d} {n_unique:>8d} "
              f"{c_min:>10.2f} {c_p25:>10.2f} {c_p50:>10.2f} {c_p75:>10.2f} {c_max:>10.2f} "
              f"{n_sp:>7d} {sp_min:>10.2f} {sp_max:>10.2f} {flag:>6s}")
        mismatch_table.append({
            "feat": f, "n_valid": n_valid, "n_unique": n_unique,
            "c_min": c_min, "c_p50": c_p50, "c_max": c_max,
            "sp_min": sp_min, "sp_max": sp_max,
            "flag": flag, "n_splits": n_sp
        })

print()
print(f"嫌疑特征数: {len(mismatch_table)}")

# 保存完整报告
all_rows = []
for i, f in enumerate(FEATS69):
    col = X_arr[:, i]
    valid = ~np.isnan(col) & np.isfinite(col)
    n_valid = int(valid.sum())
    if n_valid == 0:
        all_rows.append({"feat": f, "n_valid": 0})
        continue
    v = col[valid]
    sp = splits_per_feat.get(f, np.array([]))
    all_rows.append({
        "feat": f,
        "n_valid": n_valid,
        "n_unique": int(len(np.unique(v))),
        "c_min": float(np.min(v)), "c_p25": float(np.percentile(v, 25)),
        "c_p50": float(np.median(v)), "c_p75": float(np.percentile(v, 75)),
        "c_max": float(np.max(v)),
        "n_splits": len(sp),
        "sp_min": float(np.min(sp)) if len(sp) > 0 else None,
        "sp_p50": float(np.median(sp)) if len(sp) > 0 else None,
        "sp_max": float(np.max(sp)) if len(sp) > 0 else None,
    })
df_all = pd.DataFrame(all_rows)
df_all.to_csv(ROOT / "notebooks/diag/d0/feature_mismatch_table.csv", index=False)
print(f"\n完整表已保存: feature_mismatch_table.csv ({len(df_all)} 行)")

# 保存嫌疑特征
df_mismatch = pd.DataFrame(mismatch_table)
df_mismatch.to_csv(ROOT / "notebooks/diag/d0/feature_mismatch_suspects.csv", index=False)
print(f"嫌疑特征表: feature_mismatch_suspects.csv ({len(df_mismatch)} 行)")