"""
对比: MVT 特征的模型分裂点 vs 重建特征值
目标: 找出哪些特征还有定义偏差
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
trees = bst_r7.trees_to_dataframe()

# 用训练数据构建 LUT
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

# ============================================================
# 对比分裂点 vs 实际值
# ============================================================
print(f"\n{'='*80}")
print(f"{'特征':28s} {'分裂点 p50':>10s} {'特征 p50':>10s} {'p50 比率':>10s} "
      f"{'分裂点 max':>12s} {'特征 max':>12s} {'max 比率':>10s}")
print("-" * 80)

mvt_feats = sorted([f for f in FEATS69 if f.startswith("mvt_")])

for feat in mvt_feats:
    splits = trees[trees['Feature'] == feat]['Split'].values
    if len(splits) == 0:
        continue
    
    feat_vals = pd.to_numeric(df_dep[feat], errors="coerce").values
    valid = ~np.isnan(feat_vals) & np.isfinite(feat_vals)
    if np.sum(valid) == 0:
        print(f"{feat:28s} {'N/A':>10s} {'NaN':>10s}")
        continue
    
    s_p50 = np.median(splits)
    f_p50 = np.nanmedian(feat_vals[valid])
    s_max = np.max(splits)
    f_max = np.nanmax(feat_vals[valid])
    
    r_p50 = f_p50 / s_p50 if s_p50 != 0 else float('inf')
    r_max = f_max / s_max if s_max != 0 else float('inf')
    
    # 标记异常
    flag = ""
    if r_p50 < 0.5 or r_p50 > 2.0:
        flag = " ⚠️"
    
    print(f"{feat:28s} {s_p50:>10.1f} {f_p50:>10.1f} {r_p50:>10.3f} "
          f"{s_max:>12.1f} {f_max:>12.1f} {r_max:>10.3f}{flag}")

print(f"\n✅ 对比完成")
