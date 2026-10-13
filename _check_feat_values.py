"""
快速检查: df_dep 中 MVT 特征的实际值分布
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

# 构建特征
df_rank = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
df_rank = ab.parse_time_columns(df_rank, cols=ab.TIME_COLS)
df_full = ab.build_full_features(df_rank.copy())
dep_mask = df_full["PHASE_mvt"] == "DEP"
df_dep = df_full[dep_mask].copy().reset_index(drop=True)

mask_clean = (df_dep["TAXITIME_SEC_mvt"] > 0) & (df_dep["TAXITIME_SEC_mvt"] < 10800)
df_clean = df_dep[mask_clean].reset_index(drop=True)
lut = L.build_t_unimp_lut(df_clean)

t_vals, _ = L.lookup_t_unimp(df_dep, lut)
df_dep["t_unimp_r7"] = t_vals
extra_dict = L.compute_extra_features(df_dep, label="rank")
for k, v in extra_dict.items():
    df_dep[k] = v
mvt_dict = L.compute_mvt_anchor_features(df_dep, t_unimp_vals=t_vals, label="rank")
for k, v in mvt_dict.items():
    df_dep[k] = v

# 列出所有 mvt_ 开头的列
mvt_cols = [c for c in df_dep.columns if c.startswith("mvt_")]
print(f"MVT 特征列 ({len(mvt_cols)} 个):")
print(sorted(mvt_cols))

# 打印关键特征的统计
print(f"\n{'特征':30s} {'n_valid':>8s} {'min':>10s} {'p50':>10s} {'p90':>10s} {'max':>12s}")
print("-" * 85)

key_feats = sorted(mvt_cols)
for feat in key_feats:
    vals = pd.to_numeric(df_dep[feat], errors="coerce").values
    valid = ~np.isnan(vals)
    n_valid = np.sum(valid)
    if n_valid == 0:
        print(f"{feat:30s} {n_valid:>8d} {'NaN':>10s}")
        continue
    print(f"{feat:30s} {n_valid:>8d} {np.nanmin(vals):>10.1f} "
          f"{np.nanmedian(vals):>10.1f} {np.nanpercentile(vals, 90):>10.1f} "
          f"{np.nanmax(vals):>12.1f}")

# 同样打印一些非 MVT 特征
print(f"\n其他关键特征:")
for feat in ["dep_queue_n_airport", "dep_queue_n_runway", "dep_mvt_cnt_airport_15m"]:
    if feat in df_dep.columns:
        vals = pd.to_numeric(df_dep[feat], errors="coerce").values
        valid = ~np.isnan(vals)
        print(f"  {feat:30s} n={np.sum(valid):>6d}, p50={np.nanmedian(vals):.1f}, max={np.nanmax(vals):.1f}")

print(f"\n✅ 检查完成")
