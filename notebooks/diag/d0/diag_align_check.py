"""
排查 2: 行对齐 + winter/summer 模型臂

1. 验证预测与真值按 MVT_ID merge (而非按位置), 抽查无错位
2. 确认 R7slim 模型是单模型, 不分 winter/summer
3. 看 R7slim v11 与 R7slim v9 等其他模型, 检查是否应该有月份臂
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

# 复用 holdout_r7_rebuild 全流程
import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)

bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names

tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
from src.train_xgb import _encode_categoricals
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

dep_mask = (hr7.df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = hr7.df_ho_full[dep_mask].copy().reset_index(drop=True)

# 验证 1: MVT_ID 唯一性
print("=" * 70)
print("[验证 1] MVT_ID 唯一性 (行对齐基础)")
print("=" * 70)
ids = df_ho_dep_full["MVT_ID_mvt"].values
print(f"  Holdout DEP 总数: {len(ids)}")
print(f"  MVT_ID 唯一: {len(np.unique(ids))}")
print(f"  重复 MVT_ID: {len(ids) - len(np.unique(ids))}")

# 预测
X = df_ho_dep_full[FEATS69].copy()
X_enc = _encode_categoricals(X, fit=False).values.astype(np.float32)
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
months = pd.to_datetime(df_ho_dep_full["MVT_TIME_UTC_mvt"]).dt.month.values

# 验证 2: 抽查无错位 (按位置 vs 按 MVT_ID)
print()
print("=" * 70)
print("[验证 2] 抽查: 位置 vs MVT_ID 对齐一致性")
print("=" * 70)
print(f"  位置 0: MVT_ID={ids[0]}, ADEP={df_ho_dep_full['ADEP_mvt'].iloc[0]}, y={y[0]:.0f}, pred={pred[0]:.0f}")
print(f"  位置 1000: MVT_ID={ids[1000]}, ADEP={df_ho_dep_full['ADEP_mvt'].iloc[1000]}, y={y[1000]:.0f}, pred={pred[1000]:.0f}")
print(f"  位置 100000: MVT_ID={ids[100000]}, ADEP={df_ho_dep_full['ADEP_mvt'].iloc[100000]}, y={y[100000]:.0f}, pred={pred[100000]:.0f}")

# 按 MVT_ID 排序, 看 pred 是否仍能跟 y 对齐
sort_idx = np.argsort(ids)
ids_sorted = ids[sort_idx]
y_sorted = y[sort_idx]
pred_sorted = pred[sort_idx]
print(f"  排序后位置 0: MVT_ID={ids_sorted[0]}, y={y_sorted[0]:.0f}, pred={pred_sorted[0]:.0f}")
# RMSE 必须一致
rmse_pos = float(np.sqrt(np.mean((pred - y) ** 2)))
rmse_sort = float(np.sqrt(np.mean((pred_sorted - y_sorted) ** 2)))
print(f"  位置顺序 RMSE: {rmse_pos:.2f}, 排序后 RMSE: {rmse_sort:.2f} (应一致)")

# 验证 3: 加载 v15 集成, 看是否分 winter/summer
print()
print("=" * 70)
print("[验证 3] winter/summer 模型臂检查")
print("=" * 70)
print(f"  R7slim 模型数量 (含 v9, v11):")
import os
for f in os.listdir(ROOT / "models"):
    if "R7slim" in f.lower() or "R7slim" in f:
        print(f"    {f}")

# 看 R7slim v11 是否有月份特征
print()
print(f"  R7slim 模型 (xgb_v10_R7slim_12m_seed0_final.json):")
print(f"    feature 数量: {len(FEATS69)}")
print(f"    是否包含月份特征: {'sched_month' in FEATS69}, 'mvt_n_ahead' in FEATS69")
# 看树结构里是否按月份分裂
trees = bst.trees_to_dataframe()
m_split = trees[trees["Feature"] == "sched_month"]["Split"].values
print(f"    sched_month 分裂点: {sorted(set(m_split))[:10]}")

# 看 holdout 1月 vs 7月的模型表现
print()
print("=" * 70)
print("[验证 3.1] 1月 vs 7月 模型表现对比")
print("=" * 70)
valid = (y > 0) & (y < 10800) & np.isfinite(y)
for m in [1, 7]:
    m_mask = (months == m) & valid
    n = int(m_mask.sum())
    if n == 0:
        continue
    rmse_m = float(np.sqrt(np.mean((pred[m_mask] - y[m_mask]) ** 2)))
    mae_m = float(np.mean(np.abs(pred[m_mask] - y[m_mask])))
    bias_m = float(np.mean(pred[m_mask] - y[m_mask]))
    print(f"  {m}月: n={n:,}, RMSE={rmse_m:.2f}, MAE={mae_m:.2f}, Bias={bias_m:+.2f}")

# 1+7月合并: 加权 RMSE = sqrt(sum(sse)/sum(n))
m1 = (months == 1) & valid
m7 = (months == 7) & valid
sse_1 = float(((pred[m1] - y[m1]) ** 2).sum())
sse_7 = float(((pred[m7] - y[m7]) ** 2).sum())
print(f"\n  加权 RMSE = sqrt((SSE_1+SSE_7)/(n_1+n_7)) = "
      f"{np.sqrt((sse_1+sse_7)/(m1.sum()+m7.sum())):.2f}")

# 看看是否存在"winter/summer 子模型"预期
# 检查 1月 vs 7月机场分布
print()
print(f"\n  1月 vs 7月 机场分布差异 (前 5 机场):")
for ap in ["EDDF", "LFPG", "EGLL", "LEMD", "LIRF"]:
    n1 = ((months == 1) & (df_ho_dep_full["ADEP_mvt"].astype(str).values == ap)).sum()
    n7 = ((months == 7) & (df_ho_dep_full["ADEP_mvt"].astype(str).values == ap)).sum()
    if n1 > 100 or n7 > 100:
        print(f"    {ap}: 1月={n1:,}, 7月={n7:,}")