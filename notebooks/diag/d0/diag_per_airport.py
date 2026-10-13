"""
排查 1: 按 10 机场拆分 holdout RMSE, 看 +49 是集中还是均匀
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

# 复用 holdout_r7_rebuild 的全流程
import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)

# 模型 + 推理
bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names

import json
tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
from src.train_xgb import _encode_categoricals
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

dep_mask = (hr7.df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = hr7.df_ho_full[dep_mask].copy().reset_index(drop=True)
X = df_ho_dep_full[FEATS69].copy()
X_enc = _encode_categoricals(X, fit=False).values.astype(np.float32)
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)

y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
valid = (y > 0) & (y < 10800) & np.isfinite(y)
airports = df_ho_dep_full["ADEP_mvt"].astype(str).values
months = pd.to_datetime(df_ho_dep_full["MVT_TIME_UTC_mvt"]).dt.month.values

# ============================================================
# 总体 RMSE
# ============================================================
rmse_all = float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2)))
print(f"总体 RMSE = {rmse_all:.2f} (锚点 283.55, Δ=+{rmse_all-283.55:.2f})")

# ============================================================
# 分机场 RMSE
# ============================================================
print()
print("=" * 70)
print("[按 10 机场分解 RMSE]")
print("=" * 70)
print(f"{'#':>3s} {'ADEP':6s} {'n':>8s} {'RMSE':>8s} {'Δ vs 283':>10s} {'占比%':>8s} {'Bias':>8s}")
print("-" * 65)

# 总误差平方
total_sse = float(((pred[valid] - y[valid]) ** 2).sum())
results = []
for ap in sorted(np.unique(airports)):
    mask = (airports == ap) & valid
    n = int(mask.sum())
    if n < 100:
        continue
    rmse_ap = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))
    sse_ap = float(((pred[mask] - y[mask]) ** 2).sum())
    pct = 100 * sse_ap / total_sse
    bias = float(np.mean(pred[mask] - y[mask]))
    results.append((ap, n, rmse_ap, pct, bias))

# 按 SSE 占比排序
results.sort(key=lambda x: -x[3])
for i, (ap, n, rmse, pct, bias) in enumerate(results):
    delta = rmse - 283.55
    flag = " ⚠️" if rmse > 320 else ""
    print(f"{i+1:3d} {ap:6s} {n:>8d} {rmse:>8.2f} {delta:>+10.2f} {pct:>7.1f}% {bias:>+8.2f}{flag}")

print()
print(f"前 3 机场累计 SSE 占比: {sum(r[3] for r in results[:3]):.1f}%")
print(f"后 5 机场累计 SSE 占比: {sum(r[3] for r in results[-5:]):.1f}%")

# ============================================================
# 分月份 RMSE
# ============================================================
print()
print("=" * 70)
print("[按月份分解 RMSE]")
print("=" * 70)
print(f"{'月份':>6s} {'n':>10s} {'RMSE':>8s} {'Δ vs 283':>10s} {'Bias':>8s}")
print("-" * 50)
for m in [1, 7]:
    mask = (months == m) & valid
    n = int(mask.sum())
    rmse_m = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))
    bias = float(np.mean(pred[mask] - y[mask]))
    print(f"  {m:>4d} {n:>10d} {rmse_m:>8.2f} {rmse_m-283.55:>+10.2f} {bias:>+8.2f}")

# 按机场×月份交叉
print()
print("[按机场×月份交叉 RMSE]")
print(f"{'ADEP':6s} {'1月 RMSE':>10s} {'7月 RMSE':>10s} {'差异':>8s}")
print("-" * 40)
for ap, n, _, _, _ in results:
    rmse_1 = float(np.sqrt(np.mean((pred[(airports==ap) & (months==1) & valid] - y[(airports==ap) & (months==1) & valid]) ** 2))) if ((airports==ap) & (months==1) & valid).sum() > 100 else np.nan
    rmse_7 = float(np.sqrt(np.mean((pred[(airports==ap) & (months==7) & valid] - y[(airports==ap) & (months==7) & valid]) ** 2))) if ((airports==ap) & (months==7) & valid).sum() > 100 else np.nan
    print(f"{ap:6s} {rmse_1:>10.2f} {rmse_7:>10.2f} {rmse_1-rmse_7:>+8.2f}")