"""
在 holdout 上对每个嫌疑特征做单特征替换测试:
把那个特征替换为 NaN (或随机噪声), 看 RMSE 变化幅度 → 判定它的"贡献"
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
print(f"基准 RMSE = {base:.2f}\n")

# 嫌疑特征
suspects = ["mvt_pos_rw", "mvt_na_heavy_x_pos", "mvt_na_J", "mvt_na_L",
            "mvt_n_ahead", "mvt_q_peak", "mvt_q_mean",
            "mvt_mu_rwy_5", "mvt_mu_rwy_15", "mvt_mu_rwy_30",
            "mvt_w_actual_5", "mvt_w_actual_15", "mvt_w_actual_30",
            "dep_queue_n_airport", "dep_queue_n_runway",
            "t_unimp_r7",
            "ref_was_estimated",
            "is_domestic_estimate"]

# 测试 1: 单特征改 NaN
print("=" * 70)
print("[测试 1] 单特征改 NaN — 看剔除特征后的 RMSE 变化")
print("=" * 70)
print(f"{'特征':30s} {'base':>8s} {'zero':>8s} {'Δ':>8s}")
print("-" * 70)
results = []
for f in suspects:
    X_mod = X_enc.copy()
    X_mod[:, feat_idx[f]] = -1  # xgb 的 missing value, 强制走 missing 分支
    r = rmse(X_mod)
    print(f"{f:30s} {base:>8.2f} {r:>8.2f} {r-base:>+8.2f}")
    results.append((f, r-base))

print()
print(f"{'特征':30s} {'abs(Δ)':>10s} {'排序'}")
print("-" * 50)
results.sort(key=lambda x: abs(x[1]), reverse=True)
for i, (f, d) in enumerate(results):
    print(f"{f:30s} {abs(d):>10.2f} {i+1}")