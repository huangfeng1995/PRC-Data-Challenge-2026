"""
排查 4 (优化版): 列顺序 + 抽样 SHAP 分机场定位元凶

SHAP 计算很慢, 改用抽样 5000/机场 (总 ~25000) 足够看趋势
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

# 复用 hr7 完整管线
import importlib.util
spec = importlib.util.spec_from_file_location("hr7", ROOT / "notebooks/diag/d0/holdout_r7_rebuild.py")
hr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr7)

bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names

tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

dep_mask = (hr7.df_ho_full["PHASE_mvt"] == "DEP").values
df_ho_dep_full = hr7.df_ho_full[dep_mask].copy().reset_index(drop=True)

# ============================================================
# 1. 列顺序对齐核查
# ============================================================
print("=" * 70)
print("[1] 列顺序对齐核查")
print("=" * 70)

X = df_ho_dep_full[FEATS69].copy()
print(f"  X 列序 == booster.feature_names: {list(X.columns) == list(FEATS69)}")

# 严格用 booster.feature_names 重排
X_reordered = df_ho_dep_full[FEATS69].copy()
print(f"  重新按 feature_names 取列后, 列序: {list(X_reordered.columns) == list(FEATS69)}")

# 检查每一列的源 dtype
print(f"\n  前 15 列源 dtype:")
for f in FEATS69[:15]:
    print(f"    {f}: {df_ho_dep_full[f].dtype}")

X_enc = _encode_categoricals(X_reordered, fit=False).values.astype(np.float32)
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
valid = (y > 0) & (y < 10800) & np.isfinite(y)
print(f"\n  基准 RMSE = {float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2))):.2f}")

# ============================================================
# 2. 抽样 SHAP 分机场对比
# ============================================================
print()
print("=" * 70)
print("[2] 抽样 SHAP 分机场对比 (5000/机场)")
print("=" * 70)

airports = df_ho_dep_full["ADEP_mvt"].astype(str).values
np.random.seed(42)

compare_aps = ["EGLL", "LIRF", "LTFM", "LFPG", "EDDF"]

# 对每个机场随机抽样 5000 个
indices_per_ap = {}
for ap in compare_aps:
    ap_idx = np.where((airports == ap) & valid)[0]
    n_sample = min(5000, len(ap_idx))
    indices_per_ap[ap] = np.random.choice(ap_idx, n_sample, replace=False)
    print(f"  {ap}: 全 {len(ap_idx)}, 抽 {n_sample}")

# 合并所有抽样索引
all_idx = np.concatenate([indices_per_ap[ap] for ap in compare_aps if ap in indices_per_ap])
print(f"  合并: {len(all_idx)}")

# 重新对子集计算 SHAP
X_sub = df_ho_dep_full.iloc[all_idx][FEATS69].copy()
X_sub_enc = _encode_categoricals(X_sub, fit=False).values.astype(np.float32)
dmat_sub = xgb.DMatrix(X_sub_enc, feature_names=FEATS69)
print(f"  计算 SHAP (子集 {len(all_idx)})...")
shap_sub = bst.predict(dmat_sub, pred_contribs=True)
print(f"  SHAP shape: {shap_sub.shape}")
shap_feats_sub = shap_sub[:, :-1]  # 去 BIAS 列

# 按机场汇总
sub_airports = df_ho_dep_full.iloc[all_idx]["ADEP_mvt"].astype(str).values
sub_y = y[all_idx]
sub_pred = pred[all_idx]

print(f"\n{'ADEP':6s} {'Bias':>8s} {'n':>6s} {'真均值':>8s} {'预测均值':>10s} {'RMSE':>8s}")
for ap in compare_aps:
    mask = sub_airports == ap
    if mask.sum() == 0:
        continue
    bias = float(np.mean(sub_pred[mask] - sub_y[mask]))
    rmse = float(np.sqrt(np.mean((sub_pred[mask] - sub_y[mask]) ** 2)))
    print(f"  {ap:6s} {bias:>+8.2f} {mask.sum():>6d} {sub_y[mask].mean():>8.1f} "
          f"{sub_pred[mask].mean():>10.1f} {rmse:>8.2f}")

# 算每机场 SHAP 均值
ap_shap = {}
for ap in compare_aps:
    mask = sub_airports == ap
    if mask.sum() == 0:
        continue
    ap_shap[ap] = shap_feats_sub[mask].mean(axis=0)

# ============================================================
# 2.1 EGLL 拉低元凶
# ============================================================
print()
print("=" * 70)
print("[2.1] EGLL 拉低元凶 (对比 EDDF, SHAP 负贡献 top 10)")
print("=" * 70)
egll_shap = ap_shap["EGLL"]
eddf_shap = ap_shap["EDDF"]
diff = egll_shap - eddf_shap
order = np.argsort(diff)
print(f"\n{'#':>3s} {'特征':28s} {'EGLL SHAP':>12s} {'EDDF SHAP':>12s} {'差':>12s}")
print("-" * 75)
for i in order[:10]:
    print(f"{i+1:3d} {FEATS69[i]:28s} {egll_shap[i]:>+12.2f} {eddf_shap[i]:>+12.2f} {diff[i]:>+12.2f}")

# ============================================================
# 2.2 LTFM 高估元凶
# ============================================================
print()
print("=" * 70)
print("[2.2] LTFM 高估元凶 (对比 EDDF, SHAP 正贡献 top 10)")
print("=" * 70)
ltfm_shap = ap_shap["LTFM"]
diff = ltfm_shap - eddf_shap
order = np.argsort(-diff)
print(f"\n{'#':>3s} {'特征':28s} {'LTFM SHAP':>12s} {'EDDF SHAP':>12s} {'差':>12s}")
print("-" * 75)
for i in order[:10]:
    print(f"{i+1:3d} {FEATS69[i]:28s} {ltfm_shap[i]:>+12.2f} {eddf_shap[i]:>+12.2f} {diff[i]:>+12.2f}")

# ============================================================
# 2.3 LFPG 高估元凶
# ============================================================
print()
print("=" * 70)
print("[2.3] LFPG 高估元凶")
print("=" * 70)
lfpg_shap = ap_shap["LFPG"]
diff = lfppg_shap - eddf_shap if False else (lfpg_shap - eddf_shap)
order = np.argsort(-diff)
print(f"\n{'#':>3s} {'特征':28s} {'LFPG SHAP':>12s} {'EDDF SHAP':>12s} {'差':>12s}")
print("-" * 75)
for i in order[:10]:
    print(f"{i+1:3d} {FEATS69[i]:28s} {lfpg_shap[i]:>+12.2f} {eddf_shap[i]:>+12.2f} {diff[i]:>+12.2f}")

# ============================================================
# 3. 类别编码核查 (元凶特征)
# ============================================================
print()
print("=" * 70)
print("[3] 类别编码核查: EGLL 元凶特征 unknown 比例")
print("=" * 70)

# 取 EGLL 拉低 top 5
egll_diff = egll_shap - eddf_shap
order_egll = np.argsort(egll_diff)
egll_top5 = [FEATS69[i] for i in order_egll[:5]]
print(f"  EGLL 拉低 top 5: {egll_top5}")

print(f"\n{'特征':28s} {'dtype':>10s} {'EGLL unknown%':>15s} {'EGLL sample':>14s}")
print("-" * 75)
egll_df = df_ho_dep_full[airports == "EGLL"]

for f in egll_top5:
    s = egll_df[f]
    if s.dtype.name == "category":
        s_str = s.astype(str)
        mapping = tx_cat.get(f, {})
        known_set = set(mapping.keys()) - {"__nan__"}
        cnt_unknown = (~s_str.isin(known_set)).sum()
        pct = 100 * cnt_unknown / len(s_str)
        flag = " ⚠️" if pct > 5 else ""
        print(f"{f:28s} {'category':>10s} {pct:>14.1f}% {len(s_str):>14d}{flag}")
    else:
        v = pd.to_numeric(s, errors="coerce").values
        valid_v = ~np.isnan(v)
        print(f"{f:28s} {s.dtype.name:>10s} {'N/A':>15s} {valid_v.sum():>14d}")

# 看 EGLL 在 top5 类别特征的 top values
print()
print("=" * 70)
print("[3.1] EGLL 元凶特征 top 5 值分布")
print("=" * 70)
for f in egll_top5:
    s = egll_df[f]
    if s.dtype.name == "category":
        s_str = s.astype(str)
        mapping = tx_cat.get(f, {})
        known_set = set(mapping.keys()) - {"__nan__"}
        is_unknown = ~s_str.isin(known_set)
        print(f"\n  {f} (类别):")
        print(f"    EGLL total: {len(s_str)}, unknown: {is_unknown.sum()}")
        cnt = s_str.value_counts().head(5)
        for val, n in cnt.items():
            in_map = "✅" if val in known_set else "❌"
            print(f"    {in_map} '{val}': {n}")
    else:
        v = pd.to_numeric(s, errors="coerce").values
        v = v[~np.isnan(v)]
        if len(v) > 0:
            print(f"\n  {f} (数值):")
            print(f"    p10={np.percentile(v,10):.2f}, p50={np.median(v):.2f}, p90={np.percentile(v,90):.2f}, max={np.max(v):.2f}")

# 同时看 LTFM 元凶的 unknown
print()
print("=" * 70)
print("[3.2] LTFM 元凶特征 unknown 比例 (top 5)")
print("=" * 70)
ltfm_diff = ltfm_shap - eddf_shap
order_ltfm = np.argsort(-ltfm_diff)
ltfm_top5 = [FEATS69[i] for i in order_ltfm[:5]]
print(f"  LTFM 高估 top 5: {ltfm_top5}")

ltfm_df = df_ho_dep_full[airports == "LTFM"]
for f in ltfm_top5:
    s = ltfm_df[f]
    if s.dtype.name == "category":
        s_str = s.astype(str)
        mapping = tx_cat.get(f, {})
        known_set = set(mapping.keys()) - {"__nan__"}
        is_unknown = ~s_str.isin(known_set)
        pct = 100 * is_unknown.sum() / len(s_str)
        flag = " ⚠️" if pct > 5 else ""
        print(f"{f:28s} unknown: {is_unknown.sum():>5d} ({pct:>4.1f}%){flag}")
    else:
        v = pd.to_numeric(s, errors="coerce").values
        v = v[~np.isnan(v)]
        if len(v) > 0:
            print(f"{f:28s} 数值: p50={np.median(v):.2f}, max={np.max(v):.2f}")

# 保存 SHAP 抽样结果
import pickle
with open('/tmp/holdout_shap_sample.pkl', 'wb') as f:
    pickle.dump({
        'shap_feats_sub': shap_feats_sub,
        'sub_airports': sub_airports,
        'sub_y': sub_y,
        'sub_pred': sub_pred,
        'FEATS69': FEATS69,
        'cat_mappings': tx_cat,
        'ap_shap': ap_shap,
        'egll_top5': egll_top5,
        'ltfm_top5': ltfm_top5,
    }, f)
print(f"\n  SHAP 抽样数据: /tmp/holdout_shap_sample.pkl")