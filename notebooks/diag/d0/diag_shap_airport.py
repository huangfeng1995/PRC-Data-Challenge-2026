"""
排查 4: 列顺序对齐核查 + SHAP 分机场定位元凶

1. 检查 X 列顺序是否严格等于 booster.feature_names
2. SHAP 分机场对比: EGLL vs EDDF (正常), LTFM (高估)
   找出把 EGLL 拉低的元凶特征
3. 对元凶特征, 检查在 EGLL 的 unknown/未命中比例
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

# 重新取数据 (严格按 booster.feature_names 取列)
X = df_ho_dep_full[FEATS69].copy()
print(f"  df 列数: {len(df_ho_dep_full.columns)}")
print(f"  取 booster.feature_names 后的 X 列数: {len(X.columns)}")
print(f"  X 列序前 5: {list(X.columns[:5])}")
print(f"  booster.feature_names 前 5: {list(FEATS69[:5])}")
print(f"  X 列序 == booster.feature_names: {list(X.columns) == list(FEATS69)}")

# 检查每一列是否完全一致
mismatch = []
for i, f in enumerate(FEATS69):
    if f not in df_ho_dep_full.columns:
        mismatch.append((i, f, "MISSING"))
        continue
    actual_pos = list(df_ho_dep_full.columns).index(f)
    if actual_pos != i:
        mismatch.append((i, f, f"WRONG_POS_{actual_pos}"))

if mismatch:
    print(f"\n  ⚠️ 列序不一致: {len(mismatch)} 处")
    for i, f, msg in mismatch[:10]:
        print(f"    pos {i}: {f} ({msg})")
else:
    print(f"\n  ✅ 列序完全一致")

# 编码 + DMatrix
X_enc = _encode_categoricals(X, fit=False).values.astype(np.float32)
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
valid = (y > 0) & (y < 10800) & np.isfinite(y)
print(f"\n  基准 RMSE = {float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2))):.2f}")

# ============================================================
# 2. SHAP 分机场对比
# ============================================================
print()
print("=" * 70)
print("[2] SHAP 分机场定位元凶")
print("=" * 70)

# pred_contribs=True 拿 SHAP 值 (n, n_features+1, 最后一列是 base_score)
print("  计算 SHAP 值...")
shap = bst.predict(dmat, pred_contribs=True)
print(f"  SHAP shape: {shap.shape}")  # (n, 70) 最后一列是 base
# 去掉最后一列 (BIAS)
shap_feats = shap[:, :-1]
assert shap_feats.shape == (len(df_ho_dep_full), len(FEATS69))

# 按机场汇总 SHAP 均值
airports = df_ho_dep_full["ADEP_mvt"].astype(str).values

# 选 4 个机场: EGLL(低估), LIRF(高方差), LTFM/LFPG(高估), EDDF(正常)
compare_aps = ["EGLL", "LIRF", "LTFM", "LFPG", "EDDF"]
ap_shap_mean = {}
for ap in compare_aps:
    mask = (airports == ap) & valid
    if mask.sum() < 100:
        continue
    ap_shap_mean[ap] = shap_feats[mask].mean(axis=0)  # (69,)

# 算 bias per 机场
print()
print(f"{'ADEP':6s} {'Bias':>8s} {'n':>8s} {'真均值':>8s} {'预测均值':>10s}")
for ap in compare_aps:
    mask = (airports == ap) & valid
    if mask.sum() < 100:
        continue
    bias = float(np.mean(pred[mask] - y[mask]))
    print(f"  {ap:6s} {bias:>+8.2f} {mask.sum():>8d} {y[mask].mean():>8.1f} {pred[mask].mean():>10.1f}")

# 找到把 EGLL 拉低的元凶 (SHAP 负贡献最大)
print()
print("=" * 70)
print("[2.1] EGLL 拉低元凶 (SHAP 负贡献 top 10)")
print("=" * 70)
print(f"{'特征':28s} {'EGLL SHAP':>12s} {'EDDF SHAP':>12s} {'差 (EGLL-EDDF)':>18s}")
print("-" * 75)
egll_shap = ap_shap_mean["EGLL"]
eddf_shap = ap_shap_mean["EDDF"]
diff = egll_shap - eddf_shap
order = np.argsort(diff)  # 最负的在前
for i in order[:10]:
    print(f"{FEATS69[i]:28s} {egll_shap[i]:>+12.2f} {eddf_shap[i]:>+12.2f} {diff[i]:>+18.2f}")

# 同样: LTFM 高估元凶 (SHAP 正贡献最大)
print()
print("=" * 70)
print("[2.2] LTFM 高估元凶 (SHAP 正贡献 top 10, 对比 EDDF)")
print("=" * 70)
print(f"{'特征':28s} {'LTFM SHAP':>12s} {'EDDF SHAP':>12s} {'差 (LTFM-EDDF)':>18s}")
print("-" * 75)
ltfm_shap = ap_shap_mean["LTFM"]
diff = ltfm_shap - eddf_shap
order = np.argsort(-diff)  # 最正的在前
for i in order[:10]:
    print(f"{FEATS69[i]:28s} {ltfm_shap[i]:>+12.2f} {eddf_shap[i]:>+12.2f} {diff[i]:>+18.2f}")

# 同样: LFPG 高估元凶
print()
print("=" * 70)
print("[2.3] LFPG 高估元凶 (SHAP 正贡献 top 10, 对比 EDDF)")
print("=" * 70)
lfpg_shap = ap_shap_mean["LFPG"]
diff = lfppg_shap - eddf_shap if False else (lfpg_shap - eddf_shap)
order = np.argsort(-diff)
print(f"{'特征':28s} {'LFPG SHAP':>12s} {'EDDF SHAP':>12s} {'差 (LFPG-EDDF)':>18s}")
print("-" * 75)
for i in order[:10]:
    print(f"{FEATS69[i]:28s} {lfpg_shap[i]:>+12.2f} {eddf_shap[i]:>+12.2f} {diff[i]:>+18.2f}")

# ============================================================
# 3. 类别编码核查: 元凶特征在 EGLL 的 unknown 比例
# ============================================================
print()
print("=" * 70)
print("[3] 类别编码核查: EGLL 元凶特征 unknown 比例")
print("=" * 70)

# 类别列 (用 _encode_categoricals 后变成 int32)
cat_cols_in_model = [f for f in FEATS69
                     if df_ho_dep_full[f].dtype.name == "category"]

# EGLL 数据
egll_mask = airports == "EGLL"
egll_df = df_ho_dep_full[egll_mask].copy()

print(f"  类别列: {cat_cols_in_model}")
print(f"  EGLL 样本: {len(egll_df)}")
print(f"\n{'特征':28s} {'EGLL unknown%':>15s} {'EGLL 总unique':>15s} {'其他AP unique':>15s}")
print("-" * 80)

unknown_threshold_count = 0
for col in cat_cols_in_model:
    s = egll_df[col].astype(str)
    mapping = tx_cat.get(col, {})
    # EGLL 该列中, 不在 mapping 里的值数
    s_set = set(s.unique())
    known_set = set(mapping.keys()) - {"__nan__"}
    unknown_set = s_set - known_set
    unknown_pct = 100 * sum(s.isin(unknown_set)) / len(s)
    # 其他机场 (非 EGLL) 在该列的总 unique
    other_df = df_ho_dep_full[~egll_mask]
    other_unique = other_df[col].astype(str).nunique()
    if unknown_pct > 5:
        unknown_threshold_count += 1
    print(f"{col:28s} {unknown_pct:>14.1f}% {len(s_set):>15d} {other_unique:>15d}")

print(f"\n  ⚠️ EGLL unknown > 5% 的类别列数: {unknown_threshold_count}")

# 同样看 EGLL 元凶特征 (SHAP top 10 里的) 的实际值分布
print()
print("=" * 70)
print("[3.1] EGLL 元凶特征 (top 5 拉低) 实际值分布")
print("=" * 70)
egll_top_neg = []
for i in order[:5]:  # 上面 EGLL 拉低 top 5 (按 EDDF 排序)
    egll_top_neg.append(FEATS69[i])

# 重新拿 EGLL 拉低 top 5
diff_egll = egll_shap - eddf_shap
order_egll = np.argsort(diff_egll)
egll_top_neg = [FEATS69[i] for i in order_egll[:5]]

for f in egll_top_neg:
    s = egll_df[f]
    if s.dtype.name == "category":
        s_str = s.astype(str)
        mapping = tx_cat.get(f, {})
        known_set = set(mapping.keys()) - {"__nan__"}
        cnt_unknown = (~s_str.isin(known_set)).sum()
        print(f"\n  {f} (类别列):")
        print(f"    EGLL 样本: {len(s_str)}")
        print(f"    EGLL unknown 命中: {cnt_unknown} ({100*cnt_unknown/len(s_str):.1f}%)")
        print(f"    EGLL top 5 值: {list(s_str.value_counts().head(5).items())}")
        # 与 mapping 对比, 看 EGLL 用的 runway/stand 等是否在 mapping 里
        if f in ["RUNWAY_mvt", "STAND_mvt"]:
            print(f"    mapping 含 {f} 的 key 前 10: {list(mapping.keys())[:10]}")
    else:
        v = pd.to_numeric(s, errors="coerce").values
        valid_v = ~np.isnan(v)
        print(f"\n  {f} (数值列):")
        print(f"    EGLL: p50={np.nanmedian(v):.2f}, max={np.nanmax(v):.2f}, n_valid={valid_v.sum()}")

# 保存 SHAP 表给后续步骤
import pickle
with open('/tmp/holdout_shap_diag.pkl', 'wb') as f:
    pickle.dump({
        'shap_feats': shap_feats,
        'airports': airports,
        'FEATS69': FEATS69,
        'df_ho_dep_full': df_ho_dep_full,
        'bst': bst,
        'cat_mappings': tx_cat,
    }, f)
print(f"\n  SHAP 数据已保存: /tmp/holdout_shap_diag.pkl")