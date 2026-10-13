"""
验证: 用训练数据构建 LUT, 然后计算 t_unimp, 再看 MVT 特征和 RMSE
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

# 加载模型
bst_r7 = xgb.Booster()
bst_r7.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst_r7.feature_names
feat_idx = {f: i for i, f in enumerate(FEATS69)}

# 加载分类编码映射
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = json.loads(
    (ROOT / "models/feature_lists/xgb_v9_Gd_ABD_runway_seed0_val_cat_mappings.json").read_text())

# ============================================================
# 1. 用训练数据构建 LUT (用 2-3 个月的数据就够了)
# ============================================================
print("="*60)
print("用训练数据构建 LUT...")

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
print(f"训练数据: {len(df_train)} 行")

# 只取 DEP 且有有效滑行时间的
dep_train = df_train[
    (df_train["PHASE_mvt"] == "DEP") &
    (df_train["TAXITIME_SEC_mvt"] > 0) &
    (df_train["TAXITIME_SEC_mvt"] < 10800)
].copy()
print(f"有效 DEP 训练样本: {len(dep_train)}")

lut = L.build_t_unimp_lut(dep_train)
print(f"LUT: L1={len(lut['L1'])}, L2={len(lut['L2'])}, L3={len(lut['L3'])}")

# 检查 LUT 有效值
l1_valid = lut["L1"].notna().sum()
l2_valid = lut["L2"].notna().sum()
l3_valid = lut["L3"].notna().sum()
print(f"LUT 有效值: L1={l1_valid}, L2={l2_valid}, L3={l3_valid}")

# ============================================================
# 2. 构建 ranking 特征
# ============================================================
print(f"\n{'='*60}")
print("构建 ranking 特征...")

df_rank = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
df_rank = ab.parse_time_columns(df_rank, cols=ab.TIME_COLS)
df_full = ab.build_full_features(df_rank.copy())
dep_mask = df_full["PHASE_mvt"] == "DEP"
df_dep = df_full[dep_mask].copy().reset_index(drop=True)

# 用训练好的 LUT 查 t_unimp
t_vals, t_src = L.lookup_t_unimp(df_dep, lut)
print(f"t_unimp 有效值: {np.sum(~np.isnan(t_vals))} / {len(t_vals)}")
print(f"t_unimp p50: {np.nanmedian(t_vals):.1f} 秒 = {np.nanmedian(t_vals)/60:.1f} 分钟")
print(f"t_unimp max: {np.nanmax(t_vals):.1f} 秒")

df_dep["t_unimp_r7"] = t_vals
extra_dict = L.compute_extra_features(df_dep, label="rank")
for k, v in extra_dict.items():
    df_dep[k] = v
mvt_dict = L.compute_mvt_anchor_features(df_dep, t_unimp_vals=t_vals, label="rank")
for k, v in mvt_dict.items():
    df_dep[k] = v

# ============================================================
# 3. 检查 MVT 特征值
# ============================================================
print(f"\n{'='*60}")
print("MVT 特征值统计:")

mvt_cols = sorted([c for c in df_dep.columns if c.startswith("mvt_")])
print(f"{'特征':30s} {'n_valid':>8s} {'min':>10s} {'p50':>10s} {'p90':>10s} {'max':>12s}")
print("-" * 85)
for feat in mvt_cols:
    vals = pd.to_numeric(df_dep[feat], errors="coerce").values
    valid = ~np.isnan(vals)
    n_valid = np.sum(valid)
    if n_valid == 0:
        print(f"{feat:30s} {n_valid:>8d} {'NaN':>10s}")
        continue
    print(f"{feat:30s} {n_valid:>8d} {np.nanmin(vals):>10.1f} "
          f"{np.nanmedian(vals):>10.1f} {np.nanpercentile(vals, 90):>10.1f} "
          f"{np.nanmax(vals):>12.1f}")

# ============================================================
# 4. RMSE 验证
# ============================================================
print(f"\n{'='*60}")
print("RMSE 验证:")

X = df_dep[FEATS69].copy()
X_enc = tx._encode_categoricals(X.copy(), fit=False)

sub_r7 = pd.read_parquet(ROOT / "submissions" / "jubilant-puzzle_r7_v1.parquet")
dep_ids = df_dep["MVT_ID_mvt"].values
sub_map = dict(zip(sub_r7["MVT_ID_mvt"].values, sub_r7["TAXITIME_SEC_mvt"].values))
pred_sub = np.array([sub_map.get(mid, np.nan) for mid in dep_ids])
valid = ~np.isnan(pred_sub)

dmat = xgb.DMatrix(X_enc.values.astype(np.float32), feature_names=FEATS69)
pred = bst_r7.predict(dmat)
pred = np.clip(pred, 30, 10800)
diff = pred[valid] - pred_sub[valid]
rmse = np.sqrt(np.mean(diff**2))
mean_diff = np.mean(diff)
print(f"RMSE = {rmse:.1f}")
print(f"Mean diff = {mean_diff:.1f} (正=我们预测偏大)")

print(f"\n✅ 验证完成")
