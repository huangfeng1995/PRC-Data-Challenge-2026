"""
验收点1：标准 holdout (train 2025-02~12 → val 2025-01+07)
用留存的 R7slim 模型 + 完整重建特征管线, 计算 RMSE,
与 README 锚点 283.55 (R7slim 单模型) 对照

策略: 用 src.features.build_features (旧管线) 做基础特征,
       用 _r7_lib (新) 做计数/MVT/D 特征, 确保 69 特征对齐
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

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic, BASELINE_FEATURE_COLS
from src.features_queue import add_queue_features
from src.train_xgb import _encode_categoricals
import _r7_lib as L
import _r8_drift_lib as D

# ============================================================
# 1. 加载 holdout (2025-01+07)
# ============================================================
print("=" * 70)
print("[1] 加载 holdout (2025-01+07)")
print("=" * 70)
TRAIN_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"]

ho_dfs = []
for m in [1, 7]:
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    df = load_month(f"data/raw/{fname}", columns=TRAIN_COLS)
    ho_dfs.append(df)
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
df_ho_dep = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_ho_dep = df_ho_dep[
    (df_ho_dep["TAXITIME_SEC_mvt"] > 0) &
    (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)
].reset_index(drop=True)
print(f"  Holdout DEP: {len(df_ho_dep):,}")

# ============================================================
# 2. 加载训练 (2025-02~12)
# ============================================================
print()
print("=" * 70)
print("[2] 加载训练 (2025-02~12)")
print("=" * 70)
months_to_load = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months_to_load:
    fname = (f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
             if m < 12 else "training_2025-12-01_2026-01-01.parquet")
    df = load_month(f"data/raw/{fname}", columns=TRAIN_COLS)
    train_dfs.append(df)
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[df_tr["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
df_tr_dep = df_tr_dep[
    (df_tr_dep["TAXITIME_SEC_mvt"] > 0) &
    (df_tr_dep["TAXITIME_SEC_mvt"] < 10800)
].reset_index(drop=True)
print(f"  训练 DEP (有 TAXITIME): {len(df_tr_dep):,}")

# ============================================================
# 3. 构建 LUT (训练数据)
# ============================================================
print()
print("=" * 70)
print("[3] 构建 t_unimp LUT (用训练数据)")
print("=" * 70)
lut = L.build_t_unimp_lut(df_tr_dep)
print(f"  LUT: L1={len(lut['L1'])}, L2={len(lut['L2'])}, L3={len(lut['L3'])}")

# ============================================================
# 4. 重建 holdout 全套特征 (基础 + queue + 计数 + MVT + D)
# ============================================================
print()
print("=" * 70)
print("[4] 重建 holdout 全套特征")
print("=" * 70)

# 4.1 基础特征 (BASELINE_FEATURE_COLS)
df_ho_basic = build_basic(df_ho.copy())
print(f"  基础特征: {df_ho_basic.shape}")

# 4.2 queue 特征 (机场 + 跑道 + heavy + ARR)
df_ho_full = add_queue_features(
    df_ho_basic.copy(),
    include_a=True, include_b=True, include_c=True,
    include_d=True, include_e=True,
)
print(f"  +queue (A+B+C+D+E): {df_ho_full.shape}")

# 4.3 ref_was_estimated (1 if AOBT 空, else 0)
df_ho_full["ref_was_estimated"] = df_ho_full["AOBT_3_flt"].isna().astype("int8")

# 4.4 D 特征 (R8 drift, 用训练数据 ref)
d_dict = D.compute_drift_features(df_ho_full, ref_dep=df_tr, label="holdout")
for k, v in d_dict.items():
    df_ho_full[k] = v

# 4.5 t_unimp (查 LUT)
t_vals, t_src = L.lookup_t_unimp(df_ho_full, lut)
df_ho_full["t_unimp_r7"] = t_vals
print(f"  t_unimp 有效: {np.sum(~np.isnan(t_vals))}/{len(t_vals)}")

# 4.6 计数 9 个 + dly 4 个 (DEP 子集)
dep_mask_full = (df_ho_full["PHASE_mvt"] == "DEP").values
extra_dict = L.compute_extra_features(df_ho_full[dep_mask_full], label="holdout")
for k, v in extra_dict.items():
    df_ho_full[k] = np.nan
    df_ho_full.loc[dep_mask_full, k] = v

# 4.7 MVT 锚定 16 个 (DEP 子集)
mvt_dict = L.compute_mvt_anchor_features(
    df_ho_full[dep_mask_full],
    t_unimp_vals=t_vals[dep_mask_full], label="holdout")
for k, v in mvt_dict.items():
    df_ho_full[k] = np.nan
    df_ho_full.loc[dep_mask_full, k] = v

# ============================================================
# 5. 用留存模型推理
# ============================================================
print()
print("=" * 70)
print("[5] 用留存 R7slim 模型推理")
print("=" * 70)

bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names
print(f"  R7slim 模型: 69 特征")

# 取 holdout DEP
df_ho_dep_full = df_ho_full[dep_mask_full].copy().reset_index(drop=True)

# 检查所有 69 特征是否存在
missing = [f for f in FEATS69 if f not in df_ho_dep_full.columns]
if missing:
    print(f"  ⚠️ 缺失特征: {missing}")
    for f in missing:
        df_ho_dep_full[f] = np.nan

# 加载训练时用的分类编码 (R7slim 训练用的是 _rebuild_12m_cat.json)
tx_cat = json.loads(
    (ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

# 编码 + 推理
X = df_ho_dep_full[FEATS69].copy()
X_enc = _encode_categoricals(X, fit=False)
print(f"  X shape: {X_enc.shape}, dtype: {X_enc.dtypes.unique()}")

dmat = xgb.DMatrix(X_enc.values.astype(np.float32), feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
y = df_ho_dep_full["TAXITIME_SEC_mvt"].values

rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
mae = float(np.mean(np.abs(pred - y)))
bias = float(np.mean(pred - y))

# 排除 TAXITIME=0/NaN/10800+ 的行
valid_mask = (y > 0) & (y < 10800) & np.isfinite(y)
rmse_v = float(np.sqrt(np.mean((pred[valid_mask] - y[valid_mask]) ** 2)))

print()
print("=" * 70)
print(f"[验证结果]")
print("=" * 70)
print(f"  Holdout DEP: {len(y):,}")
print(f"  RMSE (全) = {rmse:.2f}")
print(f"  RMSE (有效) = {rmse_v:.2f} (README 锚点 R7slim=283.55)")
print(f"  MAE  = {mae:.2f}")
print(f"  Bias = {bias:+.2f}")
print(f"  Y mean = {np.mean(y):.2f}, Pred mean = {np.mean(pred):.2f}")
print(f"  Δ vs 锚点 = {rmse_v - 283.55:+.2f}")

if abs(rmse_v - 283.55) < 10:
    print(f"  ✅ 口径一致 (误差 < 10)")
else:
    print(f"  ⚠️ 口径有偏差, 需要逐特征诊断")

out = {
    "holdout": "2025-01+07",
    "n_dep": int(len(y)),
    "rmse_total": rmse,
    "rmse_valid": rmse_v,
    "mae": mae,
    "bias": bias,
    "anchor_r7slim": 283.55,
    "delta_vs_anchor": rmse_v - 283.55,
    "y_mean": float(np.mean(y)),
    "pred_mean": float(np.mean(pred)),
    "missing_feats": missing,
}
out_path = ROOT / "notebooks/diag/d0/holdout_r7_rebuild_result.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  结果已保存: {out_path}")