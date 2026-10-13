"""
排查 5: 训练集自洽 (in-sample) - 用训练月份 2025-03 算特征 + 喂留存模型

判定:
- 明显偏高 (>>283) → 管线本身错
- 很小 (<<283) → 管线对, 问题在 holdout 月份口径
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

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
from src.train_xgb import _encode_categoricals
import _r7_lib as L
import _r8_drift_lib as D

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


def build_all_features(df_raw):
    """完整特征重建 (与 holdout_r7_rebuild 一致)"""
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(
        df_basic.copy(),
        include_a=True, include_b=True, include_c=True,
        include_d=True, include_e=True,
    )
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    if "ADES_flt" in df_full.columns and "ADES_FILED_flt" in df_full.columns:
        df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    else:
        df_full["is_diverted"] = 0
    return df_full


# ============================================================
# 1. 加载 2025-03 作为 in-sample 验证集
# ============================================================
print("=" * 70)
print("[1] 加载 2025-03 (训练月份, in-sample)")
print("=" * 70)
df_mar = load_month("data/raw/training_2025-03-01_2025-04-01.parquet", columns=TRAIN_COLS)
df_mar_dep = df_mar[
    (df_mar["PHASE_mvt"] == "DEP") &
    (df_mar["TAXITIME_SEC_mvt"] > 0) &
    (df_mar["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  2025-03 DEP (有 TAXITIME): {len(df_mar_dep):,}")

# ============================================================
# 2. 构建 LUT (用 2025-02 训练, 避免数据泄漏)
# ============================================================
print()
print("=" * 70)
print("[2] 构建 LUT (用 2025-02, 避免 03 数据泄漏)")
print("=" * 70)
df_feb = load_month("data/raw/training_2025-02-01_2025-03-01.parquet", columns=TRAIN_COLS)
df_feb_dep = df_feb[
    (df_feb["PHASE_mvt"] == "DEP") &
    (df_feb["TAXITIME_SEC_mvt"] > 0) &
    (df_feb["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  2025-02 DEP (有 TAXITIME): {len(df_feb_dep):,}")

lut = L.build_t_unimp_lut(df_feb_dep)
print(f"  LUT: L1={len(lut['L1'])}, L2={len(lut['L2'])}, L3={len(lut['L3'])}")

# ============================================================
# 3. 重建 2025-03 全套特征
# ============================================================
print()
print("=" * 70)
print("[3] 重建 2025-03 全套特征")
print("=" * 70)
df_mar_full = build_all_features(df_mar)
print(f"  基础+queue: {df_mar_full.shape}")

# D 特征 (用 02 月作 ref)
d_dict = D.compute_drift_features(df_mar_full, ref_dep=df_feb, label="march")
for k, v in d_dict.items():
    df_mar_full[k] = v

# t_unimp
t_vals, t_src = L.lookup_t_unimp(df_mar_full, lut)
df_mar_full["t_unimp_r7"] = t_vals
print(f"  t_unimp 有效: {np.sum(~np.isnan(t_vals))}/{len(t_vals)}")

# 计数 + dly + MVT
dep_mask_full = (df_mar_full["PHASE_mvt"] == "DEP").values
extra_dict = L.compute_extra_features(df_mar_full[dep_mask_full], label="march")
for k, v in extra_dict.items():
    df_mar_full[k] = np.nan
    df_mar_full.loc[dep_mask_full, k] = v

mvt_dict = L.compute_mvt_anchor_features(
    df_mar_full[dep_mask_full],
    t_unimp_vals=t_vals[dep_mask_full], label="march")
for k, v in mvt_dict.items():
    df_mar_full[k] = np.nan
    df_mar_full.loc[dep_mask_full, k] = v

# ============================================================
# 4. 推理 + 评分
# ============================================================
print()
print("=" * 70)
print("[4] in-sample RMSE (2025-03 用 02 月 LUT)")
print("=" * 70)

bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names

tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

df_mar_dep_full = df_mar_full[dep_mask_full].copy().reset_index(drop=True)
missing = [f for f in FEATS69 if f not in df_mar_dep_full.columns]
if missing:
    print(f"  ⚠️ 缺失: {missing}")
    for f in missing:
        df_mar_dep_full[f] = np.nan

X = df_mar_dep_full[FEATS69].copy()
X_enc = _encode_categoricals(X, fit=False).values.astype(np.float32)
print(f"  X shape: {X_enc.shape}")

dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)
y = df_mar_dep_full["TAXITIME_SEC_mvt"].values
valid = (y > 0) & (y < 10800) & np.isfinite(y)

rmse = float(np.sqrt(np.mean((pred[valid] - y[valid]) ** 2)))
mae = float(np.mean(np.abs(pred[valid] - y[valid])))
bias = float(np.mean(pred[valid] - y[valid]))

print()
print(f"  in-sample (2025-03, LUT 来自 2025-02):")
print(f"    n={len(y):,}")
print(f"    RMSE = {rmse:.2f}")
print(f"    MAE  = {mae:.2f}")
print(f"    Bias = {bias:+.2f}")
print(f"    Y mean = {y[valid].mean():.2f}, Pred mean = {pred[valid].mean():.2f}")

# 分机场 in-sample RMSE (验证 EGLL/LTFM 在训练月份是否也是高 RMSE)
print()
print("  分机场 in-sample RMSE (2025-03):")
aps = df_mar_dep_full["ADEP_mvt"].astype(str).values
print(f"\n{'ADEP':6s} {'n':>8s} {'RMSE':>8s} {'Bias':>8s}")
for ap in sorted(np.unique(aps)):
    mask = (aps == ap) & valid
    if mask.sum() < 100:
        continue
    rmse_ap = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))
    bias_ap = float(np.mean(pred[mask] - y[mask]))
    flag = " ⚠️" if rmse_ap > 350 else ""
    print(f"  {ap:6s} {mask.sum():>8d} {rmse_ap:>8.2f} {bias_ap:>+8.2f}{flag}")

# ============================================================
# 5. 对比: 2025-03 in-sample RMSE vs holdout RMSE
# ============================================================
print()
print("=" * 70)
print("[5] 对比结论")
print("=" * 70)
print(f"  in-sample (2025-03) RMSE: {rmse:.2f}")
print(f"  holdout  (2025-01+07) RMSE: 333.07 (来自上次)")
print(f"  README 锚点 R7slim: 283.55")
print()
if rmse < 250:
    print(f"  ✅ in-sample RMSE << 283 → 管线对，问题在 holdout 月份口径")
elif rmse < 320:
    print(f"  ⚠️ in-sample RMSE 接近 283 → 管线部分正确, holdout 偏差有 holdout-specific 因素")
else:
    print(f"  ❌ in-sample RMSE >= 320 → 管线本身错，先修管线")