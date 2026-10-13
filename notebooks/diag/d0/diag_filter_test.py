"""
排查 7: 留存 R7slim 模型 + 重建特征, 按 TAXITIME 阈值剔除后评 RMSE
- 不剔除 (已知 RMSE=333)
- 剔除 TAXITIME > 3600 (1h), > 7200 (2h), > 10800 (3h)
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
import _r7_lib as L
from src.train_xgb import _encode_categoricals
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

# ============================================================
# 1. 加载 holdout (1+7)
# ============================================================
print("=" * 70)
print("[1] 加载 holdout (2025-01+07)")
print("=" * 70)
ho_dfs = []
for m in [1, 7]:
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    ho_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)

# ============================================================
# 2. 加载训练 (2-12), 构建 LUT
# ============================================================
print()
print("=" * 70)
print("[2] 加载训练 (2025-02~12)")
print("=" * 70)
months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
train_dfs = []
for m in months:
    fname = (f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
             if m < 12 else "training_2025-12-01_2026-01-01.parquet")
    train_dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_tr_raw = pd.concat(train_dfs, ignore_index=True)
df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
df_tr_dep = df_tr[
    (df_tr["PHASE_mvt"] == "DEP") &
    (df_tr["TAXITIME_SEC_mvt"] > 0) &
    (df_tr["TAXITIME_SEC_mvt"] < 10800)
].copy().reset_index(drop=True)
print(f"  训练 DEP: {len(df_tr_dep):,}")

lut = L.build_t_unimp_lut(df_tr_dep)

# ============================================================
# 3. 重建 holdout 特征 (与 holdout_r7_rebuild 一致)
# ============================================================
print()
print("=" * 70)
print("[3] 重建 holdout 特征")
print("=" * 70)

df_ho_basic = build_basic(df_ho.copy())
df_ho_full = add_queue_features(
    df_ho_basic.copy(),
    include_a=True, include_b=True, include_c=True,
    include_d=True, include_e=True,
)
df_ho_full["ref_was_estimated"] = df_ho_full["AOBT_3_flt"].isna().astype("int8")
df_ho_full["is_diverted"] = (df_ho_full["ADES_flt"] != df_ho_full["ADES_FILED_flt"]).astype("int8")

d_dict = D.compute_drift_features(df_ho_full, ref_dep=df_tr, label="holdout")
for k, v in d_dict.items():
    df_ho_full[k] = v

t_vals, _ = L.lookup_t_unimp(df_ho_full, lut)
df_ho_full["t_unimp_r7"] = t_vals

dep_mask = (df_ho_full["PHASE_mvt"] == "DEP").values
extra_dict = L.compute_extra_features(df_ho_full[dep_mask], label="holdout")
for k, v in extra_dict.items():
    df_ho_full[k] = np.nan
    df_ho_full.loc[dep_mask, k] = v

mvt_dict = L.compute_mvt_anchor_features(
    df_ho_full[dep_mask],
    t_unimp_vals=t_vals[dep_mask], label="holdout")
for k, v in mvt_dict.items():
    df_ho_full[k] = np.nan
    df_ho_full.loc[dep_mask, k] = v

df_ho_dep_full = df_ho_full[dep_mask].copy().reset_index(drop=True)
print(f"  Holdout DEP: {len(df_ho_dep_full):,}")

# ============================================================
# 4. 留存 R7slim 模型推理
# ============================================================
print()
print("=" * 70)
print("[4] 留存 R7slim 模型推理")
print("=" * 70)

bst = xgb.Booster()
bst.load_model(str(ROOT / "models" / "xgb_v10_R7slim_12m_seed0_final.json"))
FEATS69 = bst.feature_names

tx_cat = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
from src import train_xgb as tx
tx._CAT_MAPPINGS.clear()
tx._CAT_MAPPINGS = tx_cat

X = df_ho_dep_full[FEATS69].copy()
X_enc = _encode_categoricals(X, fit=False).values.astype(np.float32)
dmat = xgb.DMatrix(X_enc, feature_names=FEATS69)
pred = bst.predict(dmat)
pred = np.clip(pred, 30, 10800)

y = df_ho_dep_full["TAXITIME_SEC_mvt"].values
months_col = pd.to_datetime(df_ho_dep_full["MVT_TIME_UTC_mvt"]).dt.month.values

# ============================================================
# 5. 留存模型预测 TAXITIME>10000 的极端样本 (看老程序怎么对待)
# ============================================================
print()
print("=" * 70)
print("[5] 留存模型预测 TAXITIME > 10000 的极端样本 (是否预测>7200?)")
print("=" * 70)

# 注意: 当前 df_ho_dep_full 来自 holdout_r7_rebuild 流程, 用的是 TAXITIME>0 过滤, 但要看完整 holdout
df_ho_all = df_ho[(df_ho["PHASE_mvt"] == "DEP")].copy().reset_index(drop=True)
y_all = df_ho_all["TAXITIME_SEC_mvt"].values
extreme_mask = (y_all > 10000) & np.isfinite(y_all)
n_extreme = int(extreme_mask.sum())
print(f"  Holdout DEP (全): {len(df_ho_all):,}, TAXITIME > 10000: {n_extreme}")

# 这 n_extreme 行的 TAXITIME 分布
if n_extreme > 0:
    extreme_y = y_all[extreme_mask]
    print(f"  极端值 TAXITIME: min={extreme_y.min():.0f}, p50={np.median(extreme_y):.0f}, max={extreme_y.max():.0f}")
    # 但这些极端行没在 df_ho_dep_full 里 (因为过滤了 < 10800)
    # 我们看看 df_ho_dep_full 的 TAXITIME 分布, 是否含 > 10000
    n_in_pred = ((y > 10000) & np.isfinite(y)).sum()
    print(f"  留存模型实际预测的 (df_ho_dep_full 内) TAXITIME > 10000: {n_in_pred}")

# ============================================================
# 6. 分档评估 RMSE (按 TAXITIME 阈值剔除)
# ============================================================
print()
print("=" * 70)
print("[6] 分档评估 RMSE (按 TAXITIME 阈值剔除)")
print("=" * 70)

# 移除极端值: 仅保留 TAXITIME <= threshold
thresholds = [10800, 7200, 3600, 1800]  # 3h, 2h, 1h, 30min
valid_mask_base = np.isfinite(y) & (y > 0)

print(f"\n基线 (含全部, TAXITIME 0<x<10800 过滤后): n={valid_mask_base.sum():,}")
rmse_all = float(np.sqrt(np.mean((pred[valid_mask_base] - y[valid_mask_base]) ** 2)))
mae_all = float(np.mean(np.abs(pred[valid_mask_base] - y[valid_mask_base])))
print(f"  RMSE = {rmse_all:.2f}, MAE = {mae_all:.2f}")

print(f"\n各档评估 (剔除 TAXITIME > threshold):")
print(f"{'阈值':>10s} {'剔除行数':>10s} {'剩余 n':>10s} {'RMSE':>10s} {'MAE':>10s} {'Δ vs 全部':>10s}")
for thr in thresholds:
    keep_mask = valid_mask_base & (y <= thr)
    excl_mask = valid_mask_base & (y > thr)
    n_excl = int(excl_mask.sum())
    n_keep = int(keep_mask.sum())
    if n_keep < 100:
        continue
    rmse = float(np.sqrt(np.mean((pred[keep_mask] - y[keep_mask]) ** 2)))
    mae = float(np.mean(np.abs(pred[keep_mask] - y[keep_mask])))
    delta = rmse - rmse_all
    print(f"{'>'+str(thr):>10s} {n_excl:>10d} {n_keep:>10d} {rmse:>10.2f} {mae:>10.2f} {delta:>+10.2f}")

# 反向: 保留 TAXITIME >= threshold (只对极端)
print(f"\n反向 (只保留 TAXITIME >= threshold):")
print(f"{'阈值':>10s} {'保留 n':>10s} {'RMSE':>10s} {'MAE':>10s}")
for thr in [3600, 7200, 10800, 30000]:
    keep_mask = valid_mask_base & (y >= thr)
    n_keep = int(keep_mask.sum())
    if n_keep < 10:
        continue
    rmse = float(np.sqrt(np.mean((pred[keep_mask] - y[keep_mask]) ** 2)))
    mae = float(np.mean(np.abs(pred[keep_mask] - y[keep_mask])))
    print(f"{'>='+str(thr):>10s} {n_keep:>10d} {rmse:>10.2f} {mae:>10.2f}")

# ============================================================
# 7. 留存模型预测 vs 真值的"训练时假设"
# ============================================================
print()
print("=" * 70)
print("[7] 留存模型预测分布 (按 TAXITIME 桶)")
print("=" * 70)

print(f"\n{'TAXITIME 桶':>15s} {'n':>10s} {'真均值':>10s} {'预测均值':>12s} {'预测 p99':>10s} {'预测 max':>10s}")
bins = [(0, 600), (600, 1800), (1800, 3600), (3600, 7200), (7200, 10800)]
for lo, hi in bins:
    mask = valid_mask_base & (y > lo) & (y <= hi)
    n = int(mask.sum())
    if n < 50:
        continue
    y_mean = y[mask].mean()
    pred_mean = pred[mask].mean()
    pred_p99 = np.percentile(pred[mask], 99)
    pred_max = pred[mask].max()
    print(f"{lo:>5d}-{hi:>5d} {n:>10d} {y_mean:>10.0f} {pred_mean:>12.0f} {pred_p99:>10.0f} {pred_max:>10.0f}")

# 预测 vs 真值在极端区间 (>10800 已过滤但训练时可能有)
print(f"\n  预测 max = {pred.max():.0f}, 训练时的预测应该 clip 到 <=10800")
print(f"  真值 max (在 df_ho_dep_full 中) = {y[valid_mask_base].max():.0f}")