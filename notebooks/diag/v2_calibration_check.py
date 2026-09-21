"""
验证 v1 在同月 holdout 上的预测分布 vs 真实分布
====================================================

直接拿 v1 模型(features only basic),预测 2025-01+07 holdout,看:
- 预测 mean / median / p99  vs  真实 mean / median / p99
- 残差均值(系统性偏置检测)
- 按机场/分位段看

这是 v2 的"10 分钟验证":定案"模型偏置 vs 分布差异"假设
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

from src.data_loader import filter_departures, parse_time_columns, load_month
from src.features import build_features as build_basic, BASELINE_FEATURE_COLS
from src.train_xgb import _encode_categoricals

OUT_PATH = ROOT / "notebooks" / "diag_v2_calibration_result.json"

print("=" * 70)
print("V2 验证:v1 在同月 holdout 上的预测分布")
print("=" * 70)

# 加载 v1 模型(只 basic 特征)
print("\n加载 v1 模型...")
booster = xgb.Booster()
booster.load_model("models/baseline_xgb.json")
expected = booster.feature_names
print(f"模型期望 {len(expected)} 列")

# 加载 holdout(2025-01+07)
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

print("\n加载 holdout(2025-01+07)...")
dfs = []
for m in [1, 7]:
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(dfs, ignore_index=True)
df_ho_raw = filter_departures(df_ho_raw)
df_ho_raw = parse_time_columns(df_ho_raw, cols=TIME_COLS)

# v1 模型只用 basic features
from src.features_aggregate import build_features_v2
df_ho = build_features_v2(df_ho_raw, include_aggregate=True, include_lead=True)
df_ho = df_ho[(df_ho["TAXITIME_SEC_mvt"] > 0) & (df_ho["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
print(f"Holdout rows: {len(df_ho):,}")

# 编码 + 预测
feat_cols = [c for c in expected if c in df_ho.columns]
missing = [c for c in expected if c not in df_ho.columns]
if missing:
    print(f"⚠️ 模型期望但 holdout 缺: {missing}")
X_ho = df_ho[feat_cols]
X_ho_enc = _encode_categoricals(X_ho, fit=False)
y_ho = df_ho["TAXITIME_SEC_mvt"].values

dinfer = xgb.DMatrix(X_ho_enc)
y_pred = booster.predict(dinfer)

# 残差分析
residual = y_pred - y_ho
print(f"\n=== 同月 holdout (2025-01+07) 预测 vs 真实 ===")
print(f"真实 mean: {y_ho.mean():.2f}s, median: {np.median(y_ho):.2f}s")
print(f"预测 mean: {y_pred.mean():.2f}s, median: {np.median(y_pred):.2f}s")
print(f"差异: mean {y_pred.mean()-y_ho.mean():+.2f}s, median {np.median(y_pred)-np.median(y_ho):+.2f}s")
print(f"残差 mean: {residual.mean():+.2f}s (预测 - 真实)")
print(f"残差 median: {np.median(residual):+.2f}s")
print(f"残差 std: {residual.std():.2f}s")
print(f"残差 p25/p75: {np.percentile(residual, 25):+.1f} / {np.percentile(residual, 75):+.1f}s")

# 分位段分析
print(f"\n分位段对比:")
print(f"{'分位':<6} {'真实':>8} {'预测':>8} {'差':>8}")
for q in [50, 75, 90, 95, 99, 99.5, 99.9]:
    real_q = np.percentile(y_ho, q)
    pred_q = np.percentile(y_pred, q)
    diff = pred_q - real_q
    print(f"p{q:<5} {real_q:>8.1f} {pred_q:>8.1f} {diff:>+8.1f}")

# 按月份分解
print(f"\n按月份分解:")
sched = df_ho["SCHED_TIME_UTC_mvt"]
y_ho_arr = y_ho
for m in [1, 7]:
    mask = sched.dt.month == m
    if mask.sum() == 0:
        continue
    rmse_m = np.sqrt(((y_pred[mask] - y_ho_arr[mask]) ** 2).mean())
    mean_diff = y_pred[mask].mean() - y_ho_arr[mask].mean()
    print(f"  {m}月 (n={mask.sum():,}): RMSE={rmse_m:.2f}, 预测-真实 mean={mean_diff:+.2f}s")

# 按机场分解
print(f"\n按机场分解:")
for ap in df_ho["ADEP_mvt"].value_counts().head(5).index:
    mask = df_ho["ADEP_mvt"].values == ap
    rmse_ap = np.sqrt(((y_pred[mask] - y_ho_arr[mask]) ** 2).mean())
    mean_diff = y_pred[mask].mean() - y_ho_arr[mask].mean()
    print(f"  {ap} (n={mask.sum():,}): RMSE={rmse_ap:.0f}, 预测-真实 mean={mean_diff:+.0f}s")

# 整体 RMSE
rmse_total = np.sqrt(((y_pred - y_ho_arr) ** 2).mean())
print(f"\n整体 RMSE: {rmse_total:.2f}")

# 诊断结论
print(f"\n" + "=" * 70)
print("诊断结论:")
print("=" * 70)
mean_bias = y_pred.mean() - y_ho.mean()
if abs(mean_bias) < 30:
    print(f"✅ v1 在同月 holdout 上预测-真实 mean bias = {mean_bias:+.2f}s (很小)")
    print("   → 模型无系统性偏置,ranking 上的高估(1024 vs 真实可能是 1100-1300)必然是分布差异")
    print("   → 校准方向(value shift by +100~300)值得尝试")
elif abs(mean_bias) < 100:
    print(f"⚠️ v1 在 holdout 上有 {mean_bias:+.2f}s 的系统性偏置(中等)")
    print("   → 部分是模型偏置,部分是分布差异,需要分别校正")
else:
    print(f"❌ v1 在 holdout 上有 {mean_bias:+.2f}s 的严重系统性偏置")
    print("   → 主要是模型偏置,训练侧需要修")

# 残差 vs ranking 推断
print(f"\nranking 上的预测 mean = 1024.9")
print(f"如果 ranking 真实分布 ≈ train(同月 holdout): ranking 真实 mean ≈ {y_ho.mean():.0f}")
print(f"如果 ranking 真实分布偏高:ranking 真实 mean ≈ {y_ho.mean() + (1024.9 - y_ho.mean()):.0f} (即模型保持不变)")
print(f"\n实际 LB RMSE = 589.91,如果 LB 真实分布与 train 同步,RMSE 应该 ≈ {rmse_total:.0f}")
print(f"如果 LB RMSE = 590,说明 ranking 真实分布比 holdout 复杂得多")

# 保存
out = {
    "holdout_period": "2025-01 + 2025-07",
    "n_rows": len(df_ho),
    "y_true_mean": float(y_ho.mean()),
    "y_true_median": float(np.median(y_ho)),
    "y_pred_mean": float(y_pred.mean()),
    "y_pred_median": float(np.median(y_pred)),
    "mean_bias": float(mean_bias),
    "rmse": float(rmse_total),
    "residual_stats": {
        "mean": float(residual.mean()),
        "median": float(np.median(residual)),
        "std": float(residual.std()),
        "p25": float(np.percentile(residual, 25)),
        "p75": float(np.percentile(residual, 75)),
    },
    "quantile_diff": {
        str(q): {
            "real": float(np.percentile(y_ho, q)),
            "pred": float(np.percentile(y_pred, q)),
            "diff": float(np.percentile(y_pred, q) - np.percentile(y_ho, q)),
        }
        for q in [50, 75, 90, 95, 99, 99.5, 99.9]
    },
    "ranking_prediction_mean": 1024.9,
    "ranking_actual_lb_rmse": 589.91,
}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n结果已保存到 {OUT_PATH}")
