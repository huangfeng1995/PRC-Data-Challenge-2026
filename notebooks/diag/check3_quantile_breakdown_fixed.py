"""
修检查 3 的 RMSE 分解脚本(对比 v1 vs v7 在同月 holdout 上的分解)
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
from src.features_aggregate import build_features_v2
from src.train_xgb import _encode_categoricals

OUT_PATH = ROOT / "notebooks" / "diag_check3_fixed_result.json"
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

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


print("=" * 70)
print("修检查 3:对比 v1 和 v7 在同月 holdout 上的 RMSE 分解")
print("=" * 70)

# 加载 holdout
print("\n加载 holdout(2025-01+07)...")
dfs = []
for m in [1, 7]:
    fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
    dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
df_ho_raw = pd.concat(dfs, ignore_index=True)
df_ho_raw = filter_departures(df_ho_raw)
df_ho_raw = parse_time_columns(df_ho_raw, cols=TIME_COLS)
df_ho_v2 = build_features_v2(df_ho_raw, include_aggregate=True, include_lead=True)
df_ho_v2 = df_ho_v2[(df_ho_v2["TAXITIME_SEC_mvt"] > 0) & (df_ho_v2["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
df_ho_basic = build_basic(df_ho_raw)
df_ho_basic = df_ho_basic[(df_ho_basic["TAXITIME_SEC_mvt"] > 0) & (df_ho_basic["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
y_ho = df_ho_v2["TAXITIME_SEC_mvt"].values
print(f"Holdout rows: {len(df_ho_v2):,}")

def load_model_predict(model_path, df_ho, model_label):
    booster = xgb.Booster()
    booster.load_model(model_path)
    expected = booster.feature_names
    feat_cols = [c for c in expected if c in df_ho.columns]
    missing = [c for c in expected if c not in df_ho.columns]
    if missing:
        print(f"  [{model_label}] 模型期望但 df 缺: {missing[:3]}...")
        # 缺列填 NaN
        for c in missing:
            df_ho[c] = np.nan
    X = df_ho[expected]
    X_enc = _encode_categoricals(X, fit=False)
    dinfer = xgb.DMatrix(X_enc)
    return booster.predict(dinfer)

# v1 模型(刚重训完)
v1_model_path = "models/baseline_xgb.json"
v7_model_path = "models/baseline_xgb_v7_snapshot.json"

results = {}

for label, model_path in [("v1", v1_model_path), ("v7", v7_model_path)]:
    print(f"\n--- {label} 模型({model_path}) ---")
    y_pred = load_model_predict(model_path, df_ho_v2.copy(), label)
    rmse_total = np.sqrt(((y_pred - y_ho) ** 2).mean())
    print(f"  整体 RMSE: {rmse_total:.2f}")

    # 分位段分解
    df_h = pd.DataFrame({"y": y_ho, "pred": y_pred})
    df_h["bin"] = pd.qcut(df_h["y"], q=[0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0],
                         labels=['p0-50','p50-90','p90-95','p95-99','p99-99.5','p99.5-99.9','p99.9-100'])

    total_sse = ((y_pred - y_ho) ** 2).sum()
    print(f"\n{'分位段':<14} {'行数':>8} {'真实均值':>10} {'预测均值':>10} {'子集RMSE':>10} {'占总SSE%':>10}")
    print("-" * 75)
    quantile_breakdown = []
    for b, sub in df_h.groupby("bin", observed=True):
        sse = ((sub["y"] - sub["pred"]) ** 2).sum()
        rmse_sub = np.sqrt(sse / len(sub))
        pct = sse / total_sse * 100
        print(f"{b:<14} {len(sub):>8,} {sub['y'].mean():>10.0f} {sub['pred'].mean():>10.0f} "
              f"{rmse_sub:>10.1f} {pct:>9.1f}%")
        quantile_breakdown.append({
            "bin": b, "n": int(len(sub)),
            "y_mean": float(sub["y"].mean()), "pred_mean": float(sub["pred"].mean()),
            "rmse": float(rmse_sub), "sse": float(sse), "pct_sse": float(pct),
        })

    # 残差分析
    residual = y_pred - y_ho
    print(f"\n  残差 mean: {residual.mean():+.2f}s (预测 - 真实)")
    print(f"  残差 median: {np.median(residual):+.2f}s")
    print(f"  预测 mean: {y_pred.mean():.2f}s, 真实 mean: {y_ho.mean():.2f}s, 差: {y_pred.mean()-y_ho.mean():+.2f}s")

    # clip p99.5 理论收益
    p99_5_threshold = np.percentile(y_ho, 99.5)
    y_clip = y_pred.copy()
    n_clip = (y_pred > p99_5_threshold).sum()
    y_clip[y_clip > p99_5_threshold] = p99_5_threshold
    rmse_clip = np.sqrt(((y_clip - y_ho) ** 2).mean())
    print(f"  Clip @ p99.5 ({p99_5_threshold:.0f}s, 影响 {n_clip} 行): RMSE {rmse_clip:.2f} (-{rmse_total-rmse_clip:.2f})")

    results[label] = {
        "rmse": float(rmse_total),
        "pred_mean": float(y_pred.mean()),
        "true_mean": float(y_ho.mean()),
        "mean_diff": float(y_pred.mean() - y_ho.mean()),
        "residual_mean": float(residual.mean()),
        "residual_median": float(np.median(residual)),
        "quantile_breakdown": quantile_breakdown,
        "clip_p995_rmse": float(rmse_clip),
        "clip_p995_delta": float(rmse_total - rmse_clip),
    }

print(f"\n" + "=" * 70)
print(f"v1 vs v7 总览")
print(f"=" * 70)
print(f"{'指标':<25} {'v1':>12} {'v7':>12}")
for k in ["rmse", "pred_mean", "true_mean", "mean_diff", "residual_mean", "residual_median", "clip_p995_rmse", "clip_p995_delta"]:
    print(f"{k:<25} {results['v1'][k]:>12.3f} {results['v7'][k]:>12.3f}")

# 数字自洽检查
print(f"\n数字自洽检查:")
v1_total_sse = sum(b["sse"] for b in results["v1"]["quantile_breakdown"])
v1_recomputed_rmse = np.sqrt(v1_total_sse / len(y_ho))
print(f"  v1 分位 SSE 总和 = {v1_total_sse:,.0f}")
print(f"  v1 整体 RMSE (重算) = {v1_recomputed_rmse:.2f}")
print(f"  v1 整体 RMSE (直算) = {results['v1']['rmse']:.2f}")
print(f"  差异 = {abs(v1_recomputed_rmse - results['v1']['rmse']):.4f}  {'✅ 自洽' if abs(v1_recomputed_rmse - results['v1']['rmse']) < 0.01 else '❌ 不自洽'}")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n结果已保存到 {OUT_PATH}")
