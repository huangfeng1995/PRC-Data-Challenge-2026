"""
诊断 ranking parquet 缺失率 + 特征口径一致性
==================================================

三个具体检查:
1. ranking 每列缺失率 - 找出 v1 训练用到但 ranking 缺失的列
2. 预测分布对比 - v1 预测 ranking 的 mean/median/p99 vs 训练 target 分布
3. 同月 holdout RMSE 分解 - 看 p99.9-100 极端段贡献多少 SSE(预判 clip 收益)

输出: notebooks/diag_ranking_result.json
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "notebooks" / "diag_ranking_result.json"
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)


# ============================================================
# 检查 1:ranking 每列缺失率
# ============================================================
print("=" * 70)
print("检查 1: ranking 每列缺失率 (vs train)")
print("=" * 70)

# 读取 ranking.parquet - 一次性看所有列
print("\n加载 ranking.parquet(全部列)...")
ranking_full = pd.read_parquet("data/raw/ranking.parquet")
print(f"ranking shape: {ranking_full.shape}")
print(f"ranking 列数: {len(ranking_full.columns)}")
print()

# 只看 DEP 行
ranking_dep = ranking_full[ranking_full["PHASE_mvt"] == "DEP"]
print(f"ranking DEP 行数: {len(ranking_dep):,}")

# 关键列缺失率
key_cols = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]
print(f"\n{'列名':<28} {'rank NaN%':>10} {'rank dtype':>15}")
print("-" * 60)
missing_report = {}
for c in key_cols:
    if c in ranking_dep.columns:
        nan_pct = ranking_dep[c].isna().mean() * 100
        dtype = str(ranking_dep[c].dtype)
        marker = ""
        if nan_pct > 50:
            marker = "  ❌ 大量缺失"
        elif nan_pct > 5:
            marker = "  ⚠️ 部分缺失"
        print(f"{c:<28} {nan_pct:>9.1f}% {dtype:>15}{marker}")
        missing_report[c] = {"rank_nan_pct": float(nan_pct), "rank_dtype": dtype}

print("\n❌ 100% 缺失的列(被屏蔽的):")
fully_missing = [c for c, info in missing_report.items() if info["rank_nan_pct"] > 99]
for c in fully_missing:
    print(f"  {c}")
print("\n⚠️ 部分缺失的列(1-99%):")
partial_missing = [c for c, info in missing_report.items() if 0 < info["rank_nan_pct"] < 99]
for c in partial_missing:
    print(f"  {c}: {missing_report[c]['rank_nan_pct']:.1f}%")


# ============================================================
# 检查 2:预测分布对比
# ============================================================
print("\n" + "=" * 70)
print("检查 2: v1 预测分布 vs 训练 target 分布")
print("=" * 70)

# 读 v1 预测
v1_pred = pd.read_parquet("submissions/jubilant-puzzle_v1.parquet")["TAXITIME_SEC_mvt"].values
print(f"\nv1 预测 (n={len(v1_pred):,}):")
print(f"  mean={v1_pred.mean():.1f}, median={np.median(v1_pred):.1f}, std={v1_pred.std():.1f}")
for q in [50, 75, 90, 95, 99, 99.5, 99.9]:
    print(f"  p{q:>5} = {np.percentile(v1_pred, q):.1f}")
print(f"  min={v1_pred.min():.1f}, max={v1_pred.max():.1f}")

# train 真实 target
print(f"\n训练集(2025-06, n=181,366) 真实 TAXITIME_SEC_mvt:")
tr_real = pd.read_parquet(
    "data/raw/training_2025-06-01_2025-07-01.parquet",
    columns=["TAXITIME_SEC_mvt"]
)["TAXITIME_SEC_mvt"]
tr_real = tr_real[(tr_real > 0) & (tr_real < 10800)].values
print(f"  mean={tr_real.mean():.1f}, median={np.median(tr_real):.1f}, std={tr_real.std():.1f}")
for q in [50, 75, 90, 95, 99, 99.5, 99.9]:
    print(f"  p{q:>5} = {np.percentile(tr_real, q):.1f}")
print(f"  min={tr_real.min():.1f}, max={tr_real.max():.1f}")

# 对比
print("\n预测分布 vs 真实分布差异:")
print(f"  mean 差: {v1_pred.mean() - tr_real.mean():+.1f}s  ({'偏高' if v1_pred.mean() > tr_real.mean() else '偏低'})")
print(f"  median 差: {np.median(v1_pred) - np.median(tr_real):+.1f}s")
print(f"  std 差: {v1_pred.std() - tr_real.std():+.1f}s")
print(f"  p99 差: {np.percentile(v1_pred, 99) - np.percentile(tr_real, 99):+.1f}s")
print(f"  p99.9 差: {np.percentile(v1_pred, 99.9) - np.percentile(tr_real, 99.9):+.1f}s")

# 如果 v1 预测分布 ≈ 训练真实分布,说明特征口径没崩
# 如果 v1 预测分布偏离训练真实分布,说明特征口径有差
mean_diff = v1_pred.mean() - tr_real.mean()
std_diff = v1_pred.std() - tr_real.std()
p99_diff = np.percentile(v1_pred, 99) - np.percentile(tr_real, 99)

print("\n诊断结论:")
if abs(mean_diff) > 200 or abs(std_diff) > 100 or abs(p99_diff) > 500:
    print(f"  ⚠️ 预测分布与训练差异较大(mean {mean_diff:+.0f}s, std {std_diff:+.0f}s, p99 {p99_diff:+.0f}s)")
    print("  → 特征口径可能有问题,需要修特征管道")
else:
    print(f"  ✅ 预测分布与训练分布差异较小")
    print("  → 特征口径基本正确,230 差距可能来自 ranking target 分布本身")


# ============================================================
# 检查 3:同月 holdout RMSE 分解(预判 clip 收益)
# ============================================================
print("\n" + "=" * 70)
print("检查 3: 同月 holdout RMSE 分解 (看 clip 收益上限)")
print("=" * 70)

# 加载 holdout 结果
with open("notebooks/holdout_same_month_result.json") as f:
    holdout_result = json.load(f)

print(f"\nv1 同月 holdout RMSE: {holdout_result['v1']['rmse']:.2f}")
print(f"v7 同月 holdout RMSE: {holdout_result['v7']['rmse']:.2f}")

# 重新跑 holdout 看分位段
import xgboost as xgb
from src.data_loader import filter_departures, parse_time_columns, load_month
from src.features import BASELINE_FEATURE_COLS, build_features
from src.validation import rmse

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
# 也算 v2 features(给 cnt_dep_* 用)
from src.features_aggregate import build_features_v2
df_ho_v2 = build_features_v2(df_ho_raw, include_aggregate=True, include_lead=True)
df_ho_v2 = df_ho_v2[(df_ho_v2["TAXITIME_SEC_mvt"] > 0) & (df_ho_v2["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
df_ho_basic = build_features(df_ho_raw)
df_ho_basic = df_ho_basic[(df_ho_basic["TAXITIME_SEC_mvt"] > 0) & (df_ho_basic["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
df_ho = df_ho_v2  # 用 v2(包含 cnt_dep_*)
print(f"Holdout rows: {len(df_ho):,}")

# 加载 v1 模型
import xgboost as xgb
booster = xgb.Booster()
booster.load_model("models/baseline_xgb.json")
expected = booster.feature_names
print(f"模型期望 {len(expected)} 列")

# 编码(对 v1 模型用 BASELINE_FEATURE_COLS,不需要 cnt_dep_*)
from src.train_xgb import _encode_categoricals
feat_cols = [c for c in expected if c in df_ho.columns]
missing = [c for c in expected if c not in df_ho.columns]
if missing:
    print(f"⚠️ 模型期望但 df_ho 缺: {missing}")
X_ho = df_ho[feat_cols]
X_ho_enc = _encode_categoricals(X_ho, fit=False)
y_ho = df_ho["TAXITIME_SEC_mvt"].values

# 预测
dinfer = xgb.DMatrix(X_ho_enc)
y_pred = booster.predict(dinfer)

# 按分位段分解
df_h = pd.DataFrame({"y": y_ho, "pred": y_pred})
df_h["bin"] = pd.qcut(df_h["y"], q=[0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0],
                     labels=['p0-50','p50-90','p90-95','p95-99','p99-99.5','p99.5-99.9','p99.9-100'])

total_sse = ((y_ho - y_pred) ** 2).sum()
print(f"\n总 SSE = {total_sse:,.0f}")
print(f"\n{'分位段':<14} {'行数':>8} {'真实均值':>10} {'预测均值':>10} {'子集RMSE':>10} {'SSE':>15} {'占总SSE%':>10}")
print("-" * 80)
quantile_breakdown = []
for b, sub in df_h.groupby("bin", observed=True):
    sse = ((sub["y"] - sub["pred"]) ** 2).sum()
    rmse_sub = np.sqrt(sse / len(sub))
    pct = sse / total_sse * 100
    print(f"{b:<14} {len(sub):>8,} {sub['y'].mean():>10.0f} {sub['pred'].mean():>10.0f} "
          f"{rmse_sub:>10.1f} {sse:>15,.0f} {pct:>9.1f}%")
    quantile_breakdown.append({
        "bin": b, "n": int(len(sub)), "y_mean": float(sub["y"].mean()),
        "pred_mean": float(sub["pred"].mean()),
        "rmse": float(rmse_sub), "sse": float(sse), "pct_sse": float(pct),
    })

# 预测 clip 在同月 holdout 上的最大收益估算
# 如果把 > p99.5 的预测 clip 到 p99.5 真实值,看看能省多少 SSE
y_pred_clipped = y_pred.copy()
p99_5_threshold = np.percentile(y_ho, 99.5)
top_mask = y_pred > p99_5_threshold
n_top = top_mask.sum()
savings = ((y_ho[top_mask] - y_pred[top_mask]) ** 2).sum() - ((y_ho[top_mask] - y_pred[top_mask] * 0.5) ** 2).sum()  # 假设能压到一半
print(f"\nClip 理论收益估算(p99.5 真实阈值):")
print(f"  Top 0.5% 行数: {n_top}")
print(f"  当前 SSE: {((y_ho[top_mask] - y_pred[top_mask]) ** 2).sum():,.0f}")
print(f"  如果预测减半, SSE: {((y_ho[top_mask] - y_pred[top_mask] * 0.5) ** 2).sum():,.0f}")

# 真实 clip 收益:把 > p99.5 预测全部 clip 到 p99.5 真实值的 50%
y_clip = y_pred.copy()
for i, b in enumerate(y_pred):
    if b > p99_5_threshold:
        y_clip[i] = p99_5_threshold * 0.7  # clip 到 p99.5 的 70%
rmse_clipped = rmse(y_ho, y_clip)
print(f"  Clip @ p99.5*0.7: RMSE {rmse_clipped:.2f} (vs 原 {np.sqrt(total_sse/len(y_ho)):.2f})")

# 保存全部结果
out = {
    "check1_ranking_missing": missing_report,
    "check1_fully_missing_cols": fully_missing,
    "check1_partial_missing_cols": partial_missing,
    "check2_pred_vs_train": {
        "v1_pred_mean": float(v1_pred.mean()),
        "v1_pred_median": float(np.median(v1_pred)),
        "v1_pred_std": float(v1_pred.std()),
        "v1_pred_p99": float(np.percentile(v1_pred, 99)),
        "v1_pred_p999": float(np.percentile(v1_pred, 99.9)),
        "v1_pred_max": float(v1_pred.max()),
        "train_mean": float(tr_real.mean()),
        "train_median": float(np.median(tr_real)),
        "train_std": float(tr_real.std()),
        "train_p99": float(np.percentile(tr_real, 99)),
        "train_p999": float(np.percentile(tr_real, 99.9)),
        "train_max": float(tr_real.max()),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "p99_diff": float(p99_diff),
    },
    "check3_quantile_breakdown": quantile_breakdown,
    "check3_total_sse": float(total_sse),
    "holdout_result": holdout_result,
}

with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n结果已保存到 {OUT_PATH}")
