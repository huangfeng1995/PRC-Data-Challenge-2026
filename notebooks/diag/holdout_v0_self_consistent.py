"""
阶段 0 自洽检查:v1 + v7 在标准同月 holdout 上的完整指标 + 断言
================================================================

强制要求(用户 2026-09-21 确认):
- 用 feat_basic30.json / feat_cnt36.json 同一份特征列清单, train / holdout / ranking 三端共用
- cat 编码通过 models/feature_lists/xgb_*_cat_mappings.json 显式注入,跨进程一致
- 模型加载时**严格校验列名+顺序**(booster.feature_names 与 JSON 一致)
- 输出 v1/v7 各自完整指标 + 自洽断言
- 边界(1 月 1 日凌晨、7 月 1 日凌晨)受 2024-12 / 2025-06 历史缺失影响的行数,显式报告
- 预测 ranking,对比 ranking 预测均值 vs holdout 预测均值

输出: notebooks/diag/holdout_v0_result.json
"""
import sys, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

from src.data_loader import filter_departures, parse_time_columns, load_month, load_ranking
from src.features import build_features as build_basic, BASELINE_FEATURE_COLS, CATEGORICAL_COLS
from src.features_aggregate import build_features_v2
from src.train_xgb import _encode_categoricals, _CAT_MAPPINGS_PATH

OUT_PATH = ROOT / "notebooks" / "diag" / "holdout_v0_result.json"
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

# v0 阶段0 单一可信源:三端共用同一份 feat 列清单
FEAT_BASIC30 = json.loads((ROOT / "models/feature_lists/feat_basic30.json").read_text())
FEAT_CNT36 = json.loads((ROOT / "models/feature_lists/feat_cnt36.json").read_text())

MODELS = {
    "v1": {
        "path": ROOT / "models" / "xgb_v1_basic30_seed0_val.json",
        "feat_cols": FEAT_BASIC30,
        "cat_mappings_path": ROOT / "models" / "feature_lists" / "xgb_v1_basic30_seed0_val_cat_mappings.json",
        "feature_meta_path": ROOT / "models" / "feature_lists" / "xgb_v1_basic30_seed0_val_features.json",
    },
    "v7": {
        "path": ROOT / "models" / "xgb_v7_cnt36_seed0_val.json",
        "feat_cols": FEAT_CNT36,
        "cat_mappings_path": ROOT / "models" / "feature_lists" / "xgb_v7_cnt36_seed0_val_cat_mappings.json",
        "feature_meta_path": ROOT / "models" / "feature_lists" / "xgb_v7_cnt36_seed0_val_features.json",
    },
}


def inject_cat_mappings(version: str) -> None:
    """v0.1:跨进程注入 cat 映射(否则 _encode_categoricals 会把未知类别编码成 -1)。"""
    import src.train_xgb as tx
    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS_PATH = MODELS[version]["cat_mappings_path"]


def build_features_for(version: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    """统一特征构造:三端共用 feat 清单,basic30 直接 build_basic,cnt36 build_v2 后筛选。"""
    feat_cols = MODELS[version]["feat_cols"]
    if version == "v1":
        df = build_basic(df_raw)
    elif version == "v7":
        df = build_features_v2(df_raw, include_aggregate=True, include_lead=True)
    else:
        raise ValueError(version)
    # 严格按 feat 清单选列,确保与训练一致
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{version} feat 缺失: {missing}")
    return df[feat_cols].copy()


def load_model_with_features(version: str):
    cfg = MODELS[version]
    booster = xgb.Booster()
    booster.load_model(str(cfg["path"]))
    expected = booster.feature_names
    meta = json.loads(Path(cfg["feature_meta_path"]).read_text())
    if list(expected) != list(meta["features"]):
        raise AssertionError(
            f"{version} 特征列不一致!\nbooster: {expected[:5]}...\nmeta: {meta['features'][:5]}..."
        )
    if list(expected) != list(cfg["feat_cols"]):
        raise AssertionError(
            f"{version} 特征列与 feat_*.json 不一致!\n"
            f"booster: {list(expected)[:5]}\n"
            f"feat: {list(cfg['feat_cols'])[:5]}"
        )
    print(f"  模型 {cfg['path'].name} 加载 OK,特征列 {len(expected)} 个一致 (与 feat JSON 严格一致)")
    return booster


def evaluate(version: str, df_feat: pd.DataFrame, y_ho: np.ndarray, raw_df: pd.DataFrame):
    """统一特征构造 + 模型预测 + 完整指标"""
    cfg = MODELS[version]
    booster = load_model_with_features(version)

    # 编码 + 预测
    inject_cat_mappings(version)  # v0.1:注入持久化 cat 映射
    X = df_feat[cfg["feat_cols"]]
    X_enc = _encode_categoricals(X, fit=False)
    dinfer = xgb.DMatrix(X_enc)
    y_pred = booster.predict(dinfer)

    residual = y_pred - y_ho
    airports = raw_df["ADEP_mvt"].values
    months = raw_df["SCHED_TIME_UTC_mvt"].dt.month.values

    rmse_total = float(np.sqrt(np.mean((y_pred - y_ho) ** 2)))
    mae_total = float(np.mean(np.abs(y_pred - y_ho)))
    bias = float(np.mean(residual))
    bias_pct = bias / float(np.mean(y_ho)) * 100

    # 分位段 + 自洽 SSE 加总
    df_h = pd.DataFrame({"y": y_ho, "pred": y_pred})
    df_h["bin"] = pd.qcut(df_h["y"], q=[0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0],
                         labels=['p0-50','p50-90','p90-95','p95-99','p99-99.5','p99.5-99.9','p99.9-100'])
    quantile_breakdown = []
    total_sse = 0.0
    for b, sub in df_h.groupby("bin", observed=True):
        sse = float(((sub["y"] - sub["pred"]) ** 2).sum())
        total_sse += sse
        rmse_b = float(np.sqrt(sse / len(sub)))
        quantile_breakdown.append({
            "bin": str(b), "n": int(len(sub)),
            "y_mean": float(sub["y"].mean()), "pred_mean": float(sub["pred"].mean()),
            "rmse": rmse_b, "sse": sse, "pct_sse": 0.0,
        })
    recomputed_rmse = float(np.sqrt(total_sse / len(y_ho)))
    self_consistent_sse = abs(recomputed_rmse - rmse_total) < 1e-6
    for row in quantile_breakdown:
        row["pct_sse"] = row["sse"] / total_sse * 100

    # 按机场
    by_airport = {}
    for ap in np.unique(airports):
        mask = airports == ap
        by_airport[str(ap)] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean((y_pred[mask] - y_ho[mask]) ** 2))),
            "y_mean": float(np.mean(y_ho[mask])),
            "pred_mean": float(np.mean(y_pred[mask])),
        }

    # 按月
    by_month = {}
    for m in np.unique(months):
        mask = months == m
        by_month[int(m)] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean((y_pred[mask] - y_ho[mask]) ** 2))),
            "y_mean": float(np.mean(y_ho[mask])),
            "pred_mean": float(np.mean(y_pred[mask])),
        }

    # 边界 NaN 报告:cnt_dep_* 在月初窗口可能因缺历史而 NaN
    cnt_cols = [c for c in df_feat.columns if c.startswith("cnt_")]
    boundary_nan = {}
    if cnt_cols:
        # raw_df 含 SCHED_TIME,与 df_feat 行数对齐(都是 0..N)
        for m in [1, 7]:
            mask_m = raw_df["SCHED_TIME_UTC_mvt"].dt.month == m
            for day_thr in [1, 3, 7]:
                mask_d = raw_df["SCHED_TIME_UTC_mvt"].dt.day <= day_thr
                rows_affected = int((mask_m & mask_d).sum())
                nan_cnt = int(df_feat.loc[mask_m & mask_d, cnt_cols].isna().sum().sum())
                boundary_nan[f"month{m}_day_le{day_thr}"] = {
                    "n_rows": rows_affected, "n_nan_cells": nan_cnt,
                }

    metrics = {
        "n": int(len(y_ho)),
        "y_mean": float(np.mean(y_ho)),
        "y_median": float(np.median(y_ho)),
        "y_p50": float(np.percentile(y_ho, 50)),
        "y_p90": float(np.percentile(y_ho, 90)),
        "y_p99": float(np.percentile(y_ho, 99)),
        "y_p999": float(np.percentile(y_ho, 99.9)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_median": float(np.median(y_pred)),
        "pred_p50": float(np.percentile(y_pred, 50)),
        "pred_p90": float(np.percentile(y_pred, 90)),
        "pred_p99": float(np.percentile(y_pred, 99)),
        "pred_p999": float(np.percentile(y_pred, 99.9)),
        "rmse": rmse_total,
        "mae": mae_total,
        "bias": bias,
        "bias_pct": bias_pct,
        "residual_median": float(np.median(residual)),
        "by_airport": by_airport,
        "by_month": by_month,
        "quantile_breakdown": quantile_breakdown,
        "self_consistent_sse": self_consistent_sse,
        "recomputed_rmse": recomputed_rmse,
        "boundary_nan": boundary_nan,
    }

    print(f"  RMSE={rmse_total:.2f}, MAE={mae_total:.2f}, bias={bias:+.2f}s ({bias_pct:+.1f}%)")
    print(f"  pred mean={metrics['pred_mean']:.1f}, true mean={metrics['y_mean']:.1f}")
    print(f"  Self-consistent SSE: {self_consistent_sse} (diff={abs(recomputed_rmse - rmse_total):.2e})")
    return metrics, y_pred


def main():
    print("=" * 70)
    print("阶段 0 自洽检查:v1 + v7 在标准同月 holdout 上的完整指标")
    print("=" * 70)

    # 模型文件 hash,用于 356 vs 855 根因排查
    for v, cfg in MODELS.items():
        h = hashlib.sha256(cfg["path"].read_bytes()).hexdigest()[:16]
        print(f"[model] {v}: {cfg['path'].name}  sha256={h}  features={len(cfg['feat_cols'])}")

    # 加载 holdout(2025-01 + 2025-07)
    print("\n[1] 加载 holdout(2025-01+07)...")
    dfs = []
    for m in [1, 7]:
        fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
        dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
    df_ho_raw = pd.concat(dfs, ignore_index=True)
    df_ho_raw = filter_departures(df_ho_raw)
    df_ho_raw = parse_time_columns(df_ho_raw, cols=TIME_COLS)
    print(f"  Holdout raw: {len(df_ho_raw):,}")

    df_ho_raw = df_ho_raw[(df_ho_raw["TAXITIME_SEC_mvt"] > 0) &
                          (df_ho_raw["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
    print(f"  过滤后 holdout: {len(df_ho_raw):,}")

    # 边界 NaN 统计(全表)
    cnt_cols_v7 = [c for c in MODELS["v7"]["feat_cols"] if c.startswith("cnt_")]
    boundary_nan_globally = {}
    for m in [1, 7]:
        mask_m = df_ho_raw["SCHED_TIME_UTC_mvt"].dt.month == m
        for day_thr in [1, 3, 7, 14]:
            mask_d = df_ho_raw["SCHED_TIME_UTC_mvt"].dt.day <= day_thr
            boundary_nan_globally[f"month{m}_day_le{day_thr}"] = int((mask_m & mask_d).sum())
    print(f"  边界样本分布: {boundary_nan_globally}")

    # 特征构造(三端统一)
    print("\n[2] 构造特征(统一管线)...")
    df_ho_v1 = build_features_for("v1", df_ho_raw.copy())
    df_ho_v7 = build_features_for("v7", df_ho_raw.copy())
    print(f"  v1 holdout: {df_ho_v1.shape}, v7 holdout: {df_ho_v7.shape}")

    y_ho = df_ho_raw["TAXITIME_SEC_mvt"].values

    # 模型评估
    print("\n[3] 模型评估(v1 + v7)...")
    v1_metrics, v1_pred = evaluate("v1", df_ho_v1, y_ho, df_ho_raw)
    v7_metrics, v7_pred = evaluate("v7", df_ho_v7, y_ho, df_ho_raw)

    # ranking 推理(用同一份代码)— ranking vs holdout 预测均值差
    print("\n[4] ranking 推理(用同一份 feat 清单 + 同一份 cat 映射)...")
    df_rank_raw = load_ranking("data/raw")
    df_rank_v1 = build_features_for("v1", df_rank_raw.copy())
    df_rank_v7 = build_features_for("v7", df_rank_raw.copy())

    # v1 ranking
    inject_cat_mappings("v1")
    booster_v1 = load_model_with_features("v1")
    X_v1_enc = _encode_categoricals(df_rank_v1[MODELS["v1"]["feat_cols"]], fit=False)
    rank_pred_v1 = booster_v1.predict(xgb.DMatrix(X_v1_enc))

    # v7 ranking
    inject_cat_mappings("v7")
    booster_v7 = load_model_with_features("v7")
    X_v7_enc = _encode_categoricals(df_rank_v7[MODELS["v7"]["feat_cols"]], fit=False)
    rank_pred_v7 = booster_v7.predict(xgb.DMatrix(X_v7_enc))

    rank_metrics = {
        "v1_rank_pred_mean": float(np.mean(rank_pred_v1)),
        "v7_rank_pred_mean": float(np.mean(rank_pred_v7)),
        "v1_rank_pred_median": float(np.median(rank_pred_v1)),
        "v7_rank_pred_median": float(np.median(rank_pred_v7)),
        "v1_rank_pred_p99": float(np.percentile(rank_pred_v1, 99)),
        "v7_rank_pred_p99": float(np.percentile(rank_pred_v7, 99)),
        "v1_holdout_pred_mean": v1_metrics["pred_mean"],
        "v7_holdout_pred_mean": v7_metrics["pred_mean"],
        "v1_diff_rank_vs_holdout": float(np.mean(rank_pred_v1) - v1_metrics["pred_mean"]),
        "v7_diff_rank_vs_holdout": float(np.mean(rank_pred_v7) - v7_metrics["pred_mean"]),
        "ranking_n": int(len(rank_pred_v1)),
    }
    print(f"  v1 ranking pred mean: {rank_metrics['v1_rank_pred_mean']:.2f}  vs holdout {v1_metrics['pred_mean']:.2f} (diff {rank_metrics['v1_diff_rank_vs_holdout']:+.2f})")
    print(f"  v7 ranking pred mean: {rank_metrics['v7_rank_pred_mean']:.2f}  vs holdout {v7_metrics['pred_mean']:.2f} (diff {rank_metrics['v7_diff_rank_vs_holdout']:+.2f})")

    # 自洽断言
    print("\n" + "=" * 70)
    print("自洽断言")
    print("=" * 70)
    assertions = {}

    # 1. SSE 分桶加总 = 总 RMSE^2 * n
    print(f"  [1] SSE 自洽:")
    sse_ok_v1 = v1_metrics["self_consistent_sse"]
    sse_ok_v7 = v7_metrics["self_consistent_sse"]
    print(f"      v1: {sse_ok_v1} (diff={abs(v1_metrics['recomputed_rmse'] - v1_metrics['rmse']):.2e})")
    print(f"      v7: {sse_ok_v7} (diff={abs(v7_metrics['recomputed_rmse'] - v7_metrics['rmse']):.2e})")
    assertions["sse_self_consistent"] = {"v1": sse_ok_v1, "v7": sse_ok_v7, "pass": sse_ok_v1 and sse_ok_v7}

    # 2. 全局 RMSE 范围(预期 300 多) — 只作用于整体
    print(f"  [2] RMSE 范围(预期 250-450):")
    rmse_ok_v1 = 250 < v1_metrics["rmse"] < 450
    rmse_ok_v7 = 250 < v7_metrics["rmse"] < 450
    print(f"      v1: {v1_metrics['rmse']:.2f}  (in 250-450: {rmse_ok_v1})")
    print(f"      v7: {v7_metrics['rmse']:.2f}  (in 250-450: {rmse_ok_v7})")
    assertions["rmse_in_range"] = {"v1": rmse_ok_v1, "v7": rmse_ok_v7, "pass": rmse_ok_v1 and rmse_ok_v7}

    # 3. bias 相对口径 ≤ 0.08(8%)
    print(f"  [3] bias 相对(≤ 8%):")
    bias_ok_v1 = abs(v1_metrics["bias_pct"]) <= 8.0
    bias_ok_v7 = abs(v7_metrics["bias_pct"]) <= 8.0
    print(f"      v1: pred={v1_metrics['pred_mean']:.1f}  true={v1_metrics['y_mean']:.1f}  bias={v1_metrics['bias']:+.1f}s ({v1_metrics['bias_pct']:+.2f}%)  pass={bias_ok_v1}")
    print(f"      v7: pred={v7_metrics['pred_mean']:.1f}  true={v7_metrics['y_mean']:.1f}  bias={v7_metrics['bias']:+.1f}s ({v7_metrics['bias_pct']:+.2f}%)  pass={bias_ok_v7}")
    assertions["bias_rel_le_8pct"] = {"v1": bias_ok_v1, "v7": bias_ok_v7, "pass": bias_ok_v1 and bias_ok_v7}

    # 4. ranking vs holdout 预测均值差 < 100(容许一定分布差)
    print(f"  [4] ranking vs holdout 预测均值差(参考 < 100):")
    diff_ok_v1 = abs(rank_metrics["v1_diff_rank_vs_holdout"]) < 100
    diff_ok_v7 = abs(rank_metrics["v7_diff_rank_vs_holdout"]) < 100
    print(f"      v1: {rank_metrics['v1_diff_rank_vs_holdout']:+.2f}  pass={diff_ok_v1}")
    print(f"      v7: {rank_metrics['v7_diff_rank_vs_holdout']:+.2f}  pass={diff_ok_v7}")
    assertions["rank_vs_holdout_pred_mean"] = {"v1": diff_ok_v1, "v7": diff_ok_v7, "pass": diff_ok_v1 and diff_ok_v7}

    # 总断言
    all_pass = all(a["pass"] for a in assertions.values())
    print(f"\n全部断言: {'✅ 通过' if all_pass else '❌ 失败'}")
    assertions["all_pass"] = all_pass

    out = {
        "v1": v1_metrics,
        "v7": v7_metrics,
        "ranking_metrics": rank_metrics,
        "boundary_report": boundary_nan_globally,
        "boundary_nan_per_feat": {"v7": v7_metrics["boundary_nan"]},
        "assertions": assertions,
        "models": {v: str(cfg["path"].relative_to(ROOT)) for v, cfg in MODELS.items()},
        "feat_files": {
            "basic30": "models/feature_lists/feat_basic30.json",
            "cnt36": "models/feature_lists/feat_cnt36.json",
        },
        "model_hashes": {
            v: hashlib.sha256(cfg["path"].read_bytes()).hexdigest()[:16]
            for v, cfg in MODELS.items()
        },
        "holdout_period": "2025-01 + 2025-07",
        "train_period_excluded": "2025-01, 2025-07",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n结果已保存到 {OUT_PATH}")

    if not all_pass:
        print("\n❌ 断言失败 — 不进入阶段 1,先排查:")
        for k, a in assertions.items():
            if k != "all_pass" and not a["pass"]:
                print(f"  - {k}: v1={a.get('v1')} v7={a.get('v7')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
