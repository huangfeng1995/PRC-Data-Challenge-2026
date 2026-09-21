"""
Holdout 验证 v8:v7 特征集 + 长尾 sample_weight
==========================================

测试:给 top 1% 长尾 taxi-out 样本加权,验证 RMSE 是否改善。

基准:v1=319.40, v7=311.48

测试:
- v8a: tail_weight=2.0, tail_quantile=0.99(给 top 1% 权重 2)
- v8b: tail_weight=3.0, tail_quantile=0.99
- v8c: tail_weight=2.0, tail_quantile=0.95(给 top 5% 权重 2)

观察:每个版本在长尾分位段的 RMSE 改善情况
"""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    filter_departures, parse_time_columns, load_month
)
from src.features import build_features as build_basic, BASELINE_FEATURE_COLS
from src.features_aggregate import build_features_v2
from src.validation import rmse, mae, p95_error

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

DEFAULT_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 5,
    "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0,
    "tree_method": "hist", "nthread": -1, "verbosity": 0, "seed": 42,
}

TIME_COLS = [
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt",
]


def load_month_range(months, year=2025):
    data_dir = ROOT / "data" / "raw"
    dfs = []
    for m in months:
        if m == 12:
            fname = f"training_{year}-{m:02d}-01_{year+1}-01-01.parquet"
        else:
            fname = f"training_{year}-{m:02d}-01_{year}-{m+1:02d}-01.parquet"
        path = data_dir / fname
        d = load_month(path, columns=TRAIN_COLS)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = filter_departures(df)
    df = parse_time_columns(df, cols=TIME_COLS)
    return df


def encode_categoricals(X_tr, X_pred):
    X_tr = X_tr.copy(); X_pred = X_pred.copy()
    for col in X_tr.select_dtypes(include="category").columns:
        cats = X_tr[col].cat.categories
        cat_map = {c: i for i, c in enumerate(cats)}
        cat_map["__nan__"] = -1
        for X in [X_tr, X_pred]:
            X[col] = X[col].astype(str).map(cat_map).fillna(-1).astype("int32")
    return X_tr, X_pred


def select_features(df, feature_set="no_laglead"):
    if feature_set == "no_laglead":
        keep_agg = [c for c in df.columns if c.endswith("m") and "cnt_" in c]
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns] + keep_agg
    raise ValueError(feature_set)


def make_sample_weight(y, quantile, weight):
    """长尾加权:> quantile 的样本用 weight,其他 1.0"""
    threshold = y.quantile(quantile)
    return np.where(y > threshold, weight, 1.0).astype("float32"), threshold


def train_with_weights(X_tr, y_tr, sw_tr, X_va, y_va, params):
    X_tr_enc, X_va_enc = encode_categoricals(X_tr, X_va)
    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr, weight=sw_tr)
    dval = xgb.DMatrix(X_va_enc, label=y_va)
    booster = xgb.train(
        params, dtrain, num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100, verbose_eval=False,
    )
    y_pred = booster.predict(dval)
    return y_pred, booster


def evaluate_by_quantile(y_true, y_pred):
    """按 taxi-out 真实值分位段评估"""
    df = pd.DataFrame({"y": y_true, "pred": y_pred})
    df["bin"] = pd.qcut(df["y"], q=[0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0],
                        labels=['p0-50','p50-90','p90-95','p95-99','p99-99.5','p99.5-99.9','p99.9-100'])
    rows = []
    for b, sub in df.groupby("bin", observed=True):
        sse = ((sub["y"] - sub["pred"]) ** 2).sum()
        rows.append({
            "bin": b, "n": len(sub), "y_mean": sub["y"].mean(),
            "pred_mean": sub["pred"].mean(),
            "rmse": float(np.sqrt(sse / len(sub))),
            "sse": float(sse),
        })
    total_sse = sum(r["sse"] for r in rows)
    for r in rows:
        r["pct_sse"] = r["sse"] / total_sse * 100
    return rows


def main():
    print("=" * 70)
    print("Holdout 验证 v8: v7 特征集 + 长尾 sample_weight")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载数据...")
    df_train_raw = load_month_range(list(range(1, 12)))
    df_holdout_raw = load_month_range([12])
    print(f"Train raw: {len(df_train_raw):,}, Holdout raw: {len(df_holdout_raw):,}")

    # 算 features
    print("\n[2] build_features_v2...")
    df_train = build_features_v2(df_train_raw, include_aggregate=True, include_lead=True)
    df_holdout = build_features_v2(df_holdout_raw, include_aggregate=True, include_lead=True)
    print(f"Train: {df_train.shape}, Holdout: {df_holdout.shape}")

    df_tr = df_train.dropna(subset=["TAXITIME_SEC_mvt"]).copy()
    df_tr = df_tr[(df_tr["TAXITIME_SEC_mvt"] > 0) & (df_tr["TAXITIME_SEC_mvt"] < 10800)].copy()
    df_ho = df_holdout.dropna(subset=["TAXITIME_SEC_mvt"]).copy()
    df_ho = df_ho[(df_ho["TAXITIME_SEC_mvt"] > 0) & (df_ho["TAXITIME_SEC_mvt"] < 10800)].copy()
    feat_cols = select_features(df_train)
    print(f"Features: {len(feat_cols)}")
    print(f"Train: {len(df_tr):,}, Holdout: {len(df_ho):,}")

    X_tr = df_tr[feat_cols]
    y_tr = df_tr["TAXITIME_SEC_mvt"]
    X_ho = df_ho[feat_cols]
    y_ho = df_ho["TAXITIME_SEC_mvt"].values

    # 配置
    configs = [
        ("v7_baseline", 0.99, 1.0),  # 不加权(参考 v7)
        ("v8a_q99_w2", 0.99, 2.0),
        ("v8b_q99_w3", 0.99, 3.0),
        ("v8c_q95_w2", 0.95, 2.0),
        ("v8d_q995_w2", 0.995, 2.0),
    ]

    all_results = {}
    print("\n[3] 训练 + 评估各配置...")
    for label, q, w in configs:
        if w == 1.0:
            sw = None
            print(f"\n[{label}] baseline (no weighting)")
        else:
            sw, thresh = make_sample_weight(y_tr, q, w)
            n_tail = (y_tr > thresh).sum()
            print(f"\n[{label}] tail_quantile={q}, tail_weight={w}, "
                  f"threshold={thresh:.0f}s, n_tail={n_tail:,} ({n_tail/len(y_tr)*100:.1f}%)")
        t0 = time.time()
        y_pred, booster = train_with_weights(X_tr, y_tr, sw, X_ho, y_ho, DEFAULT_PARAMS)
        elapsed = time.time() - t0
        r = rmse(y_ho, y_pred)
        m = mae(y_ho, y_pred)
        p = p95_error(y_ho, y_pred)
        print(f"  [{label}] RMSE={r:.2f}, MAE={m:.2f}, P95={p:.2f}, best_iter={booster.best_iteration}, "
              f"time={elapsed:.0f}s")
        all_results[label] = {
            "config": {"tail_quantile": q, "tail_weight": w},
            "rmse": r, "mae": m, "p95_error": p,
            "best_iter": int(booster.best_iteration),
            "by_quantile": evaluate_by_quantile(y_ho, y_pred),
        }

    # 总结
    print("\n" + "=" * 70)
    print("总结(按 RMSE 排序)")
    print("=" * 70)
    print(f"{'config':<22} {'q':>5} {'w':>5} {'RMSE':>8} {'MAE':>8} {'P95':>8}")
    print("-" * 70)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["rmse"])
    for label, r in sorted_results:
        c = r["config"]
        print(f"{label:<22} {c['tail_quantile']:>5.2f} {c['tail_weight']:>5.1f} "
              f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r['p95_error']:>8.2f}")
    best = sorted_results[0]
    print(f"\n最佳: {best[0]} (RMSE={best[1]['rmse']:.2f})")

    # 关键对比:长尾分位段
    print("\n" + "=" * 70)
    print("各配置在长尾分位的 RMSE(看尾部是否真的改善)")
    print("=" * 70)
    print(f"{'config':<20} {'p0-50':>10} {'p95-99':>10} {'p99-99.5':>10} {'p99.9-100':>12}")
    for label, r in sorted_results:
        bins = {b["bin"]: b["rmse"] for b in r["by_quantile"]}
        print(f"{label:<20} {bins['p0-50']:>10.1f} {bins['p95-99']:>10.1f} "
              f"{bins['p99-99.5']:>10.1f} {bins['p99.9-100']:>12.1f}")

    # 保存
    out = {
        "experiment": "holdout_v8_tail_weight",
        "train_period": "2025-01 ~ 2025-11",
        "holdout_period": "2025-12",
        "results": all_results,
    }
    out_path = ROOT / "notebooks" / "holdout_v8_result.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
