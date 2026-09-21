"""
PRC 2026 Holdout Validation: v1 (basic) vs v7 (basic + cnt_dep_*)

训练期:2025-01 ~ 2025-11 (11 个月)
Holdout:2025-12 (12 月,最接近 2026-01 ranking)

关键防泄漏:
- 训练 features 必须只用 2025-01~11 数据计算,不能看 2025-12
- 所以不能用 /tmp/cache_features_v2.parquet(包含 2025-12)
- 必须重新生成 features:训练期和 holdout 期分开算
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
        if not path.exists():
            raise FileNotFoundError(path)
        d = load_month(path, columns=TRAIN_COLS)
        print(f"    {fname}: {len(d):,} 行")
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


def select_features(df, feature_set):
    if feature_set == "basic":
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns]
    elif feature_set == "no_laglead":
        keep_agg = [c for c in df.columns if c.endswith("m") and "cnt_" in c]
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns] + keep_agg
    raise ValueError(feature_set)


def train_and_predict(df_train_feat, df_holdout_feat, feature_set, params, label):
    feat_cols = select_features(df_train_feat, feature_set)
    print(f"\n[{label}] 特征数: {len(feat_cols)}")
    df_tr = df_train_feat.dropna(subset=["TAXITIME_SEC_mvt"]).copy()
    df_tr = df_tr[(df_tr["TAXITIME_SEC_mvt"] > 0) & (df_tr["TAXITIME_SEC_mvt"] < 10800)].copy()
    df_ho = df_holdout_feat.dropna(subset=["TAXITIME_SEC_mvt"]).copy()
    df_ho = df_ho[(df_ho["TAXITIME_SEC_mvt"] > 0) & (df_ho["TAXITIME_SEC_mvt"] < 10800)].copy()
    print(f"  Train: {len(df_tr):,} 行, Holdout: {len(df_ho):,} 行")

    X_tr = df_tr[feat_cols]
    y_tr = df_tr["TAXITIME_SEC_mvt"]
    X_ho = df_ho[feat_cols]
    y_ho = df_ho["TAXITIME_SEC_mvt"].values

    X_tr_enc, X_ho_enc = encode_categoricals(X_tr, X_ho)
    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
    dhold = xgb.DMatrix(X_ho_enc, label=y_ho)

    print(f"  训练中(num_boost_round=2000, early_stopping=100)...")
    t0 = time.time()
    booster = xgb.train(
        params, dtrain, num_boost_round=2000,
        evals=[(dtrain, "train"), (dhold, "holdout")],
        early_stopping_rounds=100, verbose_eval=200,
    )
    print(f"  训练耗时: {time.time() - t0:.1f}s, best_iter={booster.best_iteration}")

    y_pred = booster.predict(dhold)
    r = rmse(y_ho, y_pred); m = mae(y_ho, y_pred); p = p95_error(y_ho, y_pred)
    print(f"  [{label}] Holdout RMSE={r:.2f}, MAE={m:.2f}, P95={p:.2f}")
    return r, m, p, booster, feat_cols


def main():
    print("=" * 70)
    print("Holdout Validation: v1 (basic) vs v7 (basic + cnt_dep_*)")
    print("=" * 70)

    print("\n[1] 加载数据...")
    df_train_raw = load_month_range(list(range(1, 12)))
    print(f"Train raw: {len(df_train_raw):,} 行")
    df_holdout_raw = load_month_range([12])
    print(f"Holdout raw: {len(df_holdout_raw):,} 行")

    eobt_train_nat = df_train_raw["EOBT_1_flt"].isna().mean()
    eobt_holdout_nat = df_holdout_raw["EOBT_1_flt"].isna().mean()
    print(f"\n[NaT 检查] EOBT_1_flt NaT 率: train={eobt_train_nat*100:.2f}%, "
          f"holdout={eobt_holdout_nat*100:.2f}%")
    print(f"  (ranking 上是 1.53%,理想情况:train/holdout 都接近这个数)")

    print("\n[2] 计算 features(分开算,防泄漏)...")
    print("  Train v2 (basic + aggregate + lead)...")
    t0 = time.time()
    df_train_feat = build_features_v2(df_train_raw, include_aggregate=True, include_lead=True)
    print(f"  Train v2 耗时: {time.time() - t0:.1f}s, shape={df_train_feat.shape}")

    print("  Train basic only...")
    t0 = time.time()
    df_train_basic = build_basic(df_train_raw)
    print(f"  Train basic 耗时: {time.time() - t0:.1f}s, shape={df_train_basic.shape}")

    print("  Holdout v2...")
    t0 = time.time()
    df_holdout_feat = build_features_v2(df_holdout_raw, include_aggregate=True, include_lead=True)
    print(f"  Holdout v2 耗时: {time.time() - t0:.1f}s, shape={df_holdout_feat.shape}")

    print("  Holdout basic only...")
    t0 = time.time()
    df_holdout_basic = build_basic(df_holdout_raw)
    print(f"  Holdout basic 耗时: {time.time() - t0:.1f}s, shape={df_holdout_basic.shape}")

    print("\n[3] 训练 + holdout 评估...")
    rmse_v1, mae_v1, p95_v1, booster_v1, feat_v1 = train_and_predict(
        df_train_basic, df_holdout_basic,
        feature_set="basic", params=DEFAULT_PARAMS, label="v1 (basic only)"
    )
    rmse_v7, mae_v7, p95_v7, booster_v7, feat_v7 = train_and_predict(
        df_train_feat, df_holdout_feat,
        feature_set="no_laglead", params=DEFAULT_PARAMS, label="v7 (basic + cnt_dep_*)"
    )

    print("\n[4] v7 Top 10 特征重要性 (gain)")
    imp_v7 = booster_v7.get_score(importance_type="gain")
    for k, v in sorted(imp_v7.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k:<35} {v:.2f}")

    print("\n[5] cnt_dep_* 分布对比(train 2025-01~11 vs holdout 2025-12)")
    cnt_cols = [c for c in feat_v7 if c.startswith("cnt_dep_")]
    print(f"cnt_dep_* 共 {len(cnt_cols)} 列")
    print(f"\n{'特征':<35} {'train mean':>12} {'ho mean':>12} "
          f"{'train NaN%':>11} {'ho NaN%':>10}")
    print("-" * 84)
    dist_rows = []
    for col in cnt_cols:
        if col in df_train_feat.columns and col in df_holdout_feat.columns:
            tr_m = df_train_feat[col].mean()
            ho_m = df_holdout_feat[col].mean()
            tr_n = df_train_feat[col].isna().mean() * 100
            ho_n = df_holdout_feat[col].isna().mean() * 100
            print(f"{col:<35} {tr_m:>12.3f} {ho_m:>12.3f} {tr_n:>10.2f}% {ho_n:>9.2f}%")
            dist_rows.append({
                "feature": col, "train_mean": float(tr_m), "holdout_mean": float(ho_m),
                "train_nan_pct": float(tr_n), "holdout_nan_pct": float(ho_n),
                "mean_shift_pct": float((ho_m - tr_m) / max(abs(tr_m), 1e-6) * 100),
            })

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"v1 (basic, {len(feat_v1)} features):      "
          f"RMSE = {rmse_v1:.2f}, MAE = {mae_v1:.2f}, P95 = {p95_v1:.2f}")
    print(f"v7 (basic+cnt, {len(feat_v7)} features):   "
          f"RMSE = {rmse_v7:.2f}, MAE = {mae_v7:.2f}, P95 = {p95_v7:.2f}")
    delta = rmse_v7 - rmse_v1
    pct = delta / rmse_v1 * 100
    winner = "v7" if rmse_v7 < rmse_v1 else "v1"
    print(f"\n差异: {delta:+.2f} ({pct:+.2f}%)")
    print(f"胜者: {winner}")

    out = {
        "experiment": "holdout_2025-12",
        "train_period": "2025-01 ~ 2025-11",
        "holdout_period": "2025-12",
        "v1_basic": {"n_features": len(feat_v1), "features": feat_v1,
                     "rmse": rmse_v1, "mae": mae_v1, "p95_error": p95_v1},
        "v7_no_laglead": {"n_features": len(feat_v7), "features": feat_v7,
                          "rmse": rmse_v7, "mae": mae_v7, "p95_error": p95_v7},
        "rmse_diff": delta, "rmse_diff_pct": pct, "winner": winner,
        "eobt_nat_pct": {
            "train": float(eobt_train_nat * 100),
            "holdout": float(eobt_holdout_nat * 100),
            "ranking_target": 1.53,
        },
        "cnt_dep_distribution": dist_rows,
    }
    out_path = ROOT / "notebooks" / "holdout_validation_result.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
