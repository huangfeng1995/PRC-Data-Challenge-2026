"""
实验 A:同月 holdout 诊断 v1 vs v7
====================================

关键:用 2025-01 + 2025-07 做 holdout(与 ranking 2026-01+07 同月)
- 训练:2025-02 ~ 2025-06, 2025-08 ~ 2025-12 (10 个月)
- Holdout:2025-01 + 2025-07 (2 个月, 与 ranking 同月)

对比 v1 (basic) vs v7 (basic + cnt_dep_*) 在这个"同月" holdout 上的表现:
- 如果 v1 holdout RMSE ≈ 590 → 时间分布漂移是主因,后续方向 = 分布适配
- 如果 v1 holdout RMSE ≈ 318 → ranking 数据机制异常,需要深挖 ranking

预期:
- 12月 holdout: v1=319, v7=311
- 同月 holdout: v1 应该 ≈ 270-310(同期分布更接近),v7 应该 ≈ 260-300
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


def select_features(df, feature_set):
    if feature_set == "basic":
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns]
    elif feature_set == "no_laglead":
        keep_agg = [c for c in df.columns if c.endswith("m") and "cnt_" in c]
        return [c for c in BASELINE_FEATURE_COLS if c in df.columns] + keep_agg
    raise ValueError(feature_set)


def train_and_evaluate(X_tr, y_tr, X_ho, y_ho, params, label):
    X_tr_enc, X_ho_enc = encode_categoricals(X_tr, X_ho)
    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
    dhold = xgb.DMatrix(X_ho_enc, label=y_ho)
    print(f"\n[{label}] 训练中...")
    t0 = time.time()
    booster = xgb.train(
        params, dtrain, num_boost_round=2000,
        evals=[(dtrain, "train"), (dhold, "holdout")],
        early_stopping_rounds=100, verbose_eval=False,
    )
    y_pred = booster.predict(dhold)
    r = rmse(y_ho, y_pred); m = mae(y_ho, y_pred); p = p95_error(y_ho, y_pred)
    print(f"  [{label}] RMSE={r:.2f}, MAE={m:.2f}, P95={p:.2f}, "
          f"best_iter={booster.best_iteration}, time={time.time()-t0:.0f}s")
    return r, m, p, y_pred, booster


def split_per_month_rmse(y_ho, y_pred, months):
    """按真实月份分 RMSE(看 holdout 1 月和 7 月分别多少)"""
    rows = []
    for m in months:
        mask = pd.to_datetime(y_ho.index if hasattr(y_ho, 'index') else None).month == m if False else None
        # 简化:用 SCHED_TIME 来分月份
    return rows


def main():
    print("=" * 70)
    print("实验 A: 同月 holdout 诊断(2025-01 + 2025-07)")
    print("=" * 70)

    print("\n[1] 加载数据...")
    # 训练:2-6月 + 8-12月 (10 个月)
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    # Holdout:1月 + 7月 (与 ranking 同月)
    holdout_months = [1, 7]

    df_train_raw = load_month_range(train_months)
    df_holdout_raw = load_month_range(holdout_months)
    print(f"Train raw: {len(df_train_raw):,}")
    print(f"Holdout raw: {len(df_holdout_raw):,}")

    # EOBT NaT 一致性
    eobt_tr = df_train_raw["EOBT_1_flt"].isna().mean()
    eobt_ho = df_holdout_raw["EOBT_1_flt"].isna().mean()
    print(f"\nEOBT_1_flt NaT 率: train={eobt_tr*100:.2f}%, holdout={eobt_ho*100:.2f}%")
    print(f"  (ranking 上是 1.53%)")

    print("\n[2] build_features_v2...")
    print("  Train v2...")
    t0 = time.time()
    df_train = build_features_v2(df_train_raw, include_aggregate=True, include_lead=True)
    print(f"  Train v2 耗时: {time.time()-t0:.1f}s, shape={df_train.shape}")
    print("  Train basic...")
    t0 = time.time()
    df_train_basic = build_basic(df_train_raw)
    print(f"  Train basic 耗时: {time.time()-t0:.1f}s, shape={df_train_basic.shape}")

    print("  Holdout v2...")
    t0 = time.time()
    df_holdout = build_features_v2(df_holdout_raw, include_aggregate=True, include_lead=True)
    print(f"  Holdout v2 耗时: {time.time()-t0:.1f}s, shape={df_holdout.shape}")
    print("  Holdout basic...")
    t0 = time.time()
    df_holdout_basic = build_basic(df_holdout_raw)
    print(f"  Holdout basic 耗时: {time.time()-t0:.1f}s, shape={df_holdout_basic.shape}")

    # 过滤
    df_tr_basic = df_train_basic.dropna(subset=["TAXITIME_SEC_mvt"]).copy()
    df_tr_basic = df_tr_basic[(df_tr_basic["TAXITIME_SEC_mvt"] > 0) & (df_tr_basic["TAXITIME_SEC_mvt"] < 10800)]
    df_tr = df_train.dropna(subset=["TAXITIME_SEC_mvt"]).copy()
    df_tr = df_tr[(df_tr["TAXITIME_SEC_mvt"] > 0) & (df_tr["TAXITIME_SEC_mvt"] < 10800)]
    df_ho_basic = df_holdout_basic.dropna(subset=["TAXITIME_SEC_mvt"]).copy()
    df_ho_basic = df_ho_basic[(df_ho_basic["TAXITIME_SEC_mvt"] > 0) & (df_ho_basic["TAXITIME_SEC_mvt"] < 10800)]
    df_ho = df_holdout.dropna(subset=["TAXITIME_SEC_mvt"]).copy()
    df_ho = df_ho[(df_ho["TAXITIME_SEC_mvt"] > 0) & (df_ho["TAXITIME_SEC_mvt"] < 10800)]
    print(f"\nTrain: {len(df_tr):,}, Holdout: {len(df_ho):,}")

    # 训练 v1 (basic)
    feat_v1 = select_features(df_train_basic, "basic")
    print(f"\n[v1 (basic)] 特征数: {len(feat_v1)}")
    X_tr_v1 = df_tr_basic[feat_v1]
    X_ho_v1 = df_ho_basic[feat_v1]
    rmse_v1, mae_v1, p95_v1, y_pred_v1, booster_v1 = train_and_evaluate(
        X_tr_v1, df_tr_basic["TAXITIME_SEC_mvt"],
        X_ho_v1, df_ho_basic["TAXITIME_SEC_mvt"].values,
        DEFAULT_PARAMS, "v1 (basic only)"
    )

    # 训练 v7 (basic + cnt_dep_*)
    feat_v7 = select_features(df_train, "no_laglead")
    print(f"\n[v7 (basic + cnt_dep_*)] 特征数: {len(feat_v7)}")
    X_tr_v7 = df_tr[feat_v7]
    X_ho_v7 = df_ho[feat_v7]
    rmse_v7, mae_v7, p95_v7, y_pred_v7, booster_v7 = train_and_evaluate(
        X_tr_v7, df_tr["TAXITIME_SEC_mvt"],
        X_ho_v7, df_ho["TAXITIME_SEC_mvt"].values,
        DEFAULT_PARAMS, "v7 (basic + cnt_dep_*)"
    )

    # 按月份分解 holdout RMSE(看 1月 vs 7月 分别多少)
    sched_ho = df_ho["SCHED_TIME_UTC_mvt"].reset_index(drop=True)
    y_ho_arr = df_ho["TAXITIME_SEC_mvt"].values
    print("\n" + "=" * 70)
    print("按月份分解 holdout RMSE(看 1 月和 7 月分别多少)")
    print("=" * 70)
    for m in [1, 7]:
        mask = sched_ho.dt.month == m
        if mask.sum() == 0:
            continue
        v1_rmse_m = np.sqrt(((y_pred_v1[mask] - y_ho_arr[mask]) ** 2).mean())
        v7_rmse_m = np.sqrt(((y_pred_v7[mask] - y_ho_arr[mask]) ** 2).mean())
        print(f"  {m}月 (n={mask.sum():,}):")
        print(f"    v1 RMSE: {v1_rmse_m:.2f}")
        print(f"    v7 RMSE: {v7_rmse_m:.2f}")
        print(f"    v7 - v1: {v7_rmse_m - v1_rmse_m:+.2f}")

    # 按机场分解
    print("\n按机场分解:")
    for ap in df_ho["ADEP_mvt"].value_counts().head(5).index:
        mask = df_ho["ADEP_mvt"].values == ap
        v1_rmse_ap = np.sqrt(((y_pred_v1[mask] - y_ho_arr[mask]) ** 2).mean())
        v7_rmse_ap = np.sqrt(((y_pred_v7[mask] - y_ho_arr[mask]) ** 2).mean())
        print(f"  {ap} (n={mask.sum():,}): v1={v1_rmse_ap:.0f}, v7={v7_rmse_ap:.0f}, Δ={v7_rmse_ap-v1_rmse_ap:+.1f}")

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"v1 (basic, 30 features):      RMSE = {rmse_v1:.2f}, MAE = {mae_v1:.2f}, P95 = {p95_v1:.2f}")
    print(f"v7 (basic+cnt, 42 features):   RMSE = {rmse_v7:.2f}, MAE = {mae_v7:.2f}, P95 = {p95_v7:.2f}")
    print(f"差异: {rmse_v7 - rmse_v1:+.2f} ({(rmse_v7-rmse_v1)/rmse_v1*100:+.2f}%)")

    # 与 12 月 holdout 对比
    print("\n" + "=" * 70)
    print("与 12 月 holdout 对比(说明同月 vs 错月)")
    print("=" * 70)
    print(f"{'版本':<22} {'12月 holdout':>15} {'1月+7月 holdout':>18}")
    print(f"{'v1':<22} {319.40:>15.2f} {rmse_v1:>18.2f}")
    print(f"{'v7':<22} {311.48:>15.2f} {rmse_v7:>18.2f}")

    # 诊断结论
    print("\n" + "=" * 70)
    print("诊断结论")
    print("=" * 70)
    if rmse_v1 > 400:
        print(f"⚠️  v1 同月 holdout RMSE = {rmse_v1:.2f}, 仍然很大")
        print("   → 即使同月,LB-style RMSE 也很高,说明 ranking 分布有额外差异")
        print("   → 需要查 ranking parquet 本身的特征分布")
    elif rmse_v1 > 350:
        print(f"⚠️  v1 同月 holdout RMSE = {rmse_v1:.2f}, 介于 12月和 LB 之间")
        print("   → 部分差异来自月份漂移,部分来自其他因素")
    else:
        print(f"✅  v1 同月 holdout RMSE = {rmse_v1:.2f}, 接近验证集 318")
        print("   → 排名数据机制异常,需要查 ranking 本身的特征分布")

    # 保存
    out = {
        "experiment": "same_month_holdout",
        "train_period": "2025-02~06 + 2025-08~12",
        "holdout_period": "2025-01 + 2025-07",
        "v1": {"rmse": rmse_v1, "mae": mae_v1, "p95_error": p95_v1, "n_features": len(feat_v1)},
        "v7": {"rmse": rmse_v7, "mae": mae_v7, "p95_error": p95_v7, "n_features": len(feat_v7)},
        "diff": {"rmse_diff": rmse_v7 - rmse_v1, "pct": (rmse_v7-rmse_v1)/rmse_v1*100},
        "vs_dec_holdout": {
            "v1_dec": 319.40, "v1_jan_jul": rmse_v1,
            "v7_dec": 311.48, "v7_jan_jul": rmse_v7,
        }
    }
    out_path = ROOT / "notebooks" / "holdout_same_month_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
