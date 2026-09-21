"""
清洗 A/B runner: 严格沿用 _gate.py 的入口,只切训练 target 清洗方式
- C0: 不清洗(基线;与 gate 同)
- C1: 处理 < 60s(全局)
- C2: 每机场 [P0.1, P99.9] winsor
- C3: 每机场 [P0.5, P99.5] winsor
所有阈值仅在训练集上拟合,评估用原始 holdout target
"""
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

# 冻结入口
spec = importlib.util.spec_from_file_location(
    "ab_v2_holdout_v1_queue_ab",
    ROOT / "notebooks/diag/holdout_v1_queue_ab.py",
)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

OUT_DIR = ROOT / "notebooks/diag" / "ab_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """照搬 holdout_v1_queue_ab.main() 数据加载链(只到 G0 训练所需)"""
    # 1) holdout
    dfs = []
    for m in [1, 7]:
        fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        dfs.append(ab.load_month(f"data/raw/{fname}", columns=ab.TRAIN_COLS))
    df_ho_raw = pd.concat(dfs, ignore_index=True)
    df_ho_all_raw = ab.parse_time_columns(df_ho_raw, cols=ab.TIME_COLS)

    # 2) holdout 全套特征
    df_ho_full = ab.build_full_features(df_ho_all_raw.copy())

    # 3) 训练
    months_to_load = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    train_dfs_arr = []
    for m in months_to_load:
        if m < 12:
            fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        else:
            fname = "training_2025-12-01_2026-01-01.parquet"
        df_m = ab.load_month(f"data/raw/{fname}", columns=ab.TRAIN_COLS)
        train_dfs_arr.append(df_m)
    df_train_all = pd.concat(train_dfs_arr, ignore_index=True)
    df_train_all = ab.parse_time_columns(df_train_all, cols=ab.TIME_COLS)

    # 4) 训练全套特征
    df_train_full = ab.build_full_features(df_train_all.copy())

    return df_train_full, df_ho_full


def fit_clean_threshold_train(y_train: np.ndarray, ap_train: np.ndarray, method: str):
    """仅在训练集上拟合清洗阈值,返回 dict;
    method in {"none", "drop_lt60", "winsor_per_ap_0.1_99.9", "winsor_per_ap_0.5_99.5"}
    """
    if method == "none":
        return {"method": "none"}
    if method == "drop_lt60":
        return {"method": "drop_lt60", "threshold_s": 60.0}
    if method == "winsor_per_ap_0.1_99.9":
        thr = {}
        for ap, idx in pd.Series(ap_train).groupby(ap_train).groups.items():
            sub = y_train[idx]
            thr[str(ap)] = (float(np.percentile(sub, 0.1)), float(np.percentile(sub, 99.9)))
        return {"method": "winsor_per_ap", "bounds": thr, "lo_q": 0.1, "hi_q": 99.9}
    if method == "winsor_per_ap_0.5_99.5":
        thr = {}
        for ap in np.unique(ap_train):
            idx = ap_train == ap
            sub = y_train[idx]
            thr[str(ap)] = (float(np.percentile(sub, 0.5)), float(np.percentile(sub, 99.5)))
        return {"method": "winsor_per_ap", "bounds": thr, "lo_q": 0.5, "hi_q": 99.5}
    raise ValueError(method)


def apply_train_filter(y_train: np.ndarray, ap_train: np.ndarray, thr: dict):
    """应用训练集清洗:返回保留 mask"""
    if thr["method"] == "none":
        return np.ones(len(y_train), dtype=bool)
    if thr["method"] == "drop_lt60":
        return y_train >= thr["threshold_s"]
    if thr["method"] == "winsor_per_ap":
        m = np.ones(len(y_train), dtype=bool)
        for ap, (lo, hi) in thr["bounds"].items():
            idx = ap_train == ap
            m[idx] = (y_train[idx] >= lo) & (y_train[idx] <= hi)
        return m
    raise ValueError(thr)


def run_clean_variant(method: str):
    """跑一种清洗变体,返回 metrics dict"""
    print(f"\n{'=' * 70}", flush=True)
    print(f"[清洗 A/B] method = {method}", flush=True)
    print(f"{'=' * 70}", flush=True)

    df_train_full, df_ho_full = load_data()

    # === 训练集 target 过滤(与入口一致)===
    y_tr_full = df_train_full["TAXITIME_SEC_mvt"].values
    ap_tr = df_train_full["ADEP_mvt"].astype(str).values
    mask_valid = (y_tr_full > 0) & (y_tr_full < 10800)
    df_train_full = df_train_full[mask_valid].reset_index(drop=True)
    y_tr_full = df_train_full["TAXITIME_SEC_mvt"].values
    ap_tr_full = df_train_full["ADEP_mvt"].astype(str).values
    n_before = len(y_tr_full)

    # === 仅在训练集上拟合清洗阈值 ===
    thr = fit_clean_threshold_train(y_tr_full, ap_tr_full, method)
    keep_mask = apply_train_filter(y_tr_full, ap_tr_full, thr)
    df_train_full = df_train_full[keep_mask].reset_index(drop=True)
    y_tr_full = df_train_full["TAXITIME_SEC_mvt"].values
    n_after = len(y_tr_full)
    print(f"  训练样本: {n_before:,} -> {n_after:,} (保留 {n_after / n_before * 100:.2f}%)", flush=True)
    print(f"  清洗方法: {thr}", flush=True)

    # === holdout 取 DEP + 原始 target 过滤(评估不洗) ===
    if "PHASE_mvt" in df_ho_full.columns:
        ho_dep_mask = df_ho_full["PHASE_mvt"].astype(str).values == "DEP"
        df_ho_dep_full = df_ho_full[ho_dep_mask].reset_index(drop=True)
    else:
        df_ho_dep_full = df_ho_full
    df_ho_dep_full = df_ho_dep_full[
        (df_ho_dep_full["TAXITIME_SEC_mvt"] > 0)
        & (df_ho_dep_full["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)

    # === G0 feat ===
    G0 = ab.GROUPS["G0_basic30"]
    feat_cols = ab.get_feat_cols_for_group(df_train_full, G0)
    for c in feat_cols:
        if c not in df_train_full.columns:
            df_train_full[c] = np.nan
        if c not in df_ho_dep_full.columns:
            df_ho_dep_full[c] = np.nan
    X_tr = df_train_full[feat_cols]
    for c in feat_cols:
        if c not in df_ho_dep_full.columns:
            df_ho_dep_full[c] = np.nan
    X_ho = df_ho_dep_full[feat_cols]
    y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values

    # === cat fit + train ===
    group_name = "G0_basic30"
    cat_mappings_path = (
        ROOT / "models" / "feature_lists" / f"xgb_v9_{group_name}_seed0_val_cat_mappings.json"
    )
    if cat_mappings_path.exists():
        cat_mappings_path.unlink()
    import src.train_xgb as tx

    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS_PATH = cat_mappings_path
    X_tr_enc = tx._encode_categoricals(X_tr, fit=True)

    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr_full)
    params = {**ab.DEFAULT_PARAMS, "seed": 0}
    n_rounds = 1200
    booster = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)

    tx._CAT_MAPPINGS_PATH = cat_mappings_path
    ho_metrics, ho_pred = ab.evaluate(booster, X_ho, y_ho, df_ho_dep_full, feat_cols)

    rmse = ho_metrics["rmse"]
    ap_rmse = {ap: v["rmse"] for ap, v in ho_metrics["by_airport"].items()}
    print(
        f"  现训结果: RMSE={rmse:.4f}, bias={ho_metrics['bias']:+.4f}s ({ho_metrics['bias_pct']:+.2f}%)",
        flush=True,
    )
    print(
        f"  按机场 RMSE: LIRF={ap_rmse.get('LIRF', 0):.2f}, "
        f"LFPG={ap_rmse.get('LFPG', 0):.2f}, EGLL={ap_rmse.get('EGLL', 0):.2f}",
        flush=True,
    )
    return {
        "method": method,
        "n_train_before": n_before,
        "n_train_after": n_after,
        "rmse": rmse,
        "mae": ho_metrics["mae"],
        "bias": ho_metrics["bias"],
        "bias_pct": ho_metrics["bias_pct"],
        "ap_rmse": ap_rmse,
        "thr": thr,
    }


def main():
    methods = [
        "none",  # C0 baseline(= gate 357.14)
        "drop_lt60",  # C1
        "winsor_per_ap_0.1_99.9",  # C2
        "winsor_per_ap_0.5_99.5",  # C3
    ]
    results = []
    for m in methods:
        results.append(run_clean_variant(m))
    out_path = OUT_DIR / "clean_ab_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n✅ 清洗 A/B 结果: {out_path}", flush=True)
    print("\n=== 汇总 ===", flush=True)
    base = next(r for r in results if r["method"] == "none")
    print(f"  baseline RMSE = {base['rmse']:.4f}", flush=True)
    for r in results:
        delta = r["rmse"] - base["rmse"]
        print(
            f"  {r['method']:30s}  RMSE={r['rmse']:.4f}  Δ={delta:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()