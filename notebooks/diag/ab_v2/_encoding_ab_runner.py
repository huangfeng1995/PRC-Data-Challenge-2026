"""
编码 A/B runner: 严格沿用 _gate.py 的入口,只切编码方式
- H0: numeric label-encode(= gate baseline)
- H1: XGB enable_categorical(category dtype)
- H2: CatBoost 原生 cat_features(iterations=400, early_stopping_rounds=50, thread_count=-1)
       先用 10% 数据确认单次耗时在可接受范围,不要再开 1200 轮跑 2 小时
- H3: OOF smoothed target encoding(严格时间折/训练折内拟合,禁用 holdout/ranking 统计)

baseline = gate 357.14
"""
import sys
import json
import importlib.util
import gc
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


def load_data_cached():
    """只跑一次:加载 holdout + 训练全套特征"""
    print("\n[load] holdout + 训练全套特征...", flush=True)
    # holdout
    dfs = []
    for m in [1, 7]:
        fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        dfs.append(ab.load_month(f"data/raw/{fname}", columns=ab.TRAIN_COLS))
    df_ho_raw = pd.concat(dfs, ignore_index=True)
    df_ho_all_raw = ab.parse_time_columns(df_ho_raw, cols=ab.TIME_COLS)
    df_ho_full = ab.build_full_features(df_ho_all_raw.copy())

    # 训练
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
    df_train_full = ab.build_full_features(df_train_all.copy())

    # 训练 target 过滤
    y_tr = df_train_full["TAXITIME_SEC_mvt"].values
    mask = (y_tr > 0) & (y_tr < 10800)
    df_train_full = df_train_full[mask].reset_index(drop=True)

    # holdout DEP + target 过滤
    if "PHASE_mvt" in df_ho_full.columns:
        ho_dep_mask = df_ho_full["PHASE_mvt"].astype(str).values == "DEP"
        df_ho_dep_full = df_ho_full[ho_dep_mask].reset_index(drop=True)
    else:
        df_ho_dep_full = df_ho_full
    df_ho_dep_full = df_ho_dep_full[
        (df_ho_dep_full["TAXITIME_SEC_mvt"] > 0)
        & (df_ho_dep_full["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)

    return df_train_full, df_ho_dep_full


def make_holdout_for_eval(df_ho_dep_full, feat_cols, enc_kind, te_maps=None):
    """返回编码后的 holdout X"""
    X_ho = df_ho_dep_full[feat_cols].copy()
    y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values
    if enc_kind == "numeric":
        import src.train_xgb as tx
        cat_mappings_path = (
            ROOT
            / "models"
            / "feature_lists"
            / "xgb_v9_G0_basic30_seed0_val_cat_mappings.json"
        )
        tx._CAT_MAPPINGS.clear()
        tx._CAT_MAPPINGS_PATH = cat_mappings_path
        X_ho_enc = tx._encode_categoricals(X_ho, fit=False)
        return X_ho_enc, y_ho
    if enc_kind == "xgb_cat":
        return encode_xgb_cat(X_ho), y_ho
    if enc_kind == "te":
        return encode_te(X_ho, te_maps), y_ho
    raise ValueError(enc_kind)


def encode_xgb_cat(X: pd.DataFrame) -> pd.DataFrame:
    """把 object/category 列转成 category dtype,供 XGB enable_categorical=True"""
    X2 = X.copy()
    for c in X2.columns:
        if X2[c].dtype == object or pd.api.types.is_categorical_dtype(X2[c]):
            X2[c] = X2[c].astype("category")
        elif X2[c].dtype.name == "string":
            X2[c] = X2[c].astype("category")
    return X2


def encode_te(X: pd.DataFrame, te_maps: dict) -> pd.DataFrame:
    """te_maps: {col: {value: smoothed_target}}。unseen 走全局均值。"""
    X2 = X.copy()
    for c, m in te_maps.items():
        if c not in X2.columns:
            continue
        global_mean = m.get("__global_mean__", 0.0)
        X2[c] = X2[c].astype(str).map(lambda v, _m=m, _g=global_mean: _m.get(v, _g))
    return X2


def fit_oof_target_encoding(df_train: pd.DataFrame, y: np.ndarray, cat_cols, n_splits=5, seed=0, m_smooth=20):
    """OOF smoothed target encoding:按时间顺序 5 折,每折用其他折训练 fit,本折 transform"""
    n = len(df_train)
    # 按时间排序切分(用 MVT_TIME_UTC_mvt 排序)
    if "MVT_TIME_UTC_mvt" in df_train.columns:
        order = np.argsort(df_train["MVT_TIME_UTC_mvt"].values)
    else:
        order = np.arange(n)
    fold_id = np.zeros(n, dtype=int)
    chunk = n // n_splits
    for k in range(n_splits):
        lo = k * chunk
        hi = (k + 1) * chunk if k < n_splits - 1 else n
        fold_id[order[lo:hi]] = k

    maps_per_col = {c: {} for c in cat_cols}
    global_mean = float(np.mean(y))

    # 对每折:用非本折训练 fit,本折 transform
    for k in range(n_splits):
        train_mask = fold_id != k
        valid_mask = fold_id == k
        for c in cat_cols:
            vals_tr = df_train.loc[train_mask, c].astype(str).values
            y_tr = y[train_mask]
            vals_va = df_train.loc[valid_mask, c].astype(str).values
            # group stats
            df_sub = pd.DataFrame({"v": vals_tr, "y": y_tr})
            stats = df_sub.groupby("v")["y"].agg(["mean", "count"])
            # smoothed: (n * mean + m * global_mean) / (n + m)
            smoothed = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
                stats["count"] + m_smooth
            )
            m_local = smoothed.to_dict()
            maps_per_col[c].update(m_local)
    # 补 global_mean
    for c in cat_cols:
        maps_per_col[c]["__global_mean__"] = global_mean
    return maps_per_col


def run_variant(df_train_full, df_ho_dep_full, variant: str):
    """variant in {H0, H1, H2, H3}"""
    print(f"\n{'=' * 70}", flush=True)
    print(f"[编码 A/B] variant = {variant}", flush=True)
    print(f"{'=' * 70}", flush=True)

    G0 = ab.GROUPS["G0_basic30"]
    feat_cols = ab.get_feat_cols_for_group(df_train_full, G0)
    print(f"  feat_cols n={len(feat_cols)}", flush=True)

    X_tr_raw = df_train_full[feat_cols].copy()
    X_ho_raw = df_ho_dep_full[feat_cols].copy()
    y_tr = df_train_full["TAXITIME_SEC_mvt"].values
    y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values

    cat_cols = [c for c in feat_cols if X_tr_raw[c].dtype == object or str(X_tr_raw[c].dtype) == "category"]
    print(f"  cat cols ({len(cat_cols)}): {cat_cols}", flush=True)

    booster = None
    enc_kind_train = None
    te_maps = None

    if variant == "H0":
        # numeric label-encode(同入口)
        import src.train_xgb as tx
        cat_mappings_path = (
            ROOT / "models" / "feature_lists" / "xgb_v9_G0_basic30_seed0_val_cat_mappings.json"
        )
        if cat_mappings_path.exists():
            cat_mappings_path.unlink()
        tx._CAT_MAPPINGS.clear()
        tx._CAT_MAPPINGS_PATH = cat_mappings_path
        X_tr_enc = tx._encode_categoricals(X_tr_raw, fit=True)
        dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
        params = {**ab.DEFAULT_PARAMS, "seed": 0, "enable_categorical": False}
        booster = xgb.train(params, dtrain, num_boost_round=1200, verbose_eval=False)
        enc_kind_train = "numeric"

    elif variant == "H1":
        # XGB enable_categorical(category dtype)
        X_tr_enc = encode_xgb_cat(X_tr_raw)
        dtrain = xgb.DMatrix(X_tr_enc, label=y_tr, enable_categorical=True)
        params = {**ab.DEFAULT_PARAMS, "seed": 0, "enable_categorical": True}
        booster = xgb.train(params, dtrain, num_boost_round=1200, verbose_eval=False)
        enc_kind_train = "xgb_cat"

    elif variant == "H2":
        # CatBoost:iterations=400, early_stopping_rounds=50, thread_count=-1
        from catboost import CatBoostRegressor, Pool

        X_tr_enc = X_tr_raw.copy()
        cat_features_idx = [X_tr_enc.columns.get_loc(c) for c in cat_cols]
        # CatBoost 要求 cat 列是 str/int, 缺失填 "__NAN__"
        for c in cat_cols:
            X_tr_enc[c] = X_tr_enc[c].astype(str).fillna("__NAN__")
        # 10% 数据先做耗时测试(用户要求)
        n_test = max(int(len(X_tr_enc) * 0.1), 100000)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_tr_enc), size=n_test, replace=False)
        X_t = X_tr_enc.iloc[idx]
        y_t = y_tr[idx]
        pool_t = Pool(X_t, y_t, cat_features=cat_features_idx)
        import time
        t0 = time.time()
        m_test = CatBoostRegressor(
            iterations=400, early_stopping_rounds=50, thread_count=-1,
            random_seed=0, verbose=False, loss_function="RMSE",
        )
        m_test.fit(pool_t)
        t10pct = time.time() - t0
        print(f"  [H2 耗时测试] 10%={n_test:,}行, iter=400 耗时 {t10pct:.1f}s", flush=True)
        # 阈值:如果 10%>1800s(30min),终止不再跑
        if t10pct > 1800:
            print(f"  [H2 终止] 单次耗时 {t10pct:.1f}s 过大,放弃 H2 全量训练", flush=True)
            return {"variant": variant, "rmse": None, "skipped": "too_slow", "t10pct": t10pct}

        # 全量训练
        pool_full = Pool(X_tr_enc, y_tr, cat_features=cat_features_idx)
        model = CatBoostRegressor(
            iterations=400, early_stopping_rounds=50, thread_count=-1,
            random_seed=0, verbose=200, loss_function="RMSE",
        )
        model.fit(pool_full)
        booster = model
        enc_kind_train = "catboost"

    elif variant == "H3":
        # OOF smoothed target encoding
        te_maps = fit_oof_target_encoding(df_train_full, y_tr, cat_cols, n_splits=5, seed=0)
        X_tr_enc = encode_te(X_tr_raw, te_maps)
        dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
        params = {**ab.DEFAULT_PARAMS, "seed": 0, "enable_categorical": False}
        booster = xgb.train(params, dtrain, num_boost_round=1200, verbose_eval=False)
        enc_kind_train = "te"

    else:
        raise ValueError(variant)

    # 评估 holdout
    if enc_kind_train == "catboost":
        X_ho_enc = X_ho_raw.copy()
        for c in cat_cols:
            X_ho_enc[c] = X_ho_enc[c].astype(str).fillna("__NAN__")
        y_pred = booster.predict(X_ho_enc)
    elif enc_kind_train == "xgb_cat":
        X_ho_enc = encode_xgb_cat(X_ho_raw)
        y_pred = booster.predict(xgb.DMatrix(X_ho_enc, enable_categorical=True))
    elif enc_kind_train == "numeric":
        import src.train_xgb as tx
        tx._CAT_MAPPINGS_PATH = (
            ROOT / "models" / "feature_lists" / "xgb_v9_G0_basic30_seed0_val_cat_mappings.json"
        )
        X_ho_enc = tx._encode_categoricals(X_ho_raw, fit=False)
        y_pred = booster.predict(xgb.DMatrix(X_ho_enc))
    elif enc_kind_train == "te":
        X_ho_enc = encode_te(X_ho_raw, te_maps)
        y_pred = booster.predict(xgb.DMatrix(X_ho_enc))

    rmse = float(np.sqrt(np.mean((y_pred - y_ho) ** 2)))
    bias = float(np.mean(y_pred - y_ho))
    bias_pct = bias / float(np.mean(y_ho)) * 100
    # 按机场 RMSE
    ap_rmse = {}
    for ap in np.unique(df_ho_dep_full["ADEP_mvt"].values):
        m = df_ho_dep_full["ADEP_mvt"].values == ap
        if m.sum() == 0:
            continue
        sse = float(np.sum((y_pred[m] - y_ho[m]) ** 2))
        ap_rmse[str(ap)] = round(float(np.sqrt(sse / m.sum())), 2)
    print(
        f"  现训结果: RMSE={rmse:.4f}, bias={bias:+.4f}s ({bias_pct:+.2f}%)",
        flush=True,
    )
    print(
        f"  按机场 RMSE: LIRF={ap_rmse.get('LIRF', 0):.2f}, "
        f"LFPG={ap_rmse.get('LFPG', 0):.2f}, EGLL={ap_rmse.get('EGLL', 0):.2f}",
        flush=True,
    )
    return {
        "variant": variant,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(y_pred - y_ho))),
        "bias": bias,
        "bias_pct": bias_pct,
        "ap_rmse": ap_rmse,
        "n_train": int(len(y_tr)),
        "n_holdout": int(len(y_ho)),
        "feat_cols_n": len(feat_cols),
        "cat_cols_n": len(cat_cols),
    }


def main():
    df_train_full, df_ho_dep_full = load_data_cached()

    variants = ["H0", "H1", "H3", "H2"]  # H2 留最后(最慢);H0 baseline 必须先跑
    results = []
    for v in variants:
        r = run_variant(df_train_full, df_ho_dep_full, v)
        results.append(r)
        gc.collect()

    out_path = OUT_DIR / "encoding_ab_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n✅ 编码 A/B 结果: {out_path}", flush=True)
    print("\n=== 汇总 ===", flush=True)
    base = next((r for r in results if r["variant"] == "H0"), None)
    if base and base.get("rmse") is not None:
        print(f"  H0 baseline RMSE = {base['rmse']:.4f}", flush=True)
        for r in results:
            if r.get("rmse") is None:
                print(f"  {r['variant']:8s}  skipped={r.get('skipped')}", flush=True)
                continue
            delta = r["rmse"] - base["rmse"]
            print(f"  {r['variant']:8s}  RMSE={r['rmse']:.4f}  Δ={delta:+.4f}", flush=True)


if __name__ == "__main__":
    main()