"""
A/B 汇总: 加载清洗 A/B 与编码 A/B 的预测,做 paired + 按天 block bootstrap
输出 ab_summary.json
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

spec = importlib.util.spec_from_file_location(
    "ab_v2_holdout_v1_queue_ab",
    ROOT / "notebooks/diag/holdout_v1_queue_ab.py",
)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

OUT_DIR = ROOT / "notebooks/diag" / "ab_v2"


def load_holdout_data():
    """加载 holdout DEP 过滤后的数据(与 runner 一致)"""
    dfs = []
    for m in [1, 7]:
        fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        dfs.append(ab.load_month(f"data/raw/{fname}", columns=ab.TRAIN_COLS))
    df_ho_raw = pd.concat(dfs, ignore_index=True)
    df_ho_all_raw = ab.parse_time_columns(df_ho_raw, cols=ab.TIME_COLS)
    df_ho_full = ab.build_full_features(df_ho_all_raw.copy())

    if "PHASE_mvt" in df_ho_full.columns:
        ho_dep_mask = df_ho_full["PHASE_mvt"].astype(str).values == "DEP"
        df_ho_dep = df_ho_full[ho_dep_mask].reset_index(drop=True)
    else:
        df_ho_dep = df_ho_full
    df_ho_dep = df_ho_dep[
        (df_ho_dep["TAXITIME_SEC_mvt"] > 0)
        & (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)
    return df_ho_dep


def train_baseline(df_train_full, df_ho_dep):
    """重训 baseline 得到 y_pred_base。复用 _gate 逻辑(简化)。"""
    G0 = ab.GROUPS["G0_basic30"]
    feat_cols = ab.get_feat_cols_for_group(df_train_full, G0)
    for c in feat_cols:
        if c not in df_train_full.columns:
            df_train_full[c] = np.nan
        if c not in df_ho_dep.columns:
            df_ho_dep[c] = np.nan
    X_tr = df_train_full[feat_cols]
    X_ho = df_ho_dep[feat_cols]

    import src.train_xgb as tx
    cat_mappings_path = (
        ROOT / "models" / "feature_lists" / "xgb_v9_G0_basic30_seed0_val_cat_mappings.json"
    )
    if cat_mappings_path.exists():
        cat_mappings_path.unlink()
    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS_PATH = cat_mappings_path
    X_tr_enc = tx._encode_categoricals(X_tr, fit=True)

    dtrain = xgb.DMatrix(X_tr_enc, label=df_train_full["TAXITIME_SEC_mvt"].values)
    params = {**ab.DEFAULT_PARAMS, "seed": 0}
    booster = xgb.train(params, dtrain, num_boost_round=1200, verbose_eval=False)

    tx._CAT_MAPPINGS_PATH = cat_mappings_path
    X_ho_enc = tx._encode_categoricals(X_ho, fit=False)
    y_pred = booster.predict(xgb.DMatrix(X_ho_enc))
    return y_pred, feat_cols


def load_train():
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
    y = df_train_full["TAXITIME_SEC_mvt"].values
    mask = (y > 0) & (y < 10800)
    df_train_full = df_train_full[mask].reset_index(drop=True)
    return df_train_full


def compute_sse_by_ap(y_true, y_pred, ap):
    sse = {}
    for a in np.unique(ap):
        m = ap == a
        sse[str(a)] = float(np.sum((y_true[m] - y_pred[m]) ** 2))
    return sse


def block_bootstrap_delta(y_true, y_pred_base, y_pred_var, day_key, n_boot=2000, seed=42):
    """按天 block bootstrap ΔRMSE 的 95% CI"""
    rng = np.random.default_rng(seed)
    unique_days = np.unique(day_key)
    n_days = len(unique_days)
    day_to_idx = {d: np.where(day_key == d)[0] for d in unique_days}

    delta_boot = []
    base_sse = float(np.sum((y_true - y_pred_base) ** 2))
    var_sse = float(np.sum((y_true - y_pred_var) ** 2))
    n = len(y_true)
    base_rmse = float(np.sqrt(base_sse / n))
    var_rmse = float(np.sqrt(var_sse / n))
    delta_point = var_rmse - base_rmse

    for _ in range(n_boot):
        sampled_days = rng.choice(unique_days, size=n_days, replace=True)
        idx = np.concatenate([day_to_idx[d] for d in sampled_days])
        b_sse_base = float(np.sum((y_true[idx] - y_pred_base[idx]) ** 2))
        b_sse_var = float(np.sum((y_true[idx] - y_pred_var[idx]) ** 2))
        b_n = len(idx)
        b_delta = float(np.sqrt(b_sse_var / b_n) - np.sqrt(b_sse_base / b_n))
        delta_boot.append(b_delta)
    delta_boot = np.array(delta_boot)
    ci_lo = float(np.percentile(delta_boot, 2.5))
    ci_hi = float(np.percentile(delta_boot, 97.5))
    # p-value (two-sided): fraction of boot deltas with same sign as point? 简化: 0 是否在 CI 内
    significant = not (ci_lo <= 0 <= ci_hi)
    return {
        "delta_rmse": delta_point,
        "base_rmse": base_rmse,
        "var_rmse": var_rmse,
        "ci_2_5": ci_lo,
        "ci_97_5": ci_hi,
        "significant": significant,
        "n_boot": n_boot,
        "n_days": n_days,
    }


def run_variant_clean(df_train_full, df_ho_dep, method):
    """重跑一个清洗变体,返回 y_pred"""
    G0 = ab.GROUPS["G0_basic30"]
    feat_cols = ab.get_feat_cols_for_group(df_train_full, G0)

    df_train = df_train_full.copy()
    y_tr = df_train["TAXITIME_SEC_mvt"].values
    ap_tr = df_train["ADEP_mvt"].astype(str).values

    if method == "none":
        pass
    elif method == "drop_lt60":
        mask = y_tr >= 60
        df_train = df_train[mask].reset_index(drop=True)
    elif method == "winsor_per_ap_0.1_99.9":
        keep = np.ones(len(y_tr), dtype=bool)
        for ap in np.unique(ap_tr):
            idx = ap_tr == ap
            lo = np.percentile(y_tr[idx], 0.1)
            hi = np.percentile(y_tr[idx], 99.9)
            keep[idx] = (y_tr[idx] >= lo) & (y_tr[idx] <= hi)
        df_train = df_train[keep].reset_index(drop=True)
    elif method == "winsor_per_ap_0.5_99.5":
        keep = np.ones(len(y_tr), dtype=bool)
        for ap in np.unique(ap_tr):
            idx = ap_tr == ap
            lo = np.percentile(y_tr[idx], 0.5)
            hi = np.percentile(y_tr[idx], 99.5)
            keep[idx] = (y_tr[idx] >= lo) & (y_tr[idx] <= hi)
        df_train = df_train[keep].reset_index(drop=True)
    else:
        raise ValueError(method)

    for c in feat_cols:
        if c not in df_train.columns:
            df_train[c] = np.nan
        if c not in df_ho_dep.columns:
            df_ho_dep[c] = np.nan
    X_tr = df_train[feat_cols]
    X_ho = df_ho_dep[feat_cols]

    import src.train_xgb as tx
    cat_mappings_path = (
        ROOT / "models" / "feature_lists" / f"xgb_v9_G0_basic30_seed0_val_cat_mappings.json"
    )
    if cat_mappings_path.exists():
        cat_mappings_path.unlink()
    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS_PATH = cat_mappings_path
    X_tr_enc = tx._encode_categoricals(X_tr, fit=True)

    dtrain = xgb.DMatrix(X_tr_enc, label=df_train["TAXITIME_SEC_mvt"].values)
    params = {**ab.DEFAULT_PARAMS, "seed": 0}
    booster = xgb.train(params, dtrain, num_boost_round=1200, verbose_eval=False)

    tx._CAT_MAPPINGS_PATH = cat_mappings_path
    X_ho_enc = tx._encode_categoricals(X_ho, fit=False)
    y_pred = booster.predict(xgb.DMatrix(X_ho_enc))
    return y_pred


def run_variant_encoding(df_train_full, df_ho_dep, variant, feat_cols):
    """重跑一个编码变体,返回 y_pred"""
    cat_cols = [c for c in feat_cols if df_train_full[c].dtype == object]
    X_tr = df_train_full[feat_cols].copy()
    X_ho = df_ho_dep[feat_cols].copy()
    y_tr = df_train_full["TAXITIME_SEC_mvt"].values

    if variant == "H0":
        import src.train_xgb as tx
        cat_mappings_path = (
            ROOT / "models" / "feature_lists" / "xgb_v9_G0_basic30_seed0_val_cat_mappings.json"
        )
        if cat_mappings_path.exists():
            cat_mappings_path.unlink()
        tx._CAT_MAPPINGS.clear()
        tx._CAT_MAPPINGS_PATH = cat_mappings_path
        X_tr_enc = tx._encode_categoricals(X_tr, fit=True)
        dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
        booster = xgb.train({**ab.DEFAULT_PARAMS, "seed": 0}, dtrain, num_boost_round=1200, verbose_eval=False)
        tx._CAT_MAPPINGS_PATH = cat_mappings_path
        X_ho_enc = tx._encode_categoricals(X_ho, fit=False)
        return booster.predict(xgb.DMatrix(X_ho_enc))

    if variant == "H1":
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("category")
            X_ho[c] = X_ho[c].astype("category")
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        booster = xgb.train(
            {**ab.DEFAULT_PARAMS, "seed": 0, "enable_categorical": True},
            dtrain, num_boost_round=1200, verbose_eval=False
        )
        return booster.predict(xgb.DMatrix(X_ho, enable_categorical=True))

    if variant == "H3":
        # OOF smoothed target encoding(时间折)
        order = np.argsort(df_train_full["MVT_TIME_UTC_mvt"].values)
        n = len(df_train_full)
        n_splits = 5
        chunk = n // n_splits
        fold_id = np.zeros(n, dtype=int)
        for k in range(n_splits):
            lo = k * chunk
            hi = (k + 1) * chunk if k < n_splits - 1 else n
            fold_id[order[lo:hi]] = k
        global_mean = float(np.mean(y_tr))
        m_smooth = 20

        # 先把非 cat 列转成数值(cat 列后面替换为 TE float 值)
        num_cols = [c for c in feat_cols if c not in cat_cols]

        def _fit_te_mask(mask):
            maps = {}
            for c in cat_cols:
                vals = df_train_full.loc[mask, c].astype(str).values
                ys = y_tr[mask]
                df_sub = pd.DataFrame({"v": vals, "y": ys})
                stats = df_sub.groupby("v")["y"].agg(["mean", "count"])
                smoothed = (
                    stats["count"] * stats["mean"] + m_smooth * global_mean
                ) / (stats["count"] + m_smooth)
                maps[c] = smoothed.to_dict()
            return maps

        # 全量 fit(给 holdout 用)
        full_maps = _fit_te_mask(np.ones(n, dtype=bool))

        # OOF transform train:每折用其他折 fit
        X_tr_te = pd.DataFrame(index=X_tr.index)
        for c in num_cols:
            X_tr_te[c] = X_tr[c].values
        for c in cat_cols:
            X_tr_te[c] = global_mean  # 兜底
        for k in range(n_splits):
            valid_mask = fold_id == k
            train_mask = fold_id != k
            fold_maps = _fit_te_mask(train_mask)
            for c in cat_cols:
                valid_vals = df_train_full.loc[valid_mask, c].astype(str).values
                m = fold_maps[c]
                X_tr_te.loc[valid_mask, c] = [
                    m.get(v, global_mean) for v in valid_vals
                ]

        # transform holdout (用全量 fit)
        X_ho_te = pd.DataFrame(index=X_ho.index)
        for c in num_cols:
            X_ho_te[c] = X_ho[c].values
        for c in cat_cols:
            X_ho_te[c] = (
                X_ho[c]
                .astype(str)
                .map(lambda v, m=full_maps[c], g=global_mean: m.get(v, g))
                .astype(float)
            )

        dtrain = xgb.DMatrix(X_tr_te, label=y_tr)
        booster = xgb.train(
            {**ab.DEFAULT_PARAMS, "seed": 0},
            dtrain,
            num_boost_round=1200,
            verbose_eval=False,
        )
        return booster.predict(xgb.DMatrix(X_ho_te))

    raise ValueError(variant)


def main():
    print("Loading data...", flush=True)
    df_train_full = load_train()
    df_ho_dep = load_holdout_data()
    y_true = df_ho_dep["TAXITIME_SEC_mvt"].values
    ap_ho = df_ho_dep["ADEP_mvt"].astype(str).values
    # 按天 key
    day_key = pd.to_datetime(df_ho_dep["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m-%d").values

    G0 = ab.GROUPS["G0_basic30"]
    feat_cols = ab.get_feat_cols_for_group(df_train_full, G0)

    # baseline 预测
    print("\n[baseline] H0/C0 numeric + no clean", flush=True)
    y_pred_base, _ = train_baseline(df_train_full.copy(), df_ho_dep.copy())
    base_rmse = float(np.sqrt(np.mean((y_true - y_pred_base) ** 2)))
    print(f"  baseline RMSE = {base_rmse:.4f}", flush=True)

    # 清洗 A/B: C0 = baseline(直接用 y_pred_base),C1~C3 重训
    base_sse_ap = compute_sse_by_ap(y_true, y_pred_base, ap_ho)
    base_mae = float(np.mean(np.abs(y_true - y_pred_base)))
    base_bias = float(np.mean(y_pred_base - y_true))
    base_bias_pct = base_bias / float(np.mean(y_true)) * 100
    base_ap_rmse = {}
    for a in np.unique(ap_ho):
        m = ap_ho == a
        base_ap_rmse[str(a)] = round(
            float(np.sqrt(np.mean((y_true[m] - y_pred_base[m]) ** 2))), 2
        )

    clean_methods = [
        ("C1", "drop_lt60"),
        ("C2", "winsor_per_ap_0.1_99.9"),
        ("C3", "winsor_per_ap_0.5_99.5"),
    ]
    clean_results = [
        {
            "label": "C0",
            "method": "none",
            "rmse": base_rmse,
            "mae": base_mae,
            "bias": base_bias,
            "bias_pct": base_bias_pct,
            "ap_rmse": base_ap_rmse,
            "ap_sse": base_sse_ap,
            "bootstrap_delta": {
                "delta_rmse": 0.0,
                "base_rmse": base_rmse,
                "var_rmse": base_rmse,
                "ci_2_5": 0.0,
                "ci_97_5": 0.0,
                "significant": False,
                "note": "baseline 自身",
            },
        }
    ]
    for label, method in clean_methods:
        print(f"\n[clean] {label} = {method}", flush=True)
        y_pred = run_variant_clean(df_train_full.copy(), df_ho_dep.copy(), method)
        sse_ap = compute_sse_by_ap(y_true, y_pred, ap_ho)
        boot = block_bootstrap_delta(y_true, y_pred_base, y_pred, day_key)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        bias = float(np.mean(y_pred - y_true))
        bias_pct = bias / float(np.mean(y_true)) * 100
        ap_rmse = {}
        for a in np.unique(ap_ho):
            m = ap_ho == a
            ap_rmse[str(a)] = round(float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))), 2)
        clean_results.append({
            "label": label,
            "method": method,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "bias_pct": bias_pct,
            "ap_rmse": ap_rmse,
            "ap_sse": sse_ap,
            "bootstrap_delta": boot,
        })
        print(f"  RMSE={rmse:.4f}, Δ={rmse - base_rmse:+.4f}, sig={boot['significant']}", flush=True)

    # 编码 A/B: H0 = baseline(直接用 y_pred_base),H1/H3 重训,H2 读已有结果
    enc_variants = [
        ("H1", "H1"),
        ("H3", "H3"),
    ]
    enc_results = [
        {
            "label": "H0",
            "variant": "H0",
            "rmse": base_rmse,
            "mae": base_mae,
            "bias": base_bias,
            "bias_pct": base_bias_pct,
            "ap_rmse": base_ap_rmse,
            "ap_sse": base_sse_ap,
            "bootstrap_delta": {
                "delta_rmse": 0.0,
                "base_rmse": base_rmse,
                "var_rmse": base_rmse,
                "ci_2_5": 0.0,
                "ci_97_5": 0.0,
                "significant": False,
                "note": "baseline 自身",
            },
        }
    ]
    for label, variant in enc_variants:
        print(f"\n[enc] {label}", flush=True)
        y_pred = run_variant_encoding(df_train_full.copy(), df_ho_dep.copy(), variant, feat_cols)
        sse_ap = compute_sse_by_ap(y_true, y_pred, ap_ho)
        boot = block_bootstrap_delta(y_true, y_pred_base, y_pred, day_key)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        bias = float(np.mean(y_pred - y_true))
        bias_pct = bias / float(np.mean(y_true)) * 100
        ap_rmse = {}
        for a in np.unique(ap_ho):
            m = ap_ho == a
            ap_rmse[str(a)] = round(float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))), 2)
        enc_results.append({
            "label": label,
            "variant": variant,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "bias_pct": bias_pct,
            "ap_rmse": ap_rmse,
            "ap_sse": sse_ap,
            "bootstrap_delta": boot,
        })
        print(f"  RMSE={rmse:.4f}, Δ={rmse - base_rmse:+.4f}, sig={boot['significant']}", flush=True)

    # 从已有结果里读 H2(只需要 RMSE + Δ,不需要 bootstrap,因为劣化明显)
    enc_all = json.load(open(OUT_DIR / "encoding_ab_results.json"))
    h2 = next(x for x in enc_all if x["variant"] == "H2")
    enc_results.append({
        "label": "H2",
        "variant": "H2",
        "rmse": h2["rmse"],
        "mae": h2["mae"],
        "bias": h2["bias"],
        "bias_pct": h2["bias_pct"],
        "ap_rmse": h2["ap_rmse"],
        "bootstrap_delta": {
            "delta_rmse": h2["rmse"] - base_rmse,
            "note": "未做 bootstrap(劣化显著,跳过)",
        },
    })

    out = {
        "baseline": {
            "rmse": base_rmse,
            "mae": float(np.mean(np.abs(y_true - y_pred_base))),
            "bias": float(np.mean(y_pred_base - y_true)),
            "bias_pct": float(np.mean(y_pred_base - y_true) / np.mean(y_true) * 100),
        },
        "clean_ab": clean_results,
        "encoding_ab": enc_results,
        "n_holdout": int(len(y_true)),
        "n_train": int(len(df_train_full)),
        "feat_cols_n": len(feat_cols),
    }
    out_path = OUT_DIR / "ab_summary.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✅ 汇总结果: {out_path}", flush=True)

    # 打印总表
    print("\n" + "=" * 90)
    print("清洗 A/B (baseline = C0 = 357.14)")
    print("=" * 90)
    print(f"{'方法':8s} {'RMSE':>10s} {'ΔRMSE':>10s} {'CI 2.5%':>10s} {'CI 97.5%':>10s} {'显著?':>6s} {'MAE':>8s} {'bias%':>8s}")
    for r in clean_results:
        b = r["bootstrap_delta"]
        print(
            f"{r['label']:8s} {r['rmse']:10.4f} {b['delta_rmse']:10.4f} "
            f"{b['ci_2_5']:10.4f} {b['ci_97_5']:10.4f} "
            f"{'Y' if b['significant'] else 'N':>6s} {r['mae']:8.2f} {r['bias_pct']:8.2f}%"
        )

    print("\n" + "=" * 90)
    print("编码 A/B (baseline = H0 = 357.14)")
    print("=" * 90)
    print(f"{'方法':8s} {'RMSE':>10s} {'ΔRMSE':>10s} {'CI 2.5%':>10s} {'CI 97.5%':>10s} {'显著?':>6s} {'MAE':>8s} {'bias%':>8s}")
    for r in enc_results:
        b = r["bootstrap_delta"]
        ci_lo = b.get("ci_2_5", float("nan"))
        ci_hi = b.get("ci_97_5", float("nan"))
        sig = "Y" if b.get("significant", False) else "N"
        print(
            f"{r['label']:8s} {r['rmse']:10.4f} {b['delta_rmse']:10.4f} "
            f"{ci_lo:10.4f} {ci_hi:10.4f} "
            f"{sig:>6s} {r['mae']:8.2f} {r['bias_pct']:8.2f}%"
        )


if __name__ == "__main__":
    main()