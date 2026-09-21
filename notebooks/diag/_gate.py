"""
Step1 复现门禁: 严格 1:1 照 holdout_v1_queue_ab.main() 入口,只跑 G0_basic30 一组
- 必须 RMSE=357.x / bias≈-1.8% / LIRF≈587 / LFPG≈406
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

# 冻结入口: 直接 import holdout_v1_queue_ab.py 作为模块,符号全复用
spec = importlib.util.spec_from_file_location(
    "holdout_v1_queue_ab",
    str(ROOT / "notebooks/diag/holdout_v1_queue_ab.py"),
)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def main():
    print("=" * 70, flush=True)
    print("Step1 复现门禁 — 严格 1:1 照 holdout_v1_queue_ab.main()", flush=True)
    print("只跑 G0_basic30, num_boost_round=1200, seed=0", flush=True)
    print("=" * 70, flush=True)

    # ===== 1. 加载 holdout =====
    print("\n[1] 加载 holdout(2025-01+07)...", flush=True)
    dfs = []
    for m in [1, 7]:
        fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        dfs.append(ab.load_month(f"data/raw/{fname}", columns=ab.TRAIN_COLS))
    df_ho_raw = pd.concat(dfs, ignore_index=True)
    df_ho_all_raw = ab.parse_time_columns(df_ho_raw, cols=ab.TIME_COLS)
    df_ho_dep = ab.filter_departures(df_ho_all_raw)
    df_ho_dep = df_ho_dep[
        (df_ho_dep["TAXITIME_SEC_mvt"] > 0)
        & (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)
    print(f"  Holdout DEP: {len(df_ho_dep):,}", flush=True)

    # ===== 2. 构造 holdout 全套特征 =====
    print("\n[2] 构造 holdout 全套特征(G5)...", flush=True)
    df_ho_full = ab.build_full_features(df_ho_all_raw.copy())
    print(f"  Holdout 全套: {df_ho_full.shape}, 列数={len(df_ho_full.columns)}", flush=True)

    # ===== 3. 加载训练数据 =====
    print("\n[3] 加载训练(2025 全年, DEP+ARR, 剔除 1+7)...", flush=True)
    months_to_load = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    train_dfs_arr = []
    for m in months_to_load:
        if m < 12:
            fname = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        else:
            fname = f"training_2025-12-01_2026-01-01.parquet"
        df_m = ab.load_month(f"data/raw/{fname}", columns=ab.TRAIN_COLS)
        train_dfs_arr.append(df_m)
    df_train_all = pd.concat(train_dfs_arr, ignore_index=True)
    df_train_all = ab.parse_time_columns(df_train_all, cols=ab.TIME_COLS)
    n_dep = (df_train_all["PHASE_mvt"] == "DEP").sum()
    n_arr = (df_train_all["PHASE_mvt"] == "ARR").sum()
    print(f"  训练(含 ARR): {len(df_train_all):,} (DEP={n_dep:,}, ARR={n_arr:,})", flush=True)

    # ===== 4. 构造训练全套特征 =====
    print("\n[4] 构造训练全套特征(G5)...", flush=True)
    df_train_full = ab.build_full_features(df_train_all.copy())
    print(f"  训练全套: {df_train_full.shape}, 列数={len(df_train_full.columns)}", flush=True)

    # ===== 5/6. ranking (略,不在此评估用) =====

    # ===== 7. 训练 G0_basic30 (单组,与 main 循环一致) =====
    y_tr_full = df_train_full["TAXITIME_SEC_mvt"].values
    mask_valid_tr = (df_train_full["TAXITIME_SEC_mvt"] > 0) & (
        df_train_full["TAXITIME_SEC_mvt"] < 10800
    )
    df_train_full = df_train_full[mask_valid_tr].reset_index(drop=True)
    y_tr_full = df_train_full["TAXITIME_SEC_mvt"].values

    group_name = "G0_basic30"
    g = ab.GROUPS[group_name]
    print(f"\n========== {group_name}: {g['desc']} ==========", flush=True)

    feat_cols = ab.get_feat_cols_for_group(df_train_full, g)
    print(f"  特征: {len(feat_cols)} 列", flush=True)

    # 缺失补 NaN
    for c in feat_cols:
        if c not in df_train_full.columns:
            df_train_full[c] = np.nan
        if c not in df_ho_full.columns:
            df_ho_full[c] = np.nan

    X_tr = df_train_full[feat_cols]

    # holdout 取 DEP + TAXITIME 过滤
    if "PHASE_mvt" in df_ho_full.columns:
        ho_dep_mask = df_ho_full["PHASE_mvt"].astype(str).values == "DEP"
        df_ho_dep_full = df_ho_full[ho_dep_mask].reset_index(drop=True)
    else:
        df_ho_dep_full = df_ho_full
    df_ho_dep_full = df_ho_dep_full[
        (df_ho_dep_full["TAXITIME_SEC_mvt"] > 0)
        & (df_ho_dep_full["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)
    for c in feat_cols:
        if c not in df_ho_dep_full.columns:
            df_ho_dep_full[c] = np.nan
    X_ho = df_ho_dep_full[feat_cols]
    y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values

    # cat 映射持久化
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
    print(f"  训练 {n_rounds} 轮...", flush=True)
    booster = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=400)

    # holdout 评估
    tx._CAT_MAPPINGS_PATH = cat_mappings_path
    ho_metrics, ho_pred = ab.evaluate(booster, X_ho, y_ho, df_ho_dep, feat_cols)

    # 报告
    rmse = ho_metrics["rmse"]
    bias = ho_metrics["bias"]
    bias_pct = ho_metrics["bias_pct"]
    ap_rmse = {ap: v["rmse"] for ap, v in ho_metrics["by_airport"].items()}
    print(f"\n  现训 G0 RMSE={rmse:.4f}, bias={bias:+.4f}s ({bias_pct:+.2f}%)", flush=True)
    print("  目标: RMSE≈357.14, bias≈-1.8%, LIRF≈587, LFPG≈406", flush=True)
    print("  按机场 RMSE(应 EGLL≈447, LFPG≈406, LIRF≈587):", flush=True)
    for ap in ["EGLL", "LFPG", "LIRF", "EHAM", "LTFM", "EDDF"]:
        print(f"    {ap}: {ap_rmse.get(ap, 'N/A')}", flush=True)

    # 门禁判定
    target_rmse = 357.14
    target_lirf = 587
    target_lfpg = 406
    target_egll = 447
    rmse_ok = abs(rmse - target_rmse) < 2.0
    lirf_ok = abs(ap_rmse.get("LIRF", 0) - target_lirf) < 20
    lfpg_ok = abs(ap_rmse.get("LFPG", 0) - target_lfpg) < 30
    egll_ok = abs(ap_rmse.get("EGLL", 0) - target_egll) < 15

    print("\n  门禁判定:", flush=True)
    print(f"    RMSE 差 {rmse - target_rmse:+.2f}: {'PASS' if rmse_ok else 'FAIL'}", flush=True)
    print(
        f"    LIRF 差 {ap_rmse.get('LIRF', 0) - target_lirf:+.2f}: {'PASS' if lirf_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"    LFPG 差 {ap_rmse.get('LFPG', 0) - target_lfpg:+.2f}: {'PASS' if lfpg_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"    EGLL 差 {ap_rmse.get('EGLL', 0) - target_egll:+.2f}: {'PASS' if egll_ok else 'FAIL'}",
        flush=True,
    )

    out = {
        "current_train_rmse": rmse,
        "current_bias": bias,
        "current_bias_pct": bias_pct,
        "ap_rmse": ap_rmse,
        "target_rmse": target_rmse,
        "target_lirf": target_lirf,
        "target_lfpg": target_lfpg,
        "target_egll": target_egll,
        "gate_pass": rmse_ok and lirf_ok and lfpg_ok and egll_ok,
        "n_holdout_eval": int(len(y_ho)),
        "n_train_used": int(len(y_tr_full)),
        "feat_cols_n": len(feat_cols),
        "mae": ho_metrics["mae"],
        "self_consistent_sse": ho_metrics["self_consistent_sse"],
    }
    out_path = ROOT / "notebooks/diag" / "gate_reproduce.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✅ 现训门禁结果保存到 {out_path}", flush=True)
    print("\n=== 结论 ===", flush=True)
    if out["gate_pass"]:
        print("  [PASS] 现训 H0=357.x, 与阶段 0/1 一致。可进入下一步 A/B。", flush=True)
    else:
        print("  [FAIL] 现训 H0 仍 != 357.x。停止一切 A/B, 先排查根因。", flush=True)
        print(
            "    排查方向: cat fit 时机/作用域、训练月份、特征清单、超参、target 过滤、dtypes",
            flush=True,
        )


if __name__ == "__main__":
    main()