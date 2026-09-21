"""
阶段1 A/B Holdout 评估 (V2: 特征工程一次, 7 组 train+eval)
==========================================================

- 一次性构造全部 7 组共享的"最大特征集"(G5 全套 = basic30 + dep_mvt + queue + queue_heavy + arr)
- 每组按 feat_cols 筛选训练
- 每组 1000 轮 XGBoost 训练
- 沿用阶段 0 自洽断言

输出: notebooks/diag/holdout_v1_queue_ab_result.json
模型: models/xgb_v9_*_seed0_val.json
"""
import sys, json, hashlib, gc
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
sys.path.insert(0, str(ROOT))

from src.data_loader import filter_departures, parse_time_columns, load_month, load_ranking
from src.features import build_features as build_basic, BASELINE_FEATURE_COLS
from src.features_queue import add_queue_features
from src.train_xgb import _encode_categoricals, DEFAULT_PARAMS

OUT_PATH = ROOT / "notebooks" / "diag" / "holdout_v1_queue_ab_result.json"
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

# 各组的 feat 列(相对 G5 全套的子集)
GROUPS = {
    "G0_basic30": {
        "desc": "basic30 (基线, 对照)",
        "include_a": False, "include_b": False, "include_c": False,
        "include_d": False, "include_e": False,
    },
    "G1_A": {
        "desc": "+ A: dep_mvt_cnt_airport_{5,10,15}m",
        "include_a": True, "include_b": False, "include_c": False,
        "include_d": False, "include_e": False,
    },
    "G2_B": {
        "desc": "+ B: dep_queue_n_airport",
        "include_a": False, "include_b": True, "include_c": False,
        "include_d": False, "include_e": False,
    },
    "G3_AB": {
        "desc": "+ A + B",
        "include_a": True, "include_b": True, "include_c": False,
        "include_d": False, "include_e": False,
    },
    "G4_ABC": {
        "desc": "+ A + B + C(heavy)",
        "include_a": True, "include_b": True, "include_c": True,
        "include_d": False, "include_e": False,
    },
    "G5_ABCE": {
        "desc": "+ A + B + C + E(ARR, 全套)",
        "include_a": True, "include_b": True, "include_c": True,
        "include_d": False, "include_e": True,
    },
    "Gd_ABD_runway": {
        "desc": "+ A + B + D 跑道级(对照,非主)",
        "include_a": True, "include_b": True, "include_c": False,
        "include_d": True, "include_e": False,
    },
}


def build_full_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """构造 G5 全套(最大特征集):basic + A + B + C + D + E。
    返回包含所有可能需要的列。
    """
    df = build_basic(df_raw)
    df = add_queue_features(
        df,
        include_a=True, include_b=True, include_c=True,
        include_d=True, include_e=True,
    )
    return df


def get_feat_cols_for_group(df_full: pd.DataFrame, g: dict) -> list[str]:
    """根据组定义,从 full 特征集中筛选对应子集。"""
    new_feats = []
    if g["include_a"]:
        new_feats += ["dep_mvt_cnt_airport_5m", "dep_mvt_cnt_airport_10m", "dep_mvt_cnt_airport_15m"]
    if g["include_b"]:
        new_feats += ["dep_queue_n_airport"]
    if g["include_c"]:
        new_feats += ["dep_queue_heavy_n_airport"]
    if g["include_d"]:
        new_feats += ["dep_mvt_cnt_runway_5m", "dep_mvt_cnt_runway_10m", "dep_mvt_cnt_runway_15m",
                      "dep_queue_n_runway"]
    if g["include_e"]:
        new_feats += ["arr_mvt_cnt_airport_10m", "arr_mvt_cnt_airport_15m"]
    new_feats += ["ref_was_estimated"]
    return list(BASELINE_FEATURE_COLS) + [c for c in new_feats if c in df_full.columns]


def evaluate(booster, X, y, df_raw, feat_cols):
    X_enc = _encode_categoricals(X.copy(), fit=False)
    y_pred = booster.predict(xgb.DMatrix(X_enc))

    airports = df_raw["ADEP_mvt"].values
    rmse_total = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    mae_total = float(np.mean(np.abs(y_pred - y)))
    bias = float(np.mean(y_pred - y))
    bias_pct = bias / float(np.mean(y)) * 100

    df_h = pd.DataFrame({"y": y, "pred": y_pred})
    df_h["bin"] = pd.qcut(df_h["y"], q=[0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0],
                         labels=['p0-50','p50-90','p90-95','p95-99','p99-99.5','p99.5-99.9','p99.9-100'])
    total_sse = 0.0
    for b, sub in df_h.groupby("bin", observed=True):
        total_sse += float(((sub["y"] - sub["pred"]) ** 2).sum())
    recomputed_rmse = float(np.sqrt(total_sse / len(y)))
    sse_self_consistent = abs(recomputed_rmse - rmse_total) < 1e-6

    by_airport = {}
    for ap in np.unique(airports):
        mask = airports == ap
        by_airport[str(ap)] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean((y_pred[mask] - y[mask]) ** 2))),
        }

    importance = booster.get_score(importance_type="gain")
    return {
        "n": int(len(y)),
        "y_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(y_pred)),
        "rmse": rmse_total,
        "mae": mae_total,
        "bias": bias,
        "bias_pct": bias_pct,
        "self_consistent_sse": bool(sse_self_consistent),
        "by_airport": by_airport,
        "feature_importance_top10": sorted(
            [{"feat": k, "gain": float(v)} for k, v in importance.items()],
            key=lambda x: -x["gain"]
        )[:10],
    }, y_pred


def main():
    print("=" * 70, flush=True)
    print("阶段1 A/B Holdout 评估 V2 — 7 组 + 1 跑道级对照", flush=True)
    print("=" * 70, flush=True)

    # ===== 1. 加载 holdout =====
    print("\n[1] 加载 holdout(2025-01+07)...", flush=True)
    dfs = []
    for m in [1, 7]:
        fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
        dfs.append(load_month(f"data/raw/{fname}", columns=TRAIN_COLS))
    df_ho_raw = pd.concat(dfs, ignore_index=True)
    df_ho_all_raw = parse_time_columns(df_ho_raw, cols=TIME_COLS)
    df_ho_dep = filter_departures(df_ho_all_raw)
    df_ho_dep = df_ho_dep[(df_ho_dep["TAXITIME_SEC_mvt"] > 0) &
                          (df_ho_dep["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
    print(f"  Holdout DEP: {len(df_ho_dep):,}", flush=True)

    # ===== 2. 一次性构造 holdout 全套特征 =====
    print("\n[2] 构造 holdout 全套特征(G5) ...", flush=True)
    df_ho_full = build_full_features(df_ho_all_raw.copy())
    print(f"  Holdout 全套: {df_ho_full.shape}, 列数={len(df_ho_full.columns)}", flush=True)

    # ===== 3. 加载训练数据 =====
    print("\n[3] 加载训练(2025 全年,DEP+ARR,剔除 1+7)...", flush=True)
    months_to_load = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    train_dfs_arr = []
    for m in months_to_load:
        if m < 12:
            fname = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
        else:
            fname = f"training_2025-12-01_2026-01-01.parquet"
        df_m = load_month(f"data/raw/{fname}", columns=TRAIN_COLS)
        train_dfs_arr.append(df_m)
    df_train_all = pd.concat(train_dfs_arr, ignore_index=True)
    df_train_all = parse_time_columns(df_train_all, cols=TIME_COLS)
    n_dep = (df_train_all["PHASE_mvt"] == "DEP").sum()
    n_arr = (df_train_all["PHASE_mvt"] == "ARR").sum()
    print(f"  训练(含 ARR): {len(df_train_all):,} (DEP={n_dep:,}, ARR={n_arr:,})", flush=True)

    # ===== 4. 一次性构造训练全套特征 =====
    print("\n[4] 构造训练全套特征(G5) ...", flush=True)
    df_train_full = build_full_features(df_train_all.copy())
    print(f"  训练全套: {df_train_full.shape}, 列数={len(df_train_full.columns)}", flush=True)

    # ===== 5. 加载 ranking =====
    print("\n[5] 加载 ranking(DEP+ARR) ...", flush=True)
    df_rank_raw = load_ranking("data/raw")
    df_rank_all = parse_time_columns(df_rank_raw, cols=TIME_COLS)
    print(f"  Ranking raw: {len(df_rank_all):,}", flush=True)

    print("\n[6] 构造 ranking 全套特征(G5) ...", flush=True)
    df_rank_full = build_full_features(df_rank_all.copy())
    print(f"  Ranking 全套: {df_rank_full.shape}", flush=True)

    # ===== 7. 训练 7 组 =====
    y_tr_full = df_train_full["TAXITIME_SEC_mvt"].values
    mask_valid_tr = (df_train_full["TAXITIME_SEC_mvt"] > 0) & (df_train_full["TAXITIME_SEC_mvt"] < 10800)
    df_train_full = df_train_full[mask_valid_tr].reset_index(drop=True)
    y_tr_full = df_train_full["TAXITIME_SEC_mvt"].values

    y_ho = df_ho_dep["TAXITIME_SEC_mvt"].values

    results = {}
    for group_name, g in GROUPS.items():
        print(f"\n========== {group_name}: {g['desc']} ==========", flush=True)
        feat_cols = get_feat_cols_for_group(df_train_full, g)
        print(f"  特征: {len(feat_cols)} 列", flush=True)
        # 缺失补 NaN
        for c in feat_cols:
            if c not in df_train_full.columns:
                df_train_full[c] = np.nan
            if c not in df_ho_full.columns:
                df_ho_full[c] = np.nan
            if c not in df_rank_full.columns:
                df_rank_full[c] = np.nan

        X_tr = df_train_full[feat_cols]
        # holdout: 只取 DEP 部分的行,并应用 TAXITIME 过滤(与 df_ho_dep 一致)
        if "PHASE_mvt" in df_ho_full.columns:
            ho_dep_mask = df_ho_full["PHASE_mvt"].astype(str).values == "DEP"
            df_ho_dep_full = df_ho_full[ho_dep_mask].reset_index(drop=True)
        else:
            df_ho_dep_full = df_ho_full
        # TAXITIME 过滤(与 df_ho_dep 一致)
        df_ho_dep_full = df_ho_dep_full[
            (df_ho_dep_full["TAXITIME_SEC_mvt"] > 0) &
            (df_ho_dep_full["TAXITIME_SEC_mvt"] < 10800)
        ].reset_index(drop=True)
        for c in feat_cols:
            if c not in df_ho_dep_full.columns:
                df_ho_dep_full[c] = np.nan
        X_ho = df_ho_dep_full[feat_cols]
        y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values  # 重新取,确保一致
        X_rank = df_rank_full[feat_cols]

        # 报告新特征缺失率
        new_feats = [c for c in feat_cols if c not in BASELINE_FEATURE_COLS]
        miss_rates_tr = {c: float(X_tr[c].isna().mean()) for c in new_feats}
        miss_rates_ho = {c: float(X_ho[c].isna().mean()) for c in new_feats}
        print(f"  新特征缺失率(训练): {miss_rates_tr}", flush=True)
        print(f"  新特征缺失率(holdout): {miss_rates_ho}", flush=True)

        # cat 映射持久化
        cat_mappings_path = ROOT / "models" / "feature_lists" / f"xgb_v9_{group_name}_seed0_val_cat_mappings.json"
        if cat_mappings_path.exists():
            cat_mappings_path.unlink()

        import src.train_xgb as tx
        tx._CAT_MAPPINGS.clear()
        tx._CAT_MAPPINGS_PATH = cat_mappings_path
        X_tr_enc = tx._encode_categoricals(X_tr, fit=True)

        dtrain = xgb.DMatrix(X_tr_enc, label=y_tr_full)
        params = {**DEFAULT_PARAMS, "seed": 0}
        n_rounds = 1200
        print(f"  训练 {n_rounds} 轮...", flush=True)
        booster = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=400)

        model_path = ROOT / "models" / f"xgb_v9_{group_name}_seed0_val.json"
        booster.save_model(str(model_path))
        feat_list_path = ROOT / "models" / "feature_lists" / f"xgb_v9_{group_name}_seed0_val_features.json"
        feat_list_path.write_text(json.dumps({
            "model_name": model_path.name, "group": group_name, "stage": "val",
            "model_version": "v9", "seed": 0, "n_features": len(feat_cols),
            "features": feat_cols,
        }, indent=2))
        importance_df = pd.DataFrame([
            {"feature": k, "importance_gain": v}
            for k, v in booster.get_score(importance_type="gain").items()
        ]).sort_values("importance_gain", ascending=False)
        importance_path = ROOT / "models" / "feature_lists" / f"xgb_v9_{group_name}_seed0_val_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        # 评估 holdout
        tx._CAT_MAPPINGS_PATH = cat_mappings_path
        ho_metrics, ho_pred = evaluate(booster, X_ho, y_ho, df_ho_dep, feat_cols)

        # ranking
        tx._CAT_MAPPINGS_PATH = cat_mappings_path
        rank_pred = booster.predict(xgb.DMatrix(tx._encode_categoricals(X_rank.copy(), fit=False)))
        rank_pred_mean = float(np.mean(rank_pred))
        rank_vs_ho_diff = rank_pred_mean - ho_metrics["pred_mean"]

        # 断言
        assertions = {
            "sse_self_consistent": ho_metrics["self_consistent_sse"],
            "rmse_in_range": 250 < ho_metrics["rmse"] < 450,
            "bias_rel_le_8pct": abs(ho_metrics["bias_pct"]) <= 8.0,
            "rank_vs_holdout_pred_mean": abs(rank_vs_ho_diff) < 100,
        }
        assertions["all_pass"] = all(assertions.values())
        ho_metrics["rank_pred_mean"] = rank_pred_mean
        ho_metrics["rank_vs_holdout_diff"] = rank_vs_ho_diff
        ho_metrics["n_features"] = len(feat_cols)
        ho_metrics["feat_cols"] = feat_cols
        ho_metrics["miss_rates_train"] = miss_rates_tr
        ho_metrics["miss_rates_holdout"] = miss_rates_ho
        ho_metrics["assertions"] = assertions

        print(f"  RMSE={ho_metrics['rmse']:.2f}, MAE={ho_metrics['mae']:.2f}, bias={ho_metrics['bias']:+.1f}s ({ho_metrics['bias_pct']:+.2f}%)", flush=True)
        print(f"  ranking pred mean={rank_pred_mean:.1f}, diff={rank_vs_ho_diff:+.2f}", flush=True)
        print(f"  断言: {assertions}", flush=True)

        results[group_name] = {
            "desc": g["desc"],
            "model_path": str(model_path.relative_to(ROOT)),
            "feat_list_path": str(feat_list_path.relative_to(ROOT)),
            "cat_mappings_path": str(cat_mappings_path.relative_to(ROOT)),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()[:16],
            "metrics": ho_metrics,
        }

        del booster, dtrain, X_tr_enc
        gc.collect()

    # ===== 8. 汇总 =====
    base_rmse = results["G0_basic30"]["metrics"]["rmse"]
    summary_rows = []
    for gname, r in results.items():
        m = r["metrics"]
        summary_rows.append({
            "group": gname, "desc": r["desc"],
            "rmse": m["rmse"], "mae": m["mae"], "bias_pct": m["bias_pct"],
            "rank_vs_ho_diff": m["rank_vs_holdout_diff"], "n_features": m["n_features"],
            "rmse_delta_vs_G0": m["rmse"] - base_rmse,
            "rmse_delta_pct": (m["rmse"] - base_rmse) / base_rmse * 100,
            "all_pass": m["assertions"]["all_pass"],
        })

    out = {
        "holdout_period": "2025-01 + 2025-07",
        "train_period_excluded": "2025-01, 2025-07",
        "groups": results,
        "summary": summary_rows,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n\n汇总写入 {OUT_PATH}", flush=True)
    print(f"\n===== A/B 汇总(相对 G0) =====", flush=True)
    print(f"{'Group':<18} {'Desc':<45} {'RMSE':>7} {'Δs':>7} {'Δ%':>7} {'MAE':>7} {'bias%':>7} {'pass':>5}", flush=True)
    for row in summary_rows:
        print(f"{row['group']:<18} {row['desc'][:43]:<45} {row['rmse']:>7.2f} {row['rmse_delta_vs_G0']:>+7.2f} {row['rmse_delta_pct']:>+6.2f}% {row['mae']:>7.2f} {row['bias_pct']:>+6.2f}% {'✅' if row['all_pass'] else '❌':>5}", flush=True)


if __name__ == "__main__":
    main()
