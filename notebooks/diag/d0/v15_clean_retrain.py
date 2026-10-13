"""
干净重训脚本 — v15 地基验证
=====================================
口径 (严格):
- 训练折: 2025-02~12 (剔 TAXITIME <= 0 或 > 10800)
- Holdout: 2025-01 + 2025-07
- 特征: 71 = R7slim 69 + rwy_month_share + ap_month_hourly_density
- 模型: XGB / LGB / CAT 各 3 真 seed (42, 123, 2024)
- 高基数列整数编码 (*_cat_mappings), 原生 cat_features 只保留 4 低基数列
- 超参: 各库历史最优 (来自 tuned_xgb_params.json + 经验值)
- 评估口径 (固定分层):
  * 核心 RMSE: 剔 TAXITIME > 10800, n≈344,276
  * 全样本 RMSE: 保留全部 (含极端)
  * 分机场 bias/RMSE, 分 1/7月
- 输出: 每库每 seed 模型 + 每库每 seed best_iter + 集成结果 JSON

不轮询、不上传、不复跑老模型。
"""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

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

FEATS71 = [
    "sched_hour", "sched_dow", "sched_month", "sched_day", "sched_minute",
    "is_weekend", "is_morning_peak", "is_evening_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "airport_hour",
    "is_domestic_estimate", "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt",
    "is_diverted",
    "iobt_to_sched_sec", "eobt_to_sched_sec", "iobt_eobt_diff_sec",
    "dep_mvt_cnt_airport_5m", "dep_mvt_cnt_airport_10m", "dep_mvt_cnt_airport_15m",
    "dep_queue_n_airport",
    "dep_mvt_cnt_runway_5m", "dep_mvt_cnt_runway_10m", "dep_mvt_cnt_runway_15m",
    "dep_queue_n_runway",
    "ref_was_estimated",
    "mvt_n_ahead", "mvt_mu_rwy_15", "mvt_mu_rwy_30",
    "mvt_w_actual_15", "mvt_w_actual_30",
    "mvt_q_peak", "mvt_q_mean",
    "mvt_na_J", "mvt_na_H", "mvt_na_M", "mvt_na_L",
    "mvt_na_heavy_frac", "mvt_pos_rw", "mvt_na_heavy_x_pos",
    "mvt_mu_rwy_5", "mvt_w_actual_5",
    "dep_cnt_m_5", "dep_cnt_m_15", "dep_cnt_m_30",
    "aobt_cnt_m_5", "aobt_cnt_m_15", "aobt_cnt_m_30",
    "eobt_ahead_cnt_5", "eobt_ahead_cnt_15", "eobt_ahead_cnt_30",
    "dly_aobt_eobt", "dly_aobt_iobt", "dly_aobt_sched", "dly_lobt_aobt",
    "t_unimp_r7",
    "rwy_month_share", "ap_month_hourly_density",
]
assert len(FEATS71) == 71

LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
                  "AIRCRAFT_OPERATOR_flt"]

# 模型保存目录
MODEL_DIR = ROOT / "models" / "v15_clean_retrain"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = ROOT / "notebooks/diag/d0/v15_clean_retrain_result.json"

SEEDS = [42, 123, 2024]


def build_features(df_raw, ref_dep_for_d, lut, label=""):
    """重建全套特征 (71 列)."""
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(df_basic.copy(),
        include_a=True, include_b=True, include_c=True, include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    d_dict = D.compute_drift_features(df_full, ref_dep=ref_dep_for_d, label=label)
    for k, v in d_dict.items():
        df_full[k] = v
    t_vals, _ = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    extra_dict = L.compute_extra_features(df_full[dep_mask], label=label)
    for k, v in extra_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    mvt_dict = L.compute_mvt_anchor_features(df_full[dep_mask],
        t_unimp_vals=t_vals[dep_mask], label=label)
    for k, v in mvt_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    return df_full


def encode_cat_columns(df, cat_map):
    """高基数列整数编码, 低基数列转 category dtype."""
    df = df.copy()
    for col in HIGH_CARD_CATS:
        if col in df.columns:
            m = cat_map.get(col, {})
            df[col] = df[col].astype(str).map(m).fillna(-1).astype("int32")
    for col in LOW_CARD_CATS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def encode_for_catboost(df):
    """CAT 用: 全部数值化 (cat 转 int codes, object 转 float)."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].cat.codes.astype("int32")
        elif df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df


def evaluate_predictions(pred_arr, y_arr, name, aps_arr, months_arr, best_iter=None):
    """固定分层 RMSE 报告."""
    core_mask = (y_arr > 0) & (y_arr <= 10800)
    pred_core = pred_arr[core_mask]
    y_core = y_arr[core_mask]
    rmse_core = float(np.sqrt(np.mean((pred_core - y_core) ** 2)))
    bias_core = float(np.mean(pred_core - y_core))
    valid_mask = np.isfinite(y_arr) & (y_arr > 0)
    rmse_all = float(np.sqrt(np.mean((pred_arr[valid_mask] - y_arr[valid_mask]) ** 2)))
    bias_all = float(np.mean(pred_arr[valid_mask] - y_arr[valid_mask]))

    ap_per = {}
    for ap in sorted(np.unique(aps_arr)):
        m = (aps_arr == ap) & core_mask
        n = int(m.sum())
        if n < 100:
            continue
        r = float(np.sqrt(np.mean((pred_arr[m] - y_arr[m]) ** 2)))
        b = float(np.mean(pred_arr[m] - y_arr[m]))
        ap_per[ap] = {"n": n, "rmse_core": r, "bias": b}

    mo_per = {}
    for mo in [1, 7]:
        mm = (months_arr == mo) & core_mask
        n = int(mm.sum())
        if n < 10:
            continue
        r = float(np.sqrt(np.mean((pred_arr[mm] - y_arr[mm]) ** 2)))
        b = float(np.mean(pred_arr[mm] - y_arr[mm]))
        mo_per[f"month_{mo}"] = {"n": n, "rmse_core": r, "bias": b}

    return {
        "name": name,
        "best_iter": best_iter,
        "rmse_core": rmse_core,
        "bias_core": bias_core,
        "rmse_all": rmse_all,
        "bias_all": bias_all,
        "n_core": int(core_mask.sum()),
        "n_all": int(valid_mask.sum()),
        "per_airport_core": ap_per,
        "per_month_core": mo_per,
    }


def main():
    t_start_total = time.time()

    # ============================================================
    # 1. 加载数据 + 重建特征
    # ============================================================
    print("=" * 70)
    print("[1] 加载训练折 (2-12) + holdout (1+7), 重建 71 特征")
    print("=" * 70)

    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    train_dfs = []
    for m in train_months:
        fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
              if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
        train_dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_tr_raw = pd.concat(train_dfs, ignore_index=True)
    df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
    df_tr_dep_raw = df_tr[df_tr["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt = df_tr_dep_raw["TAXITIME_SEC_mvt"].values
    clean_tr = (tt > 0) & (tt <= 10800)
    df_tr_dep = df_tr_dep_raw[clean_tr].copy().reset_index(drop=True)
    print(f"  训练 DEP (清洗后, TAXITIME 0<x<10800): {len(df_tr_dep):,}")

    ho_dfs = []
    for m in [1, 7]:
        fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
        ho_dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
    df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
    df_ho_dep_raw = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)

    # LUT (训练折)
    lut = L.build_t_unimp_lut(df_tr_dep)
    print(f"  LUT: L1={len(lut['L1_q'])}, L2={len(lut['L2_q'])}, L3={len(lut['L3_q'])}")

    # 重建训练折特征
    print("  重建训练折特征 ...")
    t0 = time.time()
    df_tr_full = build_features(df_tr_raw, df_tr, lut, label="tr")
    df_tr_dep_full = df_tr_full[df_tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    df_tr_dep_full = df_tr_dep_full[
        (df_tr_dep_full["TAXITIME_SEC_mvt"] > 0) &
        (df_tr_dep_full["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)
    print(f"    训练 DEP 全套: {df_tr_dep_full.shape} ({time.time()-t0:.0f}s)")

    # 重建 holdout 特征
    print("  重建 holdout 特征 ...")
    t0 = time.time()
    df_ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
    df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    print(f"    Holdout DEP 全套: {df_ho_dep_full.shape} ({time.time()-t0:.0f}s)")

    # cat 整数编码 (高基数) + 保留 LGB 原生 cat (低基数)
    cat_map_path = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
    cat_map = json.loads(cat_map_path.read_text())
    df_tr_dep_full = encode_cat_columns(df_tr_dep_full, cat_map)
    df_ho_dep_full = encode_cat_columns(df_ho_dep_full, cat_map)

    # 补缺失列
    for split, df_check in [("tr", df_tr_dep_full), ("ho", df_ho_dep_full)]:
        miss = [f for f in FEATS71 if f not in df_check.columns]
        for f in miss:
            df_check[f] = np.nan

    # CAT 版 (整数化所有列, 不传 cat_features)
    df_tr_cat = encode_for_catboost(df_tr_dep_full[FEATS71].copy())
    df_ho_cat = encode_for_catboost(df_ho_dep_full[FEATS71].copy())

    # XGB 版 (整数编码, 但 category 列保留给 XGB 自己处理)
    from src import train_xgb as tx
    tx._CAT_MAPPINGS.clear()
    tx._CAT_MAPPINGS = cat_map
    X_tr_xgb = tx._encode_categoricals(df_tr_dep_full[FEATS71].copy(), fit=True).values.astype(np.float32)
    X_ho_xgb = tx._encode_categoricals(df_ho_dep_full[FEATS71].copy(), fit=False).values.astype(np.float32)

    # LGB 版 (低基数保留 category, XGB 编码列全转数值)
    df_tr_lgb = df_tr_dep_full[FEATS71].copy()
    df_ho_lgb = df_ho_dep_full[FEATS71].copy()

    y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
    y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
    aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
    months_col = pd.to_datetime(df_ho_dep_full["MVT_TIME_UTC_mvt"]).dt.month.values
    print(f"  y_ho 范围: {y_ho.min():.0f} ~ {y_ho.max():.0f}")

    # ============================================================
    # 2. XGB 3 seed (历史最优超参, 整数编码)
    # ============================================================
    print()
    print("=" * 70)
    print("[2] XGB 3 seed 训练")
    print("=" * 70)

    xgb_params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": 0.0147555939449363, "max_depth": 11,
        "min_child_weight": 5, "subsample": 0.9813515130184908,
        "colsample_bytree": 0.6720708449966734,
        "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
        "tree_method": "hist", "nthread": -1, "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_tr_xgb, label=y_tr, feature_names=FEATS71)
    dholdout = xgb.DMatrix(X_ho_xgb, label=y_ho, feature_names=FEATS71)

    xgb_preds = np.zeros(len(y_ho), dtype=np.float64)
    xgb_results = {}

    for seed in SEEDS:
        print(f"\n  --- XGB seed = {seed} ---")
        t0 = time.time()
        params = {**xgb_params, "seed": seed}
        booster = xgb.train(params, dtrain, num_boost_round=2500,
            evals=[(dholdout, "holdout")], early_stopping_rounds=100, verbose_eval=500)
        elapsed = time.time() - t0
        best_iter = booster.best_iteration + 1
        pred = booster.predict(dholdout, iteration_range=(0, best_iter))
        pred = np.clip(pred, 30, 10800)
        # 保存
        booster.save_model(str(MODEL_DIR / f"xgb_seed{seed}.json"))
        xgb_preds += pred / 3
        rep = evaluate_predictions(pred, y_ho, f"XGB_seed{seed}", aps, months_col, best_iter)
        rep["time_sec"] = round(elapsed, 1)
        xgb_results[f"seed{seed}"] = rep
        print(f"  best_iter={best_iter}, core={rep['rmse_core']:.2f}, all={rep['rmse_all']:.2f}, time={elapsed:.1f}s")

    # ============================================================
    # 3. LGB 3 seed (历史最优超参, 低基数 cat_features)
    # ============================================================
    print()
    print("=" * 70)
    print("[3] LGB 3 seed 训练")
    print("=" * 70)

    lgb_params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.0147555939449363,
        "num_leaves": 127, "max_depth": 11,
        "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
        "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
        "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
        "verbose": -1, "num_threads": -1,
    }

    lgb_preds = np.zeros(len(y_ho), dtype=np.float64)
    lgb_results = {}

    for seed in SEEDS:
        print(f"\n  --- LGB seed = {seed} ---")
        t0 = time.time()
        lgb_train = lgb.Dataset(df_tr_lgb, label=y_tr, categorical_feature=LOW_CARD_CATS)
        lgb_holdout = lgb.Dataset(df_ho_lgb, label=y_ho, categorical_feature=LOW_CARD_CATS, reference=lgb_train)
        params = {**lgb_params, "seed": seed}
        booster = lgb.train(params, lgb_train, num_boost_round=2500,
            valid_sets=[lgb_holdout],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
        elapsed = time.time() - t0
        best_iter = booster.best_iteration
        pred = booster.predict(df_ho_lgb, num_iteration=best_iter)
        pred = np.clip(pred, 30, 10800)
        booster.save_model(str(MODEL_DIR / f"lgb_seed{seed}.txt"))
        lgb_preds += pred / 3
        rep = evaluate_predictions(pred, y_ho, f"LGB_seed{seed}", aps, months_col, best_iter)
        rep["time_sec"] = round(elapsed, 1)
        lgb_results[f"seed{seed}"] = rep
        print(f"  best_iter={best_iter}, core={rep['rmse_core']:.2f}, all={rep['rmse_all']:.2f}, time={elapsed:.1f}s")

    # ============================================================
    # 4. CAT 3 seed (整数编码, 不传 cat_features)
    # ============================================================
    print()
    print("=" * 70)
    print("[4] CAT 3 seed 训练")
    print("=" * 70)

    cat_preds = np.zeros(len(y_ho), dtype=np.float64)
    cat_results = {}

    for seed in SEEDS:
        print(f"\n  --- CAT seed = {seed} ---")
        t0 = time.time()
        model = CatBoostRegressor(
            iterations=2500, learning_rate=0.0147555939449363,
            depth=11, l2_leaf_reg=5, random_seed=seed, verbose=500,
            early_stopping_rounds=100, thread_count=-1, eval_metric="RMSE",
            cat_features=[],
        )
        model.fit(df_tr_cat, y_tr, eval_set=(df_ho_cat, y_ho))
        elapsed = time.time() - t0
        if elapsed > 20 * 60:
            print(f"  ⚠️ CAT seed={seed} >20min, 停下")
            break
        best_iter = model.tree_count_
        pred = model.predict(df_ho_cat)
        pred = np.clip(pred, 30, 10800)
        model.save_model(str(MODEL_DIR / f"cat_seed{seed}"))
        cat_preds += pred / 3
        rep = evaluate_predictions(pred, y_ho, f"CAT_seed{seed}", aps, months_col, best_iter)
        rep["time_sec"] = round(elapsed, 1)
        cat_results[f"seed{seed}"] = rep
        print(f"  best_iter={best_iter}, core={rep['rmse_core']:.2f}, all={rep['rmse_all']:.2f}, time={elapsed:.1f}s")

    # ============================================================
    # 5. 三库等权集成
    # ============================================================
    print()
    print("=" * 70)
    print("[5] 三库等权集成")
    print("=" * 70)

    ens_preds = (xgb_preds + lgb_preds + cat_preds) / 3
    ens_report = evaluate_predictions(ens_preds, y_ho, "ENSEMBLE_3lib_avg", aps, months_col)
    ens_report["components"] = "XGB_seed{42,123,2024} + LGB_seed{42,123,2024} + CAT_seed{42,123,2024}"
    ens_report["ensemble_method"] = "simple mean (等权)"
    print(f"  集成 core RMSE = {ens_report['rmse_core']:.2f}")
    print(f"  集成 all RMSE  = {ens_report['rmse_all']:.2f}")

    # 各库 3seed 平均
    xgb_3seed = (xgb_results["seed42"]["rmse_core"] + xgb_results["seed123"]["rmse_core"] + xgb_results["seed2024"]["rmse_core"]) / 3
    lgb_3seed = (lgb_results["seed42"]["rmse_core"] + lgb_results["seed123"]["rmse_core"] + lgb_results["seed2024"]["rmse_core"]) / 3
    cat_3seed = (cat_results["seed42"]["rmse_core"] + cat_results["seed123"]["rmse_core"] + cat_results["seed2024"]["rmse_core"]) / 3
    print(f"  XGB 3seed 均值 core: {xgb_3seed:.2f}")
    print(f"  LGB 3seed 均值 core: {lgb_3seed:.2f}")
    print(f"  CAT 3seed 均值 core: {cat_3seed:.2f}")

    # ============================================================
    # 6. 保存结果
    # ============================================================
    out = {
        "task": "v15_clean_retrain_71feat_3lib_3seed",
        "config": {
            "training_months": "2025-02..12 (剔除 1+7)",
            "holdout_months": "2025-01 + 2025-07",
            "features": 71,
            "seeds": SEEDS,
            "lgb_cat_features_high_card_replaced_by_int_encoding": True,
            "low_card_cat_features_for_lgb": LOW_CARD_CATS,
            "cat_features_for_catboost": "[] (强制数值)",
        },
        "xgb_results": xgb_results,
        "lgb_results": lgb_results,
        "cat_results": cat_results,
        "xgb_3seed_avg_core": xgb_3seed,
        "lgb_3seed_avg_core": lgb_3seed,
        "cat_3seed_avg_core": cat_3seed,
        "ensemble_report": ens_report,
        "anchor_v15": {"score": 532.27, "metric": "ranking 344841 rows full-sample"},
    }
    RESULT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  结果 JSON: {RESULT_PATH}")
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  总耗时: {time.time()-t_start_total:.0f}s")

    # ============================================================
    # 7. 分机场报告 (核心口径)
    # ============================================================
    print()
    print("=" * 70)
    print("[7] 分机场报告 (集成)")
    print("=" * 70)
    print(f"{'ADEP':6s} {'n':>8s} {'core RMSE':>10s} {'core Bias':>10s}")
    for ap, info in sorted(ens_report["per_airport_core"].items()):
        print(f"  {ap:6s} {info['n']:>8d} {info['rmse_core']:>10.2f} {info['bias']:>+10.2f}")

    print(f"\n--- 集成 分月份 ---")
    for mo, info in ens_report["per_month_core"].items():
        print(f"  {mo}: n={info['n']}, core={info['rmse_core']:.2f}, bias={info['bias']:+.2f}")

    print()
    print("=" * 70)
    print(f"[报告完成] 不上传, 等用户确认")
    print(f"  集成 core (剔>10800, n={ens_report['n_core']}): {ens_report['rmse_core']:.2f}")
    print(f"  集成 all  (全样本, n={ens_report['n_all']}): {ens_report['rmse_all']:.2f}")
    print(f"  v15 真榜锚点: 532.27 (ranking 全样本)")
    print("=" * 70)


if __name__ == "__main__":
    main()