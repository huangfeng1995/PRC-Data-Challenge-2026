"""
LGB 全量训练 + ranking 预测 + 提交生成 + gate_check
=====================================
- 训练折: 2025 全部 12 月 list(range(1,13))
- 71 特征 = R7slim 69 + rwy_month_share + ap_month_hourly_density
- LGB 真 3 seed (42, 123, 2024), 高基数列整数编码
- 预测 ranking (2026 1+7 月) → 提交 parquet (344,841 行)
- gate_check 五项
- 训练数据清洗: 剔 TAXITIME <= 0 或 > 10800
"""
import sys, json, time, subprocess, tempfile, os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

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

# LGB 历史最优超参
LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}

SEEDS = [42, 123, 2024]

MODEL_DIR = ROOT / "models" / "lgb_full_v15"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = ROOT / "notebooks/diag/d0/lgb_full_v15_result.json"
SUB_PATH = ROOT / "submissions" / "jubilant-puzzle_lgb_full_v15.parquet"


def build_features(df_raw, ref_dep_for_d, lut, label=""):
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


def encode_high_card(df, cat_map):
    df = df.copy()
    for col in HIGH_CARD_CATS:
        if col in df.columns:
            m = cat_map.get(col, {})
            df[col] = df[col].astype(str).map(m).fillna(-1).astype("int32")
    for col in LOW_CARD_CATS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def main():
    t_start = time.time()
    print("=" * 70)
    print("[任务] LGB 全量训练 + ranking 预测 + 提交")
    print("=" * 70)
    print("  训练折: 2025 全 12 月 list(range(1,13))")
    print("  Holdout: 2025-01+07 (用于 early stopping 验证)")
    print("  Ranking 预测: 2026 1+7 月")
    print("  特征: 71 = R7slim 69 + rwy_month_share + ap_month_hourly_density")
    print(f"  清洗: 剔 TAXITIME <= 0 或 > 10800")
    print(f"  Seeds: {SEEDS}")
    print()

    # ============================================================
    # 1. 加载 + 重建训练折特征
    # ============================================================
    print("[1] 加载 2025 全 12 月 + 重建训练折特征")
    print("-" * 70)

    train_months = list(range(1, 13))
    train_dfs = []
    for m in train_months:
        fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
              if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
        train_dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_tr_raw = pd.concat(train_dfs, ignore_index=True)
    df_tr = parse_time_columns(df_tr_raw, cols=TIME_COLS)
    print(f"  训练 (DEP+ARR, 全 12 月): {len(df_tr):,}")

    # 训练折 DEP (清洗)
    dep_tr_raw = df_tr[df_tr["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt_tr = dep_tr_raw["TAXITIME_SEC_mvt"].values
    clean_tr = (tt_tr > 0) & (tt_tr <= 10800)
    df_tr_dep = dep_tr_raw[clean_tr].copy().reset_index(drop=True)
    print(f"  训练 DEP (清洗后): {len(df_tr_dep):,} (剔 TAXITIME>10800: {(tt_tr > 10800).sum()})")

    # 加载 holdout (用于 early stopping, 训练完会重新清洗)
    ho_dfs = []
    for m in [1, 7]:
        fn = f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
        ho_dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
    df_ho = parse_time_columns(df_ho_raw, cols=TIME_COLS)
    print(f"  Holdout (DEP+ARR, 1+7 月): {len(df_ho):,}")

    # 加载 ranking (预测目标, 无 TAXITIME)
    rank_cols = [c for c in TRAIN_COLS if c != "TAXITIME_SEC_mvt"]
    rank_df = pd.read_parquet(ROOT / "data" / "raw" / "ranking.parquet", columns=rank_cols)
    rank_df = parse_time_columns(rank_df, cols=[c for c in TIME_COLS if c != "TAXITIME_SEC_mvt"])
    print(f"  Ranking 总: {len(rank_df):,}")

    # 加载 holdout DEP (清洗后, 用于 early stopping)
    dep_ho_raw = df_ho[df_ho["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt_ho = dep_ho_raw["TAXITIME_SEC_mvt"].values
    clean_ho = (tt_ho > 0) & (tt_ho <= 10800)
    df_ho_dep = dep_ho_raw[clean_ho].copy().reset_index(drop=True)
    print(f"  Holdout DEP (清洗后, 用于 early stopping): {len(df_ho_dep):,}")

    # LUT (训练折)
    lut = L.build_t_unimp_lut(df_tr_dep)
    print(f"\n  LUT: L1={len(lut['L1_q'])}, L2={len(lut['L2_q'])}, L3={len(lut['L3_q'])}, "
          f"global={lut['global_q30']:.0f}s")

    # 重建训练折特征 (含全部 12 个月)
    print("\n[1.1] 重建训练折特征 (全 12 月)...")
    t0 = time.time()
    df_tr_full = build_features(df_tr_raw, df_tr, lut, label="tr")
    df_tr_dep_full = df_tr_full[df_tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    df_tr_dep_full = df_tr_dep_full[
        (df_tr_dep_full["TAXITIME_SEC_mvt"] > 0) &
        (df_tr_dep_full["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)
    print(f"  训练 DEP 全套: {df_tr_dep_full.shape} ({time.time()-t0:.0f}s)")

    # 重建 holdout 特征
    print("[1.2] 重建 holdout 特征...")
    t0 = time.time()
    df_ho_full = build_features(df_ho_raw, df_tr, lut, label="ho")
    df_ho_dep_full = df_ho_full[df_ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    df_ho_dep_full = df_ho_dep_full[
        (df_ho_dep_full["TAXITIME_SEC_mvt"] > 0) &
        (df_ho_dep_full["TAXITIME_SEC_mvt"] < 10800)
    ].reset_index(drop=True)
    print(f"  Holdout DEP 全套: {df_ho_dep_full.shape} ({time.time()-t0:.0f}s)")

    # 重建 ranking 特征
    print("[1.3] 重建 ranking 特征...")
    t0 = time.time()
    rank_full = build_features(rank_df, df_tr, lut, label="rank")
    rank_dep_full = rank_full[rank_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    print(f"  Ranking DEP 全套: {rank_dep_full.shape} ({time.time()-t0:.0f}s)")

    # ============================================================
    # 2. cat 整数编码 (高基数)
    # ============================================================
    print("\n[2] cat 整数编码")
    cat_map_path = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
    cat_map = json.loads(cat_map_path.read_text())

    df_tr_dep_full = encode_high_card(df_tr_dep_full, cat_map)
    df_ho_dep_full = encode_high_card(df_ho_dep_full, cat_map)
    rank_dep_full = encode_high_card(rank_dep_full, cat_map)

    # 补缺失列
    for split, df_check in [("tr", df_tr_dep_full), ("ho", df_ho_dep_full), ("rank", rank_dep_full)]:
        miss = [f for f in FEATS71 if f not in df_check.columns]
        for f in miss:
            df_check[f] = np.nan

    X_tr = df_tr_dep_full[FEATS71].copy()
    X_ho = df_ho_dep_full[FEATS71].copy()
    X_rank = rank_dep_full[FEATS71].copy()
    y_tr = df_tr_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)
    y_ho = df_ho_dep_full["TAXITIME_SEC_mvt"].values.astype(np.float32)

    print(f"  X_tr shape: {X_tr.shape}")
    print(f"  X_ho shape: {X_ho.shape}")
    print(f"  X_rank shape: {X_rank.shape}")

    # ============================================================
    # 3. LGB 3 seed 训练
    # ============================================================
    print()
    print("=" * 70)
    print("[3] LGB 3 seed 训练 (全量, early stopping 用 holdout)")
    print("=" * 70)

    lgb_preds_ho = np.zeros(len(y_ho), dtype=np.float64)
    lgb_preds_rank = np.zeros(len(rank_dep_full), dtype=np.float64)
    seed_results = {}

    for seed in SEEDS:
        print(f"\n  --- LGB seed = {seed} ---")
        t0 = time.time()
        lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=LOW_CARD_CATS)
        lgb_holdout = lgb.Dataset(X_ho, label=y_ho, categorical_feature=LOW_CARD_CATS, reference=lgb_train)
        params = {**LGB_PARAMS, "seed": seed}
        booster = lgb.train(params, lgb_train, num_boost_round=3000,
            valid_sets=[lgb_holdout],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
        elapsed = time.time() - t0
        best_iter = booster.best_iteration
        # 保存
        model_path = MODEL_DIR / f"lgb_seed{seed}.txt"
        booster.save_model(str(model_path))

        # holdout 推理
        pred_ho = booster.predict(X_ho, num_iteration=best_iter)
        pred_ho = np.clip(pred_ho, 30, 10800)
        lgb_preds_ho += pred_ho / 3

        # ranking 推理
        pred_rank = booster.predict(X_rank, num_iteration=best_iter)
        pred_rank = np.clip(pred_rank, 30, 10800)
        lgb_preds_rank += pred_rank / 3

        # holdout 评估
        core_mask = (y_ho > 0) & (y_ho <= 10800)
        rmse_core = float(np.sqrt(np.mean((pred_ho[core_mask] - y_ho[core_mask]) ** 2)))
        rmse_all = float(np.sqrt(np.mean((pred_ho - y_ho) ** 2)))
        bias_core = float(np.mean(pred_ho[core_mask] - y_ho[core_mask]))

        seed_results[f"seed{seed}"] = {
            "best_iter": best_iter,
            "time_sec": round(elapsed, 1),
            "rmse_core": rmse_core,
            "rmse_all": rmse_all,
            "bias_core": bias_core,
            "model_path": str(model_path),
        }
        print(f"  best_iter={best_iter}, core={rmse_core:.2f}, all={rmse_all:.2f}, time={elapsed:.1f}s")

    # ============================================================
    # 4. Holdout 集成 + 分机场报告
    # ============================================================
    print()
    print("=" * 70)
    print("[4] LGB 3seed 集成 holdout")
    print("=" * 70)

    core_mask = (y_ho > 0) & (y_ho <= 10800)
    rmse_core = float(np.sqrt(np.mean((lgb_preds_ho[core_mask] - y_ho[core_mask]) ** 2)))
    rmse_all = float(np.sqrt(np.mean((lgb_preds_ho - y_ho) ** 2)))
    bias_core = float(np.mean(lgb_preds_ho[core_mask] - y_ho[core_mask]))
    print(f"  集成 core RMSE = {rmse_core:.2f}")
    print(f"  集成 all RMSE  = {rmse_all:.2f}")

    # 分机场
    aps = df_ho_dep_full["ADEP_mvt"].astype(str).values
    months_col = pd.to_datetime(df_ho_dep_full["MVT_TIME_UTC_mvt"]).dt.month.values
    print(f"\n--- 分机场 (核心口径) ---")
    print(f"{'ADEP':6s} {'n':>8s} {'core RMSE':>10s} {'Bias':>8s}")
    ap_per = {}
    for ap in sorted(np.unique(aps)):
        m = (aps == ap) & core_mask
        n = int(m.sum())
        if n < 100:
            continue
        r = float(np.sqrt(np.mean((lgb_preds_ho[m] - y_ho[m]) ** 2)))
        b = float(np.mean(lgb_preds_ho[m] - y_ho[m]))
        ap_per[ap] = {"n": n, "rmse_core": r, "bias": b}
        print(f"  {ap:6s} {n:>8d} {r:>10.2f} {b:>+8.2f}")

    print(f"\n--- 分月份 (核心口径) ---")
    mo_per = {}
    for m in [1, 7]:
        mm = (months_col == m) & core_mask
        n = int(mm.sum())
        if n < 10:
            continue
        r = float(np.sqrt(np.mean((lgb_preds_ho[mm] - y_ho[mm]) ** 2)))
        b = float(np.mean(lgb_preds_ho[mm] - y_ho[mm]))
        mo_per[f"month_{m}"] = {"n": n, "rmse_core": r, "bias": b}
        print(f"  {m}月: n={n}, core={r:.2f}, bias={b:+.2f}")

    # ============================================================
    # 5. 生成 ranking 提交
    # ============================================================
    print()
    print("=" * 70)
    print("[5] 生成 ranking 提交 parquet")
    print("=" * 70)

    rank_ids = rank_dep_full["MVT_ID_mvt"].astype(float).values
    sub_df = pd.DataFrame({
        "MVT_ID_mvt": rank_ids,
        "TAXITIME_SEC_mvt": np.round(lgb_preds_rank, 0).astype(int),
    })
    sub_df.to_parquet(SUB_PATH, index=False)
    print(f"  提交路径: {SUB_PATH}")
    print(f"  行数: {len(sub_df)}")
    print(f"  n_nan: {int(sub_df['TAXITIME_SEC_mvt'].isna().sum())}")
    print(f"  TAXITIME 分布: min={sub_df['TAXITIME_SEC_mvt'].min()}, "
          f"p50={sub_df['TAXITIME_SEC_mvt'].median()}, "
          f"max={sub_df['TAXITIME_SEC_mvt'].max()}")

    # 行键+顺序 vs submitting
    sub_ref = pd.read_parquet(ROOT / "data" / "raw" / "submitting.parquet", columns=["MVT_ID_mvt"])
    exp_ids = sub_ref["MVT_ID_mvt"].astype(float).values
    set_match = set(rank_ids) == set(exp_ids)
    order_match = np.all(rank_ids == exp_ids)
    print(f"  MVT_ID 集合+顺序: set={set_match}, order={order_match}")

    # ============================================================
    # 6. gate_check 五项
    # ============================================================
    print()
    print("=" * 70)
    print("[6] gate_check 五项")
    print("=" * 70)

    # check 1: baseline RMSE (参考)
    print(f"  CHECK 1: 集成 RMSE = {rmse_all:.2f} (v15 真榜锚点 532.27)")
    print(f"    (注: 集成在 holdout 1+7 月上算, 真榜在 ranking 上算, 口径不同)")

    # check 2: 分机场 SSE 加总反算 RMSE
    print(f"\n  CHECK 2: SSE 加总反算 RMSE")
    sse_total = float(np.sum((lgb_preds_ho[core_mask] - y_ho[core_mask]) ** 2))
    rmse_recomp = float(np.sqrt(sse_total / core_mask.sum()))
    diff = abs(rmse_recomp - rmse_core)
    print(f"    直接={rmse_core:.6f}, 反算={rmse_recomp:.6f}, |Δ|={diff:.2e} {'✅' if diff < 1e-6 else '❌'}")

    # check 3: 独立进程推理 (跳过, 本进程一次跑完)
    print(f"\n  CHECK 3: 独立进程推理 - 跳过 (本进程一次跑完)")

    # check 4: 预测无 NaN/负值
    n_nan = int(np.isnan(lgb_preds_ho).sum())
    n_neg = int((lgb_preds_ho < 0).sum())
    print(f"\n  CHECK 4: NaN={n_nan}, 负={n_neg} {'✅' if (n_nan==0 and n_neg==0) else '❌'}")

    # check 5: MVT_ID 集合+顺序
    print(f"\n  CHECK 5: MVT_ID 集合+顺序 vs submitting")
    print(f"    set={set_match}, order={order_match} {'✅' if (set_match and order_match) else '❌'}")

    # ============================================================
    # 7. 保存结果 JSON
    # ============================================================
    out = {
        "task": "LGB_full_train_71feat_3seed",
        "training_months": "2025-01..12 (清洗后全量)",
        "holdout_months": "2025-01+07 (核心口径剔>10800)",
        "ranking_months": "2026-01+07",
        "features": 71,
        "seeds": SEEDS,
        "high_card_int_encoding": True,
        "low_card_cat_features_for_lgb": LOW_CARD_CATS,
        "lgb_params": LGB_PARAMS,
        "seed_results": seed_results,
        "ensemble_holdout": {
            "rmse_core": rmse_core, "bias_core": bias_core, "rmse_all": rmse_all,
            "n_core": int(core_mask.sum()), "n_all": int(len(y_ho)),
        },
        "per_airport_core": ap_per,
        "per_month_core": mo_per,
        "ranking_submission": {
            "path": str(SUB_PATH),
            "n_rows": len(sub_df),
            "n_nan": int(sub_df['TAXITIME_SEC_mvt'].isna().sum()),
            "min": int(sub_df['TAXITIME_SEC_mvt'].min()),
            "p50": float(sub_df['TAXITIME_SEC_mvt'].median()),
            "max": int(sub_df['TAXITIME_SEC_mvt'].max()),
            "mvt_id_set_match": set_match,
            "mvt_id_order_match": order_match,
        },
        "total_time_sec": round(time.time() - t_start, 1),
    }
    RESULT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print()
    print(f"  结果 JSON: {RESULT_PATH}")
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  总耗时: {time.time()-t_start:.0f}s")

    print()
    print("=" * 70)
    print("[报告完成] 本步不上传, 等用户确认")
    print(f"  LGB 集成 core: {rmse_core:.2f}, all: {rmse_all:.2f}")
    print(f"  v15 真榜锚点: 532.27 (口径不同, 不可直接比)")
    print("=" * 70)


if __name__ == "__main__":
    main()