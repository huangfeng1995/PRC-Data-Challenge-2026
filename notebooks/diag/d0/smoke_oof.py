"""
smoke_oof.py — 单折限行验证轻量化生效
=========================================
跑 1 折:
- 训练集: 从训练月份里随机抽 100,000 行 (按时间排序后取末 80% 训, 末 20% 验)
- 预测集: 从被留出月份里随机抽 20,000 行
- 1 seed, ROUNDS=1500 上界 + ES 80
打印:
  - encode_X 后各列 dtype (HIGH_CARD 整数化是否真生效)
  - LGB best_iteration (ES 真早停了多少轮)
  - 单模型训练秒数
  - 推算: 全量 (1.93M × 5 fold × 1 seed) 总耗时
"""
from __future__ import annotations
import sys, os, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

from src.data_loader import load_month, parse_time_columns

TRAIN_COLS = [
    "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
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
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]
CAT_MAP = json.loads((ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json").read_text())
LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
ROUNDS = 1500


def add_basic_features(df_raw):
    """最小特征工程, 只够让 LGB 能跑 (与 oof_calibrate.py 主流程不同的简化版).
    复用 build_features 需要 _r7_lib/_r8_drift_lib 等模块, smoke 不必要, 我们直接用最小集."""
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df["sched_hour"] = df["SCHED_TIME_UTC_mvt"].dt.hour
    df["sched_dow"] = df["SCHED_TIME_UTC_mvt"].dt.dayofweek
    df["sched_month"] = df["SCHED_TIME_UTC_mvt"].dt.month
    df["sched_day"] = df["SCHED_TIME_UTC_mvt"].dt.day
    df["sched_minute"] = df["SCHED_TIME_UTC_mvt"].dt.minute
    df["is_weekend"] = (df["sched_dow"] >= 5).astype("int8")
    df["is_morning_peak"] = ((df["sched_hour"] >= 7) & (df["sched_hour"] <= 9)).astype("int8")
    df["is_evening_peak"] = ((df["sched_hour"] >= 17) & (df["sched_hour"] <= 20)).astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * df["sched_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["sched_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["sched_dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["sched_dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["sched_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["sched_month"] / 12)
    df["airport_hour"] = df["ADEP_mvt"].astype(str) + "_" + df["sched_hour"].astype(str)
    df["is_domestic_estimate"] = (df["ADEP_mvt"] == df["ADES_mvt"]).astype("int8")
    df["is_diverted"] = (df["ADES_flt"].astype(str) != df["ADES_FILED_flt"].astype(str)).astype("int8")
    df["ref_was_estimated"] = df["AOBT_3_flt"].isna().astype("int8")
    # 简单数字特征占位 (与原口径不同, 但足够 LGB smoke)
    for c in ["iobt_to_sched_sec", "eobt_to_sched_sec", "iobt_eobt_diff_sec",
              "dep_mvt_cnt_airport_5m", "dep_mvt_cnt_airport_10m", "dep_mvt_cnt_airport_15m",
              "dep_queue_n_airport",
              "dep_mvt_cnt_runway_5m", "dep_mvt_cnt_runway_10m", "dep_mvt_cnt_runway_15m",
              "dep_queue_n_runway",
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
              "rwy_month_share", "ap_month_hourly_density"]:
        df[c] = np.nan
    return df


def encode_X(df, feats):
    X = df[feats].copy()
    for col in HIGH_CARD_CATS:
        if col in X.columns:
            m = CAT_MAP.get(col, {})
            X[col] = X[col].astype(str).map(m).fillna(-1).astype("int32")
    for col in LOW_CARD_CATS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def main():
    print("=" * 70)
    print("[smoke] 单折限行验证轻量化生效")
    print("=" * 70)
    t_start = time.time()

    # 训练月份 2-12, 抽 100K
    months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    print(f"\n加载训练月份 {months} ...")
    dfs = []
    for m in months:
        fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
              if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
        dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_tr_raw = pd.concat(dfs, ignore_index=True)
    dep_tr = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                       (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
                       (df_tr_raw["TAXITIME_SEC_mvt"] <= 10800)].copy()
    print(f"  全训练 DEP: {len(dep_tr):,}")

    # 抽 100K (按时间排序后取前面, 模拟 1 折的 80%)
    dep_tr = dep_tr.sort_values("MVT_TIME_UTC_mvt").reset_index(drop=True)
    n_tr = 100_000
    sub_tr = dep_tr.iloc[:n_tr].copy()
    print(f"  抽 {n_tr:,} 行作训练池")

    # 末 20% 作 ES val
    cut = int(n_tr * 0.8)
    tr_pos = sub_tr.iloc[:cut]
    va_pos = sub_tr.iloc[cut:]
    print(f"  ES split: train={len(tr_pos):,}, val={len(va_pos):,}")

    # 预测集: 1 月, 抽 20K
    print("\n加载预测月份 [1] ...")
    df_ho = load_month("data/raw/training_2025-01-01_2025-02-01.parquet",
                        columns=TRAIN_COLS)
    dep_ho = df_ho[(df_ho["PHASE_mvt"] == "DEP") &
                   (df_ho["TAXITIME_SEC_mvt"] > 0) &
                   (df_ho["TAXITIME_SEC_mvt"] <= 10800)].copy()
    sub_ho = dep_ho.sort_values("MVT_TIME_UTC_mvt").iloc[:20_000].copy()
    print(f"  预测集: {len(sub_ho):,}")

    # 特征工程
    print("\n构建特征 ...")
    t0 = time.time()
    df_tr_full = add_basic_features(pd.concat([tr_pos, va_pos], ignore_index=True))
    df_ho_full = add_basic_features(sub_ho)
    print(f"  特征工程: {time.time()-t0:.1f}s")

    # 编码
    X_all = pd.concat([df_tr_full, df_ho_full], ignore_index=True)
    X_enc = encode_X(X_all, FEATS71)
    print(f"\n[编码 dtype 检查]")
    for col in HIGH_CARD_CATS + LOW_CARD_CATS:
        print(f"  {col:25s} → {X_enc[col].dtype}")

    n_tr2 = len(tr_pos)
    n_va2 = len(va_pos)
    n_total_p = n_tr2 + n_va2 + len(sub_ho)
    Xtr = X_enc.iloc[:cut]
    Xva = X_enc.iloc[cut:n_tr2 + n_va2]
    Xp  = X_enc.iloc[n_tr2 + n_va2:]
    ytr = tr_pos["TAXITIME_SEC_mvt"].values.astype(np.float32)
    yva = va_pos["TAXITIME_SEC_mvt"].values.astype(np.float32)
    yp  = sub_ho["TAXITIME_SEC_mvt"].values.astype(np.float32)
    print(f"\n  Xtr {Xtr.shape}, Xva {Xva.shape}, Xp {Xp.shape}")

    # 训练
    print("\n训练 (1 seed, ROUNDS=1500 上界, ES 80) ...")
    t0 = time.time()
    ds_tr = lgb.Dataset(Xtr, label=ytr, categorical_feature=LOW_CARD_CATS)
    ds_va = lgb.Dataset(Xva, label=yva, categorical_feature=LOW_CARD_CATS, reference=ds_tr)
    b = lgb.train(dict(LGB_PARAMS, seed=42), ds_tr,
                  num_boost_round=ROUNDS,
                  valid_sets=[ds_va],
                  callbacks=[lgb.early_stopping(80, verbose=False),
                             lgb.log_evaluation(0)])
    train_time = time.time() - t0
    best_iter = b.best_iteration
    print(f"  best_iter = {best_iter} (上界 {ROUNDS}, ES patience 80)")
    print(f"  训练耗时: {train_time:.2f}s")

    # 预测
    t0 = time.time()
    p = b.predict(Xp, num_iteration=best_iter)
    pred_time = time.time() - t0
    p = np.clip(p, 30, 10800)
    rmse = float(np.sqrt(np.mean((p - yp) ** 2)))
    print(f"  预测耗时: {pred_time:.2f}s")
    print(f"  smoke RMSE: {rmse:.2f}")

    # 推算全量
    print("\n[全量推算]")
    # 单折全量 1.93M × 5 折 × 1 seed
    # smoke 训练用 80K (n_tr2=80K), 全量 1.93M (n=~1,930,000)
    # 但 ES 触发后, 训练是 rounds-bound, 不是 row-bound (对 LGB hist)
    # 假设 rounds 与数据量近似线性 (经验), 倍率 = full_train / smoke_train ≈ 24
    # 但 rounds 又被 ES cap, 所以应按 best_iter 与 rounds 之比修正
    rounds_used_frac = best_iter / ROUNDS
    smoke_train_full = 80_000
    full_train_per_fold = 1_930_000  # 11 月训练
    row_factor = full_train_per_fold / smoke_train_full  # ~24
    sec_per_fold_full = train_time * row_factor * rounds_used_frac
    # 校正: 全量每折也是 ES, best_iter 大致稳定, 假设与 smoke 同比例
    total_sec = sec_per_fold_full * 5  # 5 折
    print(f"  smoke 训练行数: {smoke_train_full:,}")
    print(f"  全量训练行数/折: {full_train_per_fold:,} (倍率 x{row_factor:.1f})")
    print(f"  rounds 利用率: {rounds_used_frac*100:.1f}% ({best_iter}/{ROUNDS})")
    print(f"  推算单折全量训练耗时: {sec_per_fold_full:.0f}s ({sec_per_fold_full/60:.1f}min)")
    print(f"  推算 5 折 × 1 seed 全量训练总耗时: {total_sec:.0f}s ({total_sec/60:.1f}min)")

    # 保存
    out = {
        "smoke_train_time_sec": round(train_time, 2),
        "smoke_pred_time_sec": round(pred_time, 2),
        "best_iter": int(best_iter),
        "rounds_cap": ROUNDS,
        "rounds_used_frac": round(rounds_used_frac, 3),
        "smoke_train_rows": smoke_train_full,
        "full_train_rows_per_fold": full_train_per_fold,
        "row_factor": round(row_factor, 1),
        "estimated_single_fold_full_sec": round(sec_per_fold_full),
        "estimated_5fold_1seed_total_sec": round(total_sec),
        "estimated_5fold_1seed_total_min": round(total_sec / 60, 1),
        "n_features": len(FEATS71),
        "high_card_cols_dtype": {c: str(X_enc[c].dtype) for c in HIGH_CARD_CATS},
        "low_card_cols_dtype": {c: str(X_enc[c].dtype) for c in LOW_CARD_CATS},
        "smoke_rmse": rmse,
    }
    fn = D0 / "smoke_oof.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n✅ 结果: {fn}")
    print(f"  smoke 总耗时: {(time.time()-t_start):.1f}s")


if __name__ == "__main__":
    main()