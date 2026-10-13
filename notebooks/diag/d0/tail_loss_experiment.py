"""
端到端尾部放开训练 (不靠事后校正)
=====================================
- split: train=2025-02..12, holdout=2025-01+07
- 特征: 71 (整数编码), cap=8000, ES on 11+12 月
- 单 seed (42), 充分训练不撞顶

组:
  A   L2 回归基线 (无 sample_weight, clip [30, 10800])
  D1  L2 + w=sqrt(TAXI/中位)归一化, clip [30, 10800]
  D2  L2 + w=(TAXI/中位)线性归一化, clip [30, 10800]
  D3  D2 但预测放开上 clip (clip [30, ∞))
  E1  objective=quantile alpha=0.5, clip [30, 10800]
  E2  objective=quantile alpha=0.1..0.9 (9 模型), 重构条件分布均值, clip [30, 10800]

评估:
  full/core RMSE, 分机场, Jan/Jul 分别
  长滑行档 (1800-3600/3600-10800) 预测均值、σpred/σtrue
  >10800 行: D3 vs A 误差对比

红线: 不改 label / 不 clip 真值训练 / 不建单机场模型 / 单阶段 / 单 seed (避免撞 11+12 ES 消耗)
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

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
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]
FOCUS_AP = ["LIRF", "EGLL", "LTFM", "LFPG", "EHAM"]

LGB_PARAMS_L2 = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
SEED = 42
ROUNDS = int(os.environ.get("ROUNDS", 8000))  # cap 8000, ES 避免撞顶
ES_ROUNDS = 100
CAT_MAP = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())


def build_features(df_raw, ref_dep_for_d, lut, label=""):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(df_basic.copy(),
                                 include_a=True, include_b=True, include_c=True,
                                 include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")
    for k, v in D.compute_drift_features(df_full, ref_dep=ref_dep_for_d, label=label).items():
        df_full[k] = v
    t_vals, _ = L.lookup_t_unimp(df_full, lut)
    df_full["t_unimp_r7"] = t_vals
    dep_mask = (df_full["PHASE_mvt"] == "DEP").values
    for k, v in L.compute_extra_features(df_full[dep_mask], label=label).items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    for k, v in L.compute_mvt_anchor_features(df_full[dep_mask],
                                              t_unimp_vals=t_vals[dep_mask], label=label).items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    return df_full


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
    t_start = time.time()
    print("=" * 70)
    print("[尾部放开训练实验] split=2025-02..12 / 1+7, 71 特征, seed=42")
    print("=" * 70)

    MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                       if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
                  for m in range(1, 13)}

    # 加载
    print("\n[1] 加载")
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    tr_dfs = [load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in train_months]
    df_tr_raw = pd.concat(tr_dfs, ignore_index=True)
    ho_dfs = [load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in [1, 7]]
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)

    # ES 验证集: 11+12 月训练折内
    es_dfs = [load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in [11, 12]]
    df_es_raw = pd.concat(es_dfs, ignore_index=True)

    # DEP 清洗
    df_tr_dep_all = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                              (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
                              (df_tr_raw["TAXITIME_SEC_mvt"] <= 10800)].copy()
    lut = L.build_t_unimp_lut(df_tr_dep_all)
    print(f"  训练 DEP (2-12, TX∈(0,10800]): {len(df_tr_dep_all):,}")

    # 重建训练集 / ES / holdout 特征
    print("\n[2] 重建特征")
    t0 = time.time()
    tr_full = build_features(df_tr_raw, df_tr_raw, lut, label="tr")
    tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt = tr_dep["TAXITIME_SEC_mvt"].values
    tr_dep = tr_dep[(tt > 0) & (tt <= 10800)].reset_index(drop=True)
    print(f"  tr_dep: {tr_dep.shape} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    es_full = build_features(df_es_raw, df_tr_raw, lut, label="es")
    es_dep = es_full[es_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    te = es_dep["TAXITIME_SEC_mvt"].values
    es_dep = es_dep[(te > 0) & (te <= 10800)].reset_index(drop=True)
    print(f"  es_dep (11+12): {es_dep.shape} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    ho_full = build_features(df_ho_raw, df_tr_raw, lut, label="ho")
    ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    th = ho_dep["TAXITIME_SEC_mvt"].values
    ho_dep = ho_dep[(th > 0) & (th <= 10800)].reset_index(drop=True)
    print(f"  ho_dep (1+7): {ho_dep.shape} ({time.time()-t0:.0f}s)")

    X_tr = encode_X(tr_dep, FEATS71)
    X_es = encode_X(es_dep, FEATS71)
    X_ho = encode_X(ho_dep, FEATS71)
    y_tr = tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    y_es = es_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    print(f"  y_tr 分布: p50={np.median(y_tr):.0f}, p99={np.percentile(y_tr, 99):.0f}, "
          f"max={y_tr.max():.0f},  >1800={(y_tr>1800).sum():,}")

    # sample_weight
    median_y = float(np.median(y_tr))
    w_sqrt = np.sqrt(y_tr / median_y).astype(np.float32)
    w_lin = (y_tr / median_y).astype(np.float32)
    # 归一化: mean = 1.0
    w_sqrt /= w_sqrt.mean()
    w_lin /= w_lin.mean()
    print(f"  sample_weight: sqrt mean={w_sqrt.mean():.3f} max={w_sqrt.max():.3f}")
    print(f"                  lin  mean={w_lin.mean():.3f} max={w_lin.max():.3f}")

    # ============= A: L2 基线 =============
    print("\n" + "=" * 70)
    print("[A] L2 基线 (无 sample_weight, cap=8000, ES 11+12)")
    print("=" * 70)
    t0 = time.time()
    ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(np.float32),
                        categorical_feature=LOW_CARD_CATS)
    ds_es = lgb.Dataset(X_es, label=y_es.astype(np.float32),
                        categorical_feature=LOW_CARD_CATS, reference=ds_tr)
    b_A = lgb.train(dict(LGB_PARAMS_L2, seed=SEED), ds_tr,
                    num_boost_round=ROUNDS,
                    valid_sets=[ds_es],
                    callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False),
                               lgb.log_evaluation(500)])
    print(f"  A 训练 {time.time()-t0:.0f}s, best_iter={b_A.best_iteration}")
    p_A = np.clip(b_A.predict(X_ho).astype(np.float64), 30.0, 10800.0)
    p_A_raw = b_A.predict(X_ho).astype(np.float64)  # 用于 σpred
    p_A_nocap = np.clip(b_A.predict(X_ho).astype(np.float64), 30.0, np.inf)

    # ============= D1: w_sqrt, clip [30, 10800] =============
    print("\n" + "=" * 70)
    print("[D1] L2 + w_sqrt (clip [30, 10800])")
    print("=" * 70)
    t0 = time.time()
    ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(np.float32),
                        weight=w_sqrt, categorical_feature=LOW_CARD_CATS)
    ds_es = lgb.Dataset(X_es, label=y_es.astype(np.float32),
                        categorical_feature=LOW_CARD_CATS, reference=ds_tr)
    b_D1 = lgb.train(dict(LGB_PARAMS_L2, seed=SEED), ds_tr,
                     num_boost_round=ROUNDS,
                     valid_sets=[ds_es],
                     callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False),
                                lgb.log_evaluation(500)])
    print(f"  D1 训练 {time.time()-t0:.0f}s, best_iter={b_D1.best_iteration}")
    p_D1 = np.clip(b_D1.predict(X_ho).astype(np.float64), 30.0, 10800.0)

    # ============= D2: w_lin, clip [30, 10800] =============
    print("\n" + "=" * 70)
    print("[D2] L2 + w_lin (clip [30, 10800])")
    print("=" * 70)
    t0 = time.time()
    ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(np.float32),
                        weight=w_lin, categorical_feature=LOW_CARD_CATS)
    ds_es = lgb.Dataset(X_es, label=y_es.astype(np.float32),
                        categorical_feature=LOW_CARD_CATS, reference=ds_tr)
    b_D2 = lgb.train(dict(LGB_PARAMS_L2, seed=SEED), ds_tr,
                     num_boost_round=ROUNDS,
                     valid_sets=[ds_es],
                     callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False),
                                lgb.log_evaluation(500)])
    print(f"  D2 训练 {time.time()-t0:.0f}s, best_iter={b_D2.best_iteration}")
    p_D2 = np.clip(b_D2.predict(X_ho).astype(np.float64), 30.0, 10800.0)

    # ============= D3: D2 但放开上 clip =============
    p_D3 = np.clip(b_D2.predict(X_ho).astype(np.float64), 30.0, np.inf)

    # ============= E1: quantile alpha=0.5 =============
    print("\n" + "=" * 70)
    print("[E1] objective=quantile alpha=0.5")
    print("=" * 70)
    t0 = time.time()
    params_E1 = dict(LGB_PARAMS_L2)
    params_E1.update({"objective": "quantile", "metric": "quantile", "alpha": 0.5})
    ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(np.float32),
                        categorical_feature=LOW_CARD_CATS)
    ds_es = lgb.Dataset(X_es, label=y_es.astype(np.float32),
                        categorical_feature=LOW_CARD_CATS, reference=ds_tr)
    b_E1 = lgb.train(dict(params_E1, seed=SEED), ds_tr,
                     num_boost_round=ROUNDS,
                     valid_sets=[ds_es],
                     callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False),
                                lgb.log_evaluation(500)])
    print(f"  E1 训练 {time.time()-t0:.0f}s, best_iter={b_E1.best_iteration}")
    p_E1 = np.clip(b_E1.predict(X_ho).astype(np.float64), 30.0, 10800.0)

    # ============= E2: 9 分位 alpha=0.1..0.9, 分布均值 =============
    print("\n" + "=" * 70)
    print("[E2] 9 分位 (q=0.1..0.9), 重构分布均值")
    print("=" * 70)
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    E2_preds = []
    for q in quantiles:
        t0 = time.time()
        params_q = dict(LGB_PARAMS_L2)
        params_q.update({"objective": "quantile", "metric": "quantile", "alpha": q})
        ds_tr = lgb.Dataset(X_tr, label=y_tr.astype(np.float32),
                            categorical_feature=LOW_CARD_CATS)
        ds_es = lgb.Dataset(X_es, label=y_es.astype(np.float32),
                            categorical_feature=LOW_CARD_CATS, reference=ds_tr)
        b_q = lgb.train(dict(params_q, seed=SEED), ds_tr,
                        num_boost_round=ROUNDS,
                        valid_sets=[ds_es],
                        callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False),
                                   lgb.log_evaluation(0)])
        print(f"  q={q} 训练 {time.time()-t0:.0f}s, best_iter={b_q.best_iteration}")
        E2_preds.append(b_q.predict(X_ho).astype(np.float64))
    E2_raw = np.mean(np.stack(E2_preds), axis=0)
    p_E2 = np.clip(E2_raw, 30.0, 10800.0)

    # ============= 评估 =============
    print("\n" + "=" * 70)
    print("[评估] holdout (1+7)")
    print("=" * 70)
    aps = ho_dep["ADEP_mvt"].astype(str).values
    month_ho = pd.to_datetime(ho_dep["MVT_TIME_UTC_mvt"]).dt.month.values

    m_full = y_ho > 0
    m_core = (y_ho > 0) & (y_ho <= 10800)
    m_short = (y_ho > 0) & (y_ho <= 1800)
    m_long = (y_ho > 1800) & (y_ho <= 10800)
    m_long1800_3600 = (y_ho > 1800) & (y_ho <= 3600)
    m_long3600_10800 = (y_ho > 3600) & (y_ho <= 10800)
    m_extreme = y_ho > 10800  # ranking 上不可能有, holdout 上有 60 行

    preds = {"A": p_A, "D1": p_D1, "D2": p_D2, "D3": p_D3, "E1": p_E1, "E2": p_E2}

    def ev(p):
        return {
            "rmse_full": float(np.sqrt(np.mean((p[m_full] - y_ho[m_full]) ** 2))),
            "rmse_core": float(np.sqrt(np.mean((p[m_core] - y_ho[m_core]) ** 2))),
            "rmse_short": float(np.sqrt(np.mean((p[m_short] - y_ho[m_short]) ** 2))),
            "rmse_long": float(np.sqrt(np.mean((p[m_long] - y_ho[m_long]) ** 2))),
            "rmse_jan": float(np.sqrt(np.mean((p[(month_ho==1) & m_full] -
                                               y_ho[(month_ho==1) & m_full]) ** 2))),
            "rmse_jul": float(np.sqrt(np.mean((p[(month_ho==7) & m_full] -
                                               y_ho[(month_ho==7) & m_full]) ** 2))),
        }

    results = {k: ev(p) for k, p in preds.items()}
    print(f"\n  {'组':10s} {'短段':>8s} {'长段':>8s} {'核心':>8s} {'全样本':>8s} "
          f"{'Jan':>8s} {'Jul':>8s}")
    for k, lbl in [("A", "A_L2"), ("D1", "D1_sqrt"), ("D2", "D2_lin"),
                  ("D3", "D3_放开"), ("E1", "E1_q05"), ("E2", "E2_9q均值")]:
        r = results[k]
        print(f"  {lbl:10s} {r['rmse_short']:>8.2f} {r['rmse_long']:>8.2f} "
              f"{r['rmse_core']:>8.2f} {r['rmse_full']:>8.2f} "
              f"{r['rmse_jan']:>8.2f} {r['rmse_jul']:>8.2f}")

    # 分机场 (全样本 RMSE)
    print(f"\n  分机场 (全样本 RMSE)")
    print(f"  {'机场':6s} {'n':>6s} {'A':>8s} {'D1':>8s} {'D2':>8s} {'D3':>8s} "
          f"{'E1':>8s} {'E2':>8s}")
    ap_per = {}
    for ap in FOCUS_AP:
        mk = (aps == ap) & m_full
        if mk.sum() < 100:
            continue
        line = [ap, f"{mk.sum():>6d}"]
        for k in ["A", "D1", "D2", "D3", "E1", "E2"]:
            line.append(f"{float(np.sqrt(np.mean((preds[k][mk] - y_ho[mk]) ** 2))):>8.2f}")
        print(f"  {' '.join(line)}")
        ap_per[ap] = {k: float(np.sqrt(np.mean((preds[k][mk] - y_ho[mk]) ** 2)))
                      for k in ["A", "D1", "D2", "D3", "E1", "E2"]}

    # 长滑行档预测分布
    print(f"\n  长滑行档 (1800-3600 / 3600-10800) 预测均值 vs 真值:")
    print(f"  {'档':15s} {'n':>5s} {'true_mean':>10s} {'A_pred':>10s} {'D1':>10s} "
          f"{'D2':>10s} {'D3':>10s} {'E1':>10s} {'E2':>10s}")
    for lbl, m in [("1800-3600", m_long1800_3600), ("3600-10800", m_long3600_10800)]:
        if m.sum() == 0:
            continue
        y_m = y_ho[m]
        line = [lbl, f"{m.sum():>5d}", f"{y_m.mean():>10.1f}"]
        for k in ["A", "D1", "D2", "D3", "E1", "E2"]:
            line.append(f"{preds[k][m].mean():>10.1f}")
        print(f"  {' '.join(line)}")

    # σpred/σtrue (core 段, 衡量预测分布展宽)
    print(f"\n  σpred/σtrue (core 段):")
    print(f"    σ_true = {y_ho[m_core].std():.2f}s")
    for k in ["A", "D1", "D2", "D3", "E1", "E2"]:
        sigma = preds[k][m_core].std()
        ratio = sigma / y_ho[m_core].std()
        print(f"    {k}: σ_pred = {sigma:.2f}s, ratio = {ratio:.4f}")

    # >10800 极端行 (放开上 clip 的影响)
    print(f"\n  >10800 极端行 (n={m_extreme.sum()}):")
    if m_extreme.sum() > 0:
        y_ext = y_ho[m_extreme]
        print(f"    true range: [{y_ext.min():.0f}, {y_ext.max():.0f}], "
              f"mean={y_ext.mean():.0f}")
        for k in ["A", "D2", "D3"]:
            p_ext = preds[k][m_extreme]
            rmse_ext = float(np.sqrt(np.mean((p_ext - y_ext) ** 2)))
            bias_ext = float(np.mean(p_ext - y_ext))
            print(f"    {k}: pred range [{p_ext.min():.0f}, {p_ext.max():.0f}], "
                  f"mean={p_ext.mean():.0f}, rmse={rmse_ext:.0f}, bias={bias_ext:+.0f}")

    # A vs D2 全样本 σ分布
    print(f"\n  预测分布 (全 holdout):")
    for k in ["A", "D1", "D2", "D3", "E1", "E2"]:
        p = preds[k]
        print(f"    {k}: p50={np.median(p):.0f}, p99={np.percentile(p, 99):.0f}, "
              f"max={p.max():.0f}, n_pred>1800={(p>1800).sum():,}, "
              f"n_pred>3600={(p>3600).sum():,}")

    # ============= 决策 =============
    print(f"\n" + "=" * 70)
    print("[决策]")
    print("=" * 70)
    # 候选: D2 (lin) + 放开上 clip
    delta_D2_full = results["D2"]["rmse_full"] - results["A"]["rmse_full"]
    delta_D3_full = results["D3"]["rmse_full"] - results["A"]["rmse_full"]
    delta_E2_full = results["E2"]["rmse_full"] - results["A"]["rmse_full"]

    # 分机场同号
    same_sign_D2 = all(ap_per[k]["D2"] < ap_per[k]["A"] for k in ap_per)
    same_sign_D3 = all(ap_per[k]["D3"] < ap_per[k]["A"] for k in ap_per)
    same_sign_E2 = all(ap_per[k]["E2"] < ap_per[k]["A"] for k in ap_per)

    print(f"  全样本 RMSE Δ:")
    print(f"    D2_lin vs A: {delta_D2_full:+.2f}s ({'↓' if delta_D2_full<0 else '↑'})  "
          f"分机场同号? {same_sign_D2}")
    print(f"    D3_放开 vs A: {delta_D3_full:+.2f}s ({'↓' if delta_D3_full<0 else '↑'})  "
          f"分机场同号? {same_sign_D3}")
    print(f"    E2_9q均值 vs A: {delta_E2_full:+.2f}s ({'↓' if delta_E2_full<0 else '↑'})  "
          f"分机场同号? {same_sign_E2}")

    # Jan/Jul 同向
    jan_sign = (results["D2"]["rmse_jan"] < results["A"]["rmse_jan"]) == \
               (results["D2"]["rmse_jul"] < results["A"]["rmse_jul"])
    print(f"  Jan/Jul 同向 (D2 vs A): {jan_sign}")

    # 落 JSON
    out = {
        "task": "tail_loss_experiment_v1_single_seed",
        "split": "train=2025-02..12 (incl 11+12), holdout=2025-01+07",
        "seed": SEED,
        "rounds_cap": ROUNDS,
        "es_rounds": ES_ROUNDS,
        "n_train": int(len(tr_dep)),
        "n_es": int(len(es_dep)),
        "n_holdout": int(len(ho_dep)),
        "median_train_y": median_y,
        "weight_sqrt_max": float(w_sqrt.max()),
        "weight_lin_max": float(w_lin.max()),
        "results": results,
        "per_airport": ap_per,
        "decision": {
            "delta_D2_full": delta_D2_full,
            "delta_D3_full": delta_D3_full,
            "delta_E2_full": delta_E2_full,
            "same_sign_D2": same_sign_D2,
            "same_sign_D3": same_sign_D3,
            "same_sign_E2": same_sign_E2,
        },
        "total_time_sec": round(time.time() - t_start, 1),
    }
    fn = D0 / "tail_loss_experiment.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n  结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()