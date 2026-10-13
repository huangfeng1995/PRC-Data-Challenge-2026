"""
OOF 二阶校正器 v2 — 利用 v15 已训练好的 LGB booster (单大模型)
=============================================================
- 基模型: v15 LGB-only booster (已存在 models/lgb_full_v15/lgb_seed{42,123,2024}.txt)
  这是用 1-12 全量训练的, 但对 1+7 月存在泄漏. 因此:
  * 用 v15 booster 对 holdout (1+7) 推理 -> 这是 baseline
  * 用 v15 booster 对 train (2-12) 推理 -> 这是 OOF (有泄漏, 但校正器只学"输入→真值"规则)
- 二阶校正器 (轻量 LGB):
  输入 = [oof_pred, AOBT_NaT, hour, ADEP_int, t_unimp, P_long_proxy]
  目标 = 真值
  训练: train 2-12 OOF + 真值, 评估: holdout 1+7
- 严谨起见: 校正器训练数据排除 1+7 月
- 红线: 不改 label / 不 clip 真值 / 不建单机场模型
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
FOCUS_AP = ["LIRF", "EGLL", "LTFM", "LFPG", "EHAM"]
CAT_MAP = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
SEEDS = [42, 123, 2024]


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
    print("[OOF 二阶校正器 v2] 用 v15 LGB 已有 booster")
    print("=" * 70)

    # ============= 加载 =============
    MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                       if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
                  for m in range(1, 13)}
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    tr_dfs = [load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in train_months]
    df_tr_raw = pd.concat(tr_dfs, ignore_index=True)
    ho_dfs = [load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in [1, 7]]
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)

    df_tr_dep_all = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                              (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
                              (df_tr_raw["TAXITIME_SEC_mvt"] <= 10800)].copy()
    lut = L.build_t_unimp_lut(df_tr_dep_all)
    print(f"  训练 DEP: {len(df_tr_dep_all):,}")
    print("\n[1] 重建训练折特征...")
    t0 = time.time()
    tr_full = build_features(df_tr_raw, df_tr_raw, lut, label="tr")
    tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt = tr_dep["TAXITIME_SEC_mvt"].values
    tr_dep = tr_dep[(tt > 0) & (tt <= 10800)].reset_index(drop=True)
    print(f"  {tr_dep.shape} ({time.time()-t0:.0f}s)")

    print("[2] 重建 holdout 特征...")
    t0 = time.time()
    ho_full = build_features(df_ho_raw, df_tr_raw, lut, label="ho")
    ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tth = ho_dep["TAXITIME_SEC_mvt"].values
    ho_dep = ho_dep[(tth > 0) & (tth <= 10800)].reset_index(drop=True)
    print(f"  {ho_dep.shape} ({time.time()-t0:.0f}s)")

    X_tr = encode_X(tr_dep, FEATS71)
    X_ho = encode_X(ho_dep, FEATS71)
    y_tr = tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)
    y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)

    # ============= 用 v15 booster 推理 =============
    print("\n[3] 用 v15 LGB 已有 booster 推理 (无 FM, 仅 71 特征)")
    MODEL_DIR = ROOT / "models" / "lgb_full_v15"

    def predict_set(X):
        preds = []
        for sd in SEEDS:
            b = lgb.Booster(model_file=str(MODEL_DIR / f"lgb_seed{sd}.txt"))
            preds.append(b.predict(X).astype(np.float64))
        return np.mean(np.stack(preds), axis=0)

    print("  predict train (2-12)...")
    t0 = time.time()
    oof_tr_pred = predict_set(X_tr)
    print(f"    {time.time()-t0:.0f}s  RMSE(全 train, 含泄漏) = "
          f"{float(np.sqrt(np.mean((oof_tr_pred - y_tr) ** 2))):.2f}")
    print("  predict holdout (1+7)...")
    t0 = time.time()
    oof_ho_pred = predict_set(X_ho)
    print(f"    {time.time()-t0:.0f}s  RMSE(ho, 含泄漏) = "
          f"{float(np.sqrt(np.mean((oof_ho_pred - y_ho) ** 2))):.2f}")

    # ============= 二阶校正器 =============
    print("\n[4] 二阶校正器")
    # holdout 与 train 一起编码特征
    aobt_nat_tr = tr_dep["AOBT_3_flt"].isna().values.astype(np.int8)
    aobt_nat_ho = ho_dep["AOBT_3_flt"].isna().values.astype(np.int8)
    hour_tr = tr_dep["SCHED_TIME_UTC_mvt"].dt.hour.values.astype(np.int32)
    hour_ho = ho_dep["SCHED_TIME_UTC_mvt"].dt.hour.values.astype(np.int32)
    adep_tr = tr_dep["ADEP_mvt"].astype(str).map(CAT_MAP.get("ADEP_mvt", {})).fillna(-1).astype("int32").values
    adep_ho = ho_dep["ADEP_mvt"].astype(str).map(CAT_MAP.get("ADEP_mvt", {})).fillna(-1).astype("int32").values
    tunimp_tr = tr_dep["t_unimp_r7"].fillna(800).astype(np.float32).values
    tunimp_ho = ho_dep["t_unimp_r7"].fillna(800).astype(np.float32).values
    plung_tr = (oof_tr_pred > 1800).astype(np.float32)
    plung_ho = (oof_ho_pred > 1800).astype(np.float32)

    def build_Z(pred, aobt_nat, hour, adep, tunimp, plung):
        return np.column_stack([
            pred.astype(np.float32),
            aobt_nat.astype(np.float32),
            hour.astype(np.float32),
            adep.astype(np.float32),
            tunimp,
            (pred - tunimp).astype(np.float32),
            plung,
        ]).astype(np.float32)

    Z_tr_corr = build_Z(oof_tr_pred, aobt_nat_tr, hour_tr, adep_tr, tunimp_tr, plung_tr)
    Z_ho_corr = build_Z(oof_ho_pred, aobt_nat_ho, hour_ho, adep_ho, tunimp_ho, plung_ho)
    Z_cols = ["oof_pred", "aobt_nat", "hour", "ADEP_int", "t_unimp", "oof_minus_tunimp", "P_long_proxy"]
    print(f"  Z_tr shape: {Z_tr_corr.shape}, Z_ho shape: {Z_ho_corr.shape}, cols: {Z_cols}")

    LGB_CORR_PARAMS = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.02, "num_leaves": 31, "max_depth": 5,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "reg_alpha": 1.0, "reg_lambda": 1.0,
        "verbose": -1, "num_threads": -1,
    }

    corr_preds, corr_per_seed = [], []
    for sd in SEEDS:
        ds = lgb.Dataset(Z_tr_corr, label=y_tr.astype(np.float32))
        b = lgb.train(dict(LGB_CORR_PARAMS, seed=sd), ds, num_boost_round=500)
        p = b.predict(Z_ho_corr).astype(np.float64)
        p = np.clip(p, 30.0, 10800.0)
        corr_preds.append(p)
        corr_per_seed.append(float(np.sqrt(np.mean((p - y_ho) ** 2))))
        print(f"    校正器 seed={sd} RMSE={corr_per_seed[-1]:.2f}")
    corr_avg = np.mean(np.stack(corr_preds), axis=0)
    corr_std = float(np.std(corr_per_seed))

    # 校正器训练集的 in-sample 表现 (参照)
    corr_in = b.predict(Z_tr_corr)
    print(f"  校正器 in-sample RMSE = {float(np.sqrt(np.mean((corr_in - y_tr) ** 2))):.2f}")

    # ============= 评估 =============
    print("\n[5] 评估 holdout (1+7)")
    m_full = y_ho > 0
    m_core = (y_ho > 0) & (y_ho <= 10800)
    m_short = (y_ho > 0) & (y_ho <= 1800)
    m_long = (y_ho > 1800) & (y_ho <= 10800)
    aps_ho = ho_dep["ADEP_mvt"].astype(str).values

    def ev(p, lbl):
        return {
            "label": lbl,
            "rmse_full": float(np.sqrt(np.mean((p[m_full] - y_ho[m_full]) ** 2))),
            "rmse_core": float(np.sqrt(np.mean((p[m_core] - y_ho[m_core]) ** 2))),
            "rmse_short": float(np.sqrt(np.mean((p[m_short] - y_ho[m_short]) ** 2))),
            "rmse_long": float(np.sqrt(np.mean((p[m_long] - y_ho[m_long]) ** 2))),
        }

    base = ev(oof_ho_pred, "base_v15_lgb")
    final = ev(corr_avg, "v15_+_corrector")
    print(f"  {'组':22s} {'短段':>9s} {'长段':>9s} {'核心':>9s} {'全样本':>9s}")
    for k, lbl in [("base_v15_lgb", "①基模型(v15 LGB)"), ("v15_+_corrector", "②+二阶校正")]:
        r = {"base_v15_lgb": base, "v15_+_corrector": final}[k]
        print(f"  {lbl:22s} {r['rmse_short']:>9.2f} {r['rmse_long']:>9.2f} "
              f"{r['rmse_core']:>9.2f} {r['rmse_full']:>9.2f}")

    # 分机场
    print(f"\n  重点 5 机场 (全样本 RMSE)")
    print(f"  {'机场':6s} {'n':>8s} {'①基':>9s} {'②+校':>9s} {'Δ':>8s}")
    ap_per = {}
    for ap in FOCUS_AP:
        mk = (aps_ho == ap) & m_full
        if mk.sum() < 100:
            continue
        rb = float(np.sqrt(np.mean((oof_ho_pred[mk] - y_ho[mk]) ** 2)))
        rc = float(np.sqrt(np.mean((corr_avg[mk] - y_ho[mk]) ** 2)))
        ap_per[ap] = {"n": int(mk.sum()), "base": rb, "corr": rc, "delta": rc - rb}
        print(f"  {ap:6s} {mk.sum():>8d} {rb:>9.2f} {rc:>9.2f} {rc-rb:>+8.2f}")

    delta_core = final["rmse_core"] - base["rmse_core"]
    same_sign = all(ap_per[ap]["delta"] < 0 for ap in ap_per)
    pass_step = (delta_core < -0.5) and same_sign
    print(f"\n[判断]")
    print(f"  核心 Δ(②-①) = {delta_core:+.2f}s, seedσ={corr_std:.2f}s")
    print(f"  重点 5 机场 Δ 同号? {same_sign}")
    print(f"  通过? {pass_step}")

    # ============= JSON =============
    out = {
        "task": "oof_2nd_stage_calibrator_v2_v15_lgb",
        "base_model": "v15 LGB-only 71 特征 × 3 seed (train 1-12 全量, holdout 1+7 有泄漏)",
        "n_train_oof": int(len(tr_dep)),
        "n_holdout": int(len(ho_dep)),
        "results": {
            "base_v15_lgb": {**base, "rmse_core_note": "泄漏乐观值, 仅作校准器输入参考"},
            "v15_+_corrector": final,
            "corrector_per_seed_rmse": [round(v, 3) for v in corr_per_seed],
        },
        "per_airport": ap_per,
        "decision": {
            "delta_core_rmse": delta_core,
            "seed_jitter": corr_std,
            "same_sign_airports": same_sign,
            "pass": pass_step,
        },
        "total_time_sec": round(time.time() - t_start, 1),
    }
    fn = D0 / "oof_calibrate_step2.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n  结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()