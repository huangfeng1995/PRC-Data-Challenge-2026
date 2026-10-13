"""
Step1 重训验证 — rwy_k / rwy_s / rwy_ks + 机制和 t_unimp + k*s
==========================================================
- split: train 2025-02..12, val 2025-01+07
- 模型: LGB 3seed 3000rounds 单阶段
- 三组对照:
  ① 71 特征基线 (短段 252 / 核心 309)
  ② 71 + k,s,k×s
  ③ 71 + offset + k,s,k×s
- 额外: t_unimp + k*s 作为强特征组合
- 排除: 极端长滑行不放回 (TX>10800 仍剔, 不动 label)
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
STEP1_FEATS = ["t_unimp_bias_offset", "t_unimp_bias_offset_known"]
QUEUE_FEATS = ["rwy_k", "rwy_s", "rwy_ks", "t_unimp_plus_ks"]

LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]

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
ROUNDS = int(os.environ.get("ROUNDS", 3000))
OFFSET_FOLDS = 5
OFFSET_SHRINK = 20.0
OFFSET_MIN_OBS = 20
OFFSET_CLIP = (-900.0, 2400.0)
FOCUS_AP = ["LIRF", "EGLL", "LTFM", "LFPG", "EHAM"]

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


def fit_offset_map(tr_df, global_off):
    rng = np.random.RandomState(0)
    fold = rng.randint(0, OFFSET_FOLDS, size=len(tr_df))
    base = pd.DataFrame({
        "ADEP_mvt": tr_df["ADEP_mvt"].astype(str).values,
        "RUNWAY_mvt": tr_df["RUNWAY_mvt"].astype(str).values,
        "y": tr_df["TAXITIME_SEC_mvt"].values.astype(np.float64),
        "t": tr_df["t_unimp_r7"].values.astype(np.float64),
        "fold": fold,
    })

    def _fit(sub):
        med = sub.groupby(["ADEP_mvt", "RUNWAY_mvt"]).agg(
            med_y=("y", "median"), med_t=("t", "median"), n=("y", "size"))
        med = med[med["n"] >= OFFSET_MIN_OBS]
        off = (med["med_y"] - med["med_t"]).clip(*OFFSET_CLIP)
        med["off"] = (off * med["n"] + global_off * OFFSET_SHRINK) / (med["n"] + OFFSET_SHRINK)
        return med[["off"]].reset_index()

    out = np.full(len(tr_df), global_off, dtype=np.float64)
    known = np.zeros(len(tr_df), dtype=np.int8)
    for f in range(OFFSET_FOLDS):
        med = _fit(base[base["fold"] != f])
        pos = np.where(fold == f)[0]
        v = base.iloc[pos][["ADEP_mvt", "RUNWAY_mvt"]].merge(
            med, how="left", on=["ADEP_mvt", "RUNWAY_mvt"])["off"].to_numpy()
        out[pos] = np.where(np.isnan(v), global_off, np.nan_to_num(v))
        known[pos] = (~np.isnan(v)).astype(np.int8)
    return out.astype(np.float32), known.astype(np.int8), _fit(base)


def apply_offset_map(keys_df, fit_map, global_off):
    v = keys_df.merge(fit_map, how="left", on=["ADEP_mvt", "RUNWAY_mvt"])["off"].to_numpy()
    return np.where(np.isnan(v), global_off, np.nan_to_num(v)).astype(np.float32), \
        (~np.isnan(v)).astype(np.int8)


def add_rwy_queue(df, thr_idx, global_s):
    """按 (ADEP, RUNWAY) 分组, 算 k_new = #{j: AOBT_j < AOBT_i ∧ MVT_j > t_rwy_i}
    并查服务间隔 s. 同时产出 t_unimp + k*s.
    """
    iNaT = np.iinfo(np.int64).min
    aobt_ns = df["AOBT_3_flt"].dt.tz_localize(None).astype("datetime64[ns]").view("int64").values
    mvt_ns = df["MVT_TIME_UTC_mvt"].dt.tz_localize(None).astype("datetime64[ns]").view("int64").values
    aobt_valid = ~np.isnan(df["AOBT_3_flt"].values) & (df["AOBT_3_flt"].notna().values)
    t_unimp = df["t_unimp_r7"].values.astype(np.float64)
    t_filled = np.where(np.isnan(t_unimp), float(np.nanmedian(t_unimp)), t_unimp)
    t_rwy_ns = (aobt_ns.astype(np.float64) + t_filled.astype(np.float64) * 1e9).astype(np.int64)

    n = len(df)
    k_arr = np.zeros(n, dtype=np.int32)
    adep = df["ADEP_mvt"].astype(str).values
    runway = df["RUNWAY_mvt"].astype(str).values
    grp = pd.DataFrame({
        "ADEP_mvt": adep, "RUNWAY_mvt": runway,
        "aobt_ns": aobt_ns, "mvt_ns": mvt_ns, "t_rwy_ns": t_rwy_ns,
        "orig_idx": np.arange(n),
    }).groupby(["ADEP_mvt", "RUNWAY_mvt"], sort=False, observed=True)
    for (ap, rw), sub in grp:
        idx = sub["orig_idx"].values
        a = sub["aobt_ns"].values.copy()
        m = sub["mvt_ns"].values.copy()
        t_rwy = sub["t_rwy_ns"].values.copy()
        order = np.argsort(a, kind="mergesort")
        a_s = a[order]; m_s = m[order]; t_rwy_s = t_rwy[order]
        cnt_le = np.searchsorted(a_s, a_s, side="right")
        for jj in range(len(order)):
            k_arr[idx[order[jj]]] = int((m_s[:cnt_le[jj]] > t_rwy_s[jj]).sum())

    # 服务间隔
    df["hour_v"] = df["SCHED_TIME_UTC_mvt"].dt.hour.values
    s_arr = np.full(n, global_s, dtype=np.float64)
    for i in range(n):
        v = thr_idx.get((adep[i], runway[i], int(df["hour_v"].iloc[i])))
        if v is not None:
            s_arr[i] = float(v)
    ks_arr = (k_arr * s_arr).astype(np.float32)
    df["rwy_k"] = k_arr
    df["rwy_s"] = s_arr.astype(np.float32)
    df["rwy_ks"] = ks_arr
    df["t_unimp_plus_ks"] = (t_unimp + ks_arr).astype(np.float32)
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


def train_predict(X_tr, y_tr, X_ho, y_ho, label):
    preds, per_seed = [], []
    for sd in SEEDS:
        ds = lgb.Dataset(X_tr, label=y_tr.astype(np.float32), categorical_feature=LOW_CARD_CATS)
        b = lgb.train(dict(LGB_PARAMS, seed=sd), ds, num_boost_round=ROUNDS)
        p = np.clip(b.predict(X_ho).astype(np.float64), 30.0, 10800.0)
        preds.append(p); per_seed.append(float(np.sqrt(np.mean((p - y_ho) ** 2))))
        print(f"      [{label}] seed={sd} RMSE={per_seed[-1]:.2f} ({time.strftime('%H:%M:%S')})", flush=True)
    avg = np.mean(np.stack(preds), axis=0)
    return avg, per_seed


def main():
    t_start = time.time()
    print("=" * 70)
    print("[Step1 跑道排队机制验证] split=2025-02..12/2025-01+07  3 seed 3000 rounds")
    print("=" * 70)
    print(f"  三组对照: ①71  ②71+k,s,k*s  ③71+offset+k,s,k*s (+ t_unimp+k*s)")

    # 加载
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    tr_dfs = []
    for m in train_months:
        fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
              if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
        tr_dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_tr_raw = pd.concat(tr_dfs, ignore_index=True)
    ho_dfs = [load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet",
                         columns=TRAIN_COLS) for m in [1, 7]]
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)

    # 训练折 DEP (剔 TX<=0 或 >10800, 标准口径, 不放回极端行)
    df_tr_dep = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP")].copy()
    tt_tr = df_tr_dep["TAXITIME_SEC_mvt"].values
    keep_tr = (tt_tr > 0) & (tt_tr <= 10800)
    df_tr_dep = df_tr_dep[keep_tr].copy().reset_index(drop=True)
    print(f"\n[1] 训练 DEP (TX>0 且 ≤10800): {len(df_tr_dep):,}")

    df_ho_dep = df_ho_raw[(df_ho_raw["PHASE_mvt"] == "DEP")].copy()
    tt_ho = df_ho_dep["TAXITIME_SEC_mvt"].values
    keep_ho = (tt_ho > 0) & (tt_ho <= 10800)
    df_ho_dep = df_ho_dep[keep_ho].copy().reset_index(drop=True)
    print(f"  holdout DEP (TX>0 且 ≤10800): {len(df_ho_dep):,}")

    # 训练折 throughput (ADEP, RUNWAY, hour)
    df_tr_dep_thr = parse_time_columns(df_tr_dep.copy(), cols=TIME_COLS)
    df_tr_dep_thr["hour"] = df_tr_dep_thr["SCHED_TIME_UTC_mvt"].dt.hour
    thr = (df_tr_dep_thr.groupby(["ADEP_mvt", "RUNWAY_mvt", "hour"])
                       .size().rename("cnt").reset_index())
    thr["s"] = 3600.0 / thr["cnt"]
    global_s = 3600.0 / (len(df_tr_dep_thr) / (10 * 30 * 24))
    thr_idx = thr.set_index(["ADEP_mvt", "RUNWAY_mvt", "hour"])["s"].to_dict()
    print(f"\n[2] throughput 组数: {len(thr):,}, global_s={global_s:.1f}s")

    # 重建特征
    print("\n[3] 重建训练折特征...")
    t0 = time.time()
    lut = L.build_t_unimp_lut(df_tr_dep)
    tr_full = build_features(df_tr_raw, df_tr_raw, lut, label="tr")
    tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    ttt = tr_dep["TAXITIME_SEC_mvt"].values
    tr_dep = tr_dep[(ttt > 0) & (ttt <= 10800)].reset_index(drop=True)
    print(f"  tr_dep: {tr_dep.shape}  ({time.time()-t0:.0f}s)")

    print("[3] 重建 holdout 特征...")
    t0 = time.time()
    ho_full = build_features(df_ho_raw, df_tr_raw, lut, label="ho")
    ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tth = ho_dep["TAXITIME_SEC_mvt"].values
    ho_dep = ho_dep[(tth > 0) & (tth <= 10800)].reset_index(drop=True)
    print(f"  ho_dep: {ho_dep.shape}  ({time.time()-t0:.0f}s)")

    # offset
    print("\n[4] t_unimp_bias_offset (5-fold OOF)")
    global_off = float(np.median(tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)) -
                       np.median(tr_dep["t_unimp_r7"].values.astype(np.float64)))
    tr_off, tr_known, full_map = fit_offset_map(tr_dep, global_off)
    tr_dep["t_unimp_bias_offset"] = tr_off
    tr_dep["t_unimp_bias_offset_known"] = tr_known
    rk_off_vals, rk_known_vals = apply_offset_map(pd.DataFrame({
        "ADEP_mvt": ho_dep["ADEP_mvt"].astype(str).values,
        "RUNWAY_mvt": ho_dep["RUNWAY_mvt"].astype(str).values,
    }), full_map, global_off)
    ho_dep["t_unimp_bias_offset"] = rk_off_vals
    ho_dep["t_unimp_bias_offset_known"] = rk_known_vals
    print(f"  global_offset={global_off:+.1f}s  tr_cov={tr_known.mean()*100:.1f}%  "
          f"ho_cov={rk_known_vals.mean()*100:.1f}%  groups={len(full_map)}")

    # rwy queue
    print("\n[5] rwy_k / rwy_s / rwy_ks / t_unimp_plus_ks")
    tr_dep = add_rwy_queue(tr_dep, thr_idx, global_s)
    ho_dep = add_rwy_queue(ho_dep, thr_idx, global_s)
    print(f"  train rwy_k 分布: "
          f"0={(tr_dep['rwy_k']==0).sum():,}, 1={(tr_dep['rwy_k']==1).sum():,}, "
          f"2={(tr_dep['rwy_k']==2).sum():,}, 3+={(tr_dep['rwy_k']>=3).sum():,}")
    print(f"  holdout rwy_k 分布: "
          f"0={(ho_dep['rwy_k']==0).sum():,}, 1={(ho_dep['rwy_k']==1).sum():,}, "
          f"2={(ho_dep['rwy_k']==2).sum():,}, 3+={(ho_dep['rwy_k']>=3).sum():,}")

    # 编码
    FEATS_A = FEATS71
    FEATS_B = FEATS71 + QUEUE_FEATS
    FEATS_C = FEATS71 + STEP1_FEATS + QUEUE_FEATS
    X_tr_A, X_tr_B, X_tr_C = (encode_X(tr_dep, FEATS_A),
                              encode_X(tr_dep, FEATS_B),
                              encode_X(tr_dep, FEATS_C))
    X_ho_A, X_ho_B, X_ho_C = (encode_X(ho_dep, FEATS_A),
                              encode_X(ho_dep, FEATS_B),
                              encode_X(ho_dep, FEATS_C))
    y_tr = tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    print(f"\n[6] 训练矩阵: A {X_tr_A.shape}, B {X_tr_B.shape}, C {X_tr_C.shape}")

    m_full = y_ho > 0
    m_core = (y_ho > 0) & (y_ho <= 10800)
    m_short = (y_ho > 0) & (y_ho <= 1800)
    m_long = (y_ho > 1800) & (y_ho <= 10800)
    aps_ho = ho_dep["ADEP_mvt"].astype(str).values

    def eval_pred(p, label):
        return {
            "label": label,
            "rmse_short": float(np.sqrt(np.mean((p[m_short] - y_ho[m_short]) ** 2))) if m_short.sum() else None,
            "rmse_long": float(np.sqrt(np.mean((p[m_long] - y_ho[m_long]) ** 2))) if m_long.sum() else None,
            "rmse_core": float(np.sqrt(np.mean((p[m_core] - y_ho[m_core]) ** 2))) if m_core.sum() else None,
            "rmse_full": float(np.sqrt(np.mean((p[m_full] - y_ho[m_full]) ** 2))) if m_full.sum() else None,
        }

    print("\n[7] ① 71 特征基线")
    t0 = time.time()
    pred_a, seeds_a = train_predict(X_tr_A, y_tr, X_ho_A, y_ho, "A_71")
    a_eval = eval_pred(pred_a, "A_71")
    a_eval["seed_std"] = float(np.std(seeds_a))
    print(f"  A done {time.time()-t0:.0f}s  full={a_eval['rmse_full']:.2f}  "
          f"core={a_eval['rmse_core']:.2f}  short={a_eval['rmse_short']:.2f}")

    print("\n[8] ② 71 + k,s,k*s, t_unimp+k*s")
    t0 = time.time()
    pred_b, seeds_b = train_predict(X_tr_B, y_tr, X_ho_B, y_ho, "B_+queue")
    b_eval = eval_pred(pred_b, "B_+queue")
    b_eval["seed_std"] = float(np.std(seeds_b))
    print(f"  B done {time.time()-t0:.0f}s  full={b_eval['rmse_full']:.2f}  "
          f"core={b_eval['rmse_core']:.2f}  short={b_eval['rmse_short']:.2f}")

    print("\n[9] ③ 71 + offset + k,s,k*s")
    t0 = time.time()
    pred_c, seeds_c = train_predict(X_tr_C, y_tr, X_ho_C, y_ho, "C_+off+queue")
    c_eval = eval_pred(pred_c, "C_+off+queue")
    c_eval["seed_std"] = float(np.std(seeds_c))
    print(f"  C done {time.time()-t0:.0f}s  full={c_eval['rmse_full']:.2f}  "
          f"core={c_eval['rmse_core']:.2f}  short={c_eval['rmse_short']:.2f}")

    # 分机场
    print("\n[10] 分机场 (全样本 RMSE)")
    print(f"  {'机场':6s} {'n':>8s} {'①71':>10s} {'②+queue':>10s} {'③+off+queue':>13s} "
          f"{'Δ②-①':>10s} {'Δ③-①':>10s}")
    ap_per = {}
    for ap in FOCUS_AP:
        mk = (aps_ho == ap) & m_full
        if mk.sum() < 100:
            continue
        ra = float(np.sqrt(np.mean((pred_a[mk] - y_ho[mk]) ** 2)))
        rb = float(np.sqrt(np.mean((pred_b[mk] - y_ho[mk]) ** 2)))
        rc = float(np.sqrt(np.mean((pred_c[mk] - y_ho[mk]) ** 2)))
        ap_per[ap] = {"n": int(mk.sum()), "A": ra, "B": rb, "C": rc,
                      "delta_BA": rb - ra, "delta_CA": rc - ra}
        print(f"  {ap:6s} {mk.sum():>8d} {ra:>10.2f} {rb:>10.2f} {rc:>13.2f} "
              f"{rb - ra:>+10.2f} {rc - ra:>+10.2f}")

    # 对照表
    print("\n" + "=" * 70)
    print("[对照表]")
    print("=" * 70)
    print(f"  {'组':20s} {'短段':>9s} {'长段':>9s} {'核心':>9s} {'全样本':>9s} {'seedσ':>7s}")
    for k, lbl in [("A_71", "①71基线"), ("B_+queue", "②+queue"),
                   ("C_+off+queue", "③+offset+queue")]:
        e = {"A_71": a_eval, "B_+queue": b_eval, "C_+off+queue": c_eval}[k]
        print(f"  {lbl:20s} {e['rmse_short']:>9.2f} {e['rmse_long']:>9.2f} "
              f"{e['rmse_core']:>9.2f} {e['rmse_full']:>9.2f} {e['seed_std']:>7.2f}")

    # 决策
    seed_jitter = max(a_eval["seed_std"], b_eval["seed_std"], c_eval["seed_std"])
    delta_short = b_eval["rmse_short"] - a_eval["rmse_short"]
    same_sign_ap = all(ap_per[ap]["delta_BA"] < 0 for ap in ap_per)
    pass_step1 = (delta_short < -0.5) and same_sign_ap
    print(f"\n[判断]")
    print(f"  短段 Δ(②-①) = {delta_short:+.2f}s,  seedσ={seed_jitter:.2f}s  "
          f"({'显著下降' if delta_short < -seed_jitter else '未达显著' if delta_short < -0.5 else '上升'})")
    print(f"  重点 5 机场 Δ(②-①) 同号? {same_sign_ap}")
    print(f"  通过 Step1? {pass_step1}")
    if pass_step1:
        print(f"  → 进入 Step2 (ranking 推理)")
    else:
        print(f"  → 停止, 排队方向已通过 Step0 但 Step1 未达到显著门槛")

    out = {
        "task": "rwy_queue_step1",
        "split": "train=2025-02..12 (剔TX>10800) val=2025-01+07",
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "n_train": int(len(tr_dep)),
        "n_holdout": int(len(ho_dep)),
        "global_offset_s": global_off,
        "rwy_k_train_dist": {
            "k=0": int((tr_dep["rwy_k"] == 0).sum()),
            "k=1": int((tr_dep["rwy_k"] == 1).sum()),
            "k=2": int((tr_dep["rwy_k"] == 2).sum()),
            "k=3+": int((tr_dep["rwy_k"] >= 3).sum()),
        },
        "rwy_k_holdout_dist": {
            "k=0": int((ho_dep["rwy_k"] == 0).sum()),
            "k=1": int((ho_dep["rwy_k"] == 1).sum()),
            "k=2": int((ho_dep["rwy_k"] == 2).sum()),
            "k=3+": int((ho_dep["rwy_k"] >= 3).sum()),
        },
        "results": {
            "A_71": {"seeds_rmse": [round(v, 3) for v in seeds_a], **a_eval},
            "B_+queue": {"seeds_rmse": [round(v, 3) for v in seeds_b], **b_eval},
            "C_+off+queue": {"seeds_rmse": [round(v, 3) for v in seeds_c], **c_eval},
        },
        "per_airport": ap_per,
        "decision": {
            "delta_short_B_minus_A": delta_short,
            "seed_jitter": seed_jitter,
            "same_sign_airports": same_sign_ap,
            "pass_step1": pass_step1,
        },
        "total_time_sec": round(time.time() - t_start, 1),
    }
    fn = D0 / "rwy_queue_step1.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n  结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()