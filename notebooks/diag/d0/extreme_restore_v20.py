"""
放回极端长滑行 (TAXITIME>10800) 训练 — Step1 holdout 验证
=========================================================
- split: train 2025-02..12, val 2025-01+07
- 训练: 仅剔 TAXITIME<=0, 保留 >10800 极端长滑行
- 模型: LGB 3 seed 单阶段
- 对照: A=71特征基线, B=71+offset
- 推理: clip [30, 144000]
- 红线: 不删极端行 / 不建单机场模型 / 单阶段

训练超参: 与 lgb_full_train.py (v15 base) 完全一致
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
    """Step1: train 内 5-fold OOF 偏移, 返回 (tr_off, tr_known, fit_map)"""
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
    full_map = _fit(base)
    return out.astype(np.float32), known.astype(np.int8), full_map


def apply_offset_map(keys_df, fit_map, global_off):
    v = keys_df.merge(fit_map, how="left", on=["ADEP_mvt", "RUNWAY_mvt"])["off"].to_numpy()
    return np.where(np.isnan(v), global_off, np.nan_to_num(v)).astype(np.float32), \
        (~np.isnan(v)).astype(np.int8)


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


def train_predict(X_tr, y_tr, X_ho, y_ho, label, clip_hi=144000.0):
    preds, per_seed = [], []
    for sd in SEEDS:
        ds = lgb.Dataset(X_tr, label=y_tr.astype(np.float32), categorical_feature=LOW_CARD_CATS)
        b = lgb.train(dict(LGB_PARAMS, seed=sd), ds, num_boost_round=ROUNDS)
        p = b.predict(X_ho).astype(np.float64)
        p = np.clip(p, 30.0, clip_hi)
        preds.append(p); per_seed.append(float(np.sqrt(np.mean((p - y_ho) ** 2))))
        print(f"      [{label}] seed={sd} RMSE={per_seed[-1]:.2f} ({time.strftime('%H:%M:%S')})", flush=True)
    avg = np.mean(np.stack(preds), axis=0)
    return avg, per_seed


# typo guard
RMSE = {}


def main():
    t_start = time.time()
    print("=" * 70)
    print("[Step1 放回极端长滑行训练验证] holdout = 2025-01+07, train=2025-02..12")
    print("=" * 70)
    print(f"  seeds={SEEDS} rounds={ROUNDS} clip_hi=144000  (不放回TAXITIME<=0, 留TX>10800)")
    print()

    # ============= 加载 =============
    print("[1] 加载 2025-02..12 (训练) + 2025-01+07 (holdout)")
    train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    tr_dfs = []
    for m in train_months:
        fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
              if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
        tr_dfs.append(load_month(fn, columns=TRAIN_COLS))
    df_tr_raw = pd.concat(tr_dfs, ignore_index=True)
    ho_dfs = []
    for m in [1, 7]:
        ho_dfs.append(load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet",
                                 columns=TRAIN_COLS))
    df_ho_raw = pd.concat(ho_dfs, ignore_index=True)
    print(f"  train_raw={len(df_tr_raw):,}  holdout_raw={len(df_ho_raw):,}")

    # ============= 训练折: 仅剔 TX<=0  (保留 TX>10800) =============
    df_tr_dep_raw = df_tr_raw[df_tr_raw["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt_tr = df_tr_dep_raw["TAXITIME_SEC_mvt"].values
    keep_tr = tt_tr > 0
    df_tr_dep = df_tr_dep_raw[keep_tr].copy().reset_index(drop=True)
    n_extreme = int((df_tr_dep["TAXITIME_SEC_mvt"].values > 10800).sum())
    print(f"\n[2] 训练 DEP (TX>0, 含 TX>10800): {len(df_tr_dep):,}  "
          f"(剔 TX<=0: {(~keep_tr).sum():,}, 保留 TX>10800: {n_extreme})")

    # holdout 同样剔 TX<=0 (无效观测)
    df_ho_dep_raw = df_ho_raw[df_ho_raw["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt_ho_raw = df_ho_dep_raw["TAXITIME_SEC_mvt"].values
    keep_ho = tt_ho_raw > 0
    df_ho_dep = df_ho_dep_raw[keep_ho].copy().reset_index(drop=True)
    n_extreme_ho = int((df_ho_dep["TAXITIME_SEC_mvt"].values > 10800).sum())
    print(f"  holdout DEP (TX>0, 含 TX>10800): {len(df_ho_dep):,}  "
          f"(剔 TX<=0: {(~keep_ho).sum():,}, 保留 TX>10800: {n_extreme_ho})")

    # ============= LUT (fit on训) =============
    lut = L.build_t_unimp_lut(df_tr_dep)
    print(f"\n  LUT L1={len(lut['L1_q'])}, L2={len(lut['L2_q'])}, L3={len(lut['L3_q'])}, "
          f"global_q30={lut['global_q30']:.0f}s")

    # ============= 重建特征 =============
    print("\n[3] 重建训练折特征...")
    t0 = time.time()
    tr_full = build_features(df_tr_raw, df_tr_raw, lut, label="tr")
    tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt2 = tr_dep["TAXITIME_SEC_mvt"].values
    tr_dep = tr_dep[tt2 > 0].reset_index(drop=True)  # 同清洗
    print(f"  train DEP 全套 (TX>0): {tr_dep.shape}  ({time.time()-t0:.0f}s)")

    print("[3] 重建 holdout 特征...")
    t0 = time.time()
    ho_full = build_features(df_ho_raw, df_tr_raw, lut, label="ho")
    ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tth = ho_dep["TAXITIME_SEC_mvt"].values
    ho_dep = ho_dep[tth > 0].reset_index(drop=True)
    print(f"  holdout DEP 全套 (TX>0): {ho_dep.shape}  ({time.time()-t0:.0f}s)")

    # ============= offset =============
    print("\n[4] t_unimp_bias_offset (train 5-fold OOF, holdout 用 train 全量拟合)")
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

    # ============= 编码 =============
    print("\n[5] 编码 + 矩阵对齐")
    FEATS_A = FEATS71
    FEATS_B = FEATS71 + STEP1_FEATS
    X_tr_A, X_tr_B = encode_X(tr_dep, FEATS_A), encode_X(tr_dep, FEATS_B)
    X_ho_A, X_ho_B = encode_X(ho_dep, FEATS_A), encode_X(ho_dep, FEATS_B)
    y_tr = tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    t_ho = ho_dep["t_unimp_r7"].values.astype(np.float64)
    aobt_nat_ho = ho_dep["AOBT_3_flt"].isna().values
    aps_ho = ho_dep["ADEP_mvt"].astype(str).values
    print(f"  X_tr_A shape: {X_tr_A.shape}")
    print(f"  X_tr_B shape: {X_tr_B.shape}")
    print(f"  X_ho shape:   {X_ho_A.shape}")
    print(f"  holdout m_full (TX>0):    {(y_ho > 0).sum():,}")
    print(f"  holdout m_core (TX<=10800):{((y_ho > 0) & (y_ho <= 10800)).sum():,}")
    print(f"  holdout m_extreme (>10800):{(y_ho > 10800).sum():,}")
    print(f"  holdout AOBT_NaT:         {aobt_nat_ho.sum():,} ({aobt_nat_ho.mean()*100:.2f}%)")

    m_full = y_ho > 0
    m_core = (y_ho > 0) & (y_ho <= 10800)

    def eval_pred(p, label):
        rmse_full = float(np.sqrt(np.mean((p[m_full] - y_ho[m_full]) ** 2)))
        rmse_core = float(np.sqrt(np.mean((p[m_core] - y_ho[m_core]) ** 2))) if m_core.sum() else None
        # AOBT_NaT 段
        mt = aobt_nat_ho & m_full
        rmse_t = float(np.sqrt(np.mean((p[mt] - y_ho[mt]) ** 2))) if mt.sum() else None
        bias_t = float(np.mean(p[mt] - y_ho[mt])) if mt.sum() else None
        sse_t = float(np.sum((p[mt] - y_ho[mt]) ** 2)) if mt.sum() else 0.0
        sse_all = float(np.sum((p[m_full] - y_ho[m_full]) ** 2))
        # max 上移
        return {
            "label": label, "rmse_full": rmse_full, "rmse_core": rmse_core,
            "max_pred": float(p.max()), "p99_pred": float(np.percentile(p, 99)),
            "p999_pred": float(np.percentile(p, 99.9)),
            "aobt_nat_n": int(mt.sum()),
            "aobt_nat_rmse": rmse_t, "aobt_nat_bias": bias_t,
            "aobt_nat_sse_pct": (sse_t / sse_all * 100) if sse_all > 0 else None,
            "aobt_nat_pred_mean": float(np.mean(p[mt])) if mt.sum() else None,
            "aobt_nat_pred_p50": float(np.median(p[mt])) if mt.sum() else None,
            "aobt_nat_pred_p99": float(np.percentile(p[mt], 99)) if mt.sum() else None,
            "aobt_nat_sse": sse_t, "total_sse": sse_all,
            "y_max": float(y_ho.max()),
        }

    # ============= A: 71 特征基线 =============
    print("\n[6] 实验 A: 71 特征 (含 >10800 极端行)...")
    t0 = time.time()
    pred_a, seeds_a = train_predict(X_tr_A, y_tr, X_ho_A, y_ho, label="A_71")
    a_eval = eval_pred(pred_a, "A_71")
    print(f"  A done {time.time()-t0:.0f}s  full={a_eval['rmse_full']:.2f}  "
          f"core={a_eval['rmse_core']:.2f}  seedσ={np.std(seeds_a):.2f}")

    # ============= B: 71 + offset =============
    print("\n[7] 实验 B: 71 + offset (含 >10800 极端行)...")
    t0 = time.time()
    pred_b, seeds_b = train_predict(X_tr_B, y_tr, X_ho_B, y_ho, label="B_71_offset")
    b_eval = eval_pred(pred_b, "B_71_offset")
    print(f"  B done {time.time()-t0:.0f}s  full={b_eval['rmse_full']:.2f}  "
          f"core={b_eval['rmse_core']:.2f}  seedσ={np.std(seeds_b):.2f}")

    # ============= 分机场 =============
    print("\n[8] 分机场 (全样本 RMSE, 重点 5)")
    print(f"  {'机场':6s} {'n':>8s} {'A':>10s} {'B':>10s} {'Δ(B-A)':>10s} {'y_mean':>10s}")
    ap_per = {}
    for ap in FOCUS_AP:
        mk = (aps_ho == ap) & m_full
        if mk.sum() < 100:
            continue
        ra = float(np.sqrt(np.mean((pred_a[mk] - y_ho[mk]) ** 2)))
        rb = float(np.sqrt(np.mean((pred_b[mk] - y_ho[mk]) ** 2)))
        ym = float(np.mean(y_ho[mk]))
        ap_per[ap] = {"n": int(mk.sum()), "A": ra, "B": rb, "delta": rb - ra, "y_mean": ym}
        print(f"  {ap:6s} {mk.sum():>8d} {ra:>10.2f} {rb:>10.2f} {rb - ra:>+10.2f} {ym:>10.1f}")

    # ============= 判断 =============
    print("\n" + "=" * 70)
    print("[判断]")
    print("=" * 70)
    seed_jitter = max(np.std(seeds_a), np.std(seeds_b))
    delta_full = b_eval["rmse_full"] - a_eval["rmse_full"]
    delta_aobt_sse = (b_eval["aobt_nat_sse"] - a_eval["aobt_nat_sse"]) / a_eval["aobt_nat_sse"] * 100 \
        if a_eval["aobt_nat_sse"] else 0.0
    pass_better = (delta_full < -seed_jitter) and (b_eval["aobt_nat_sse"] < a_eval["aobt_nat_sse"])
    print(f"  全样本 RMSE Δ(B-A) = {delta_full:+.2f}s,  seedσ≈{seed_jitter:.2f}s  "
          f"({'显著下降' if delta_full < -seed_jitter else '未下降'})")
    print(f"  AOBT_NaT SSE Δ = {delta_aobt_sse:+.2f}%  "
          f"({'下降' if b_eval['aobt_nat_sse'] < a_eval['aobt_nat_sse'] else '上升'})")
    print(f"  AOBT_NaT 段 RMSE  A={a_eval['aobt_nat_rmse']:.1f}  "
          f"B={b_eval['aobt_nat_rmse']:.1f}  Δ={b_eval['aobt_nat_rmse']-a_eval['aobt_nat_rmse']:+.1f}")
    print(f"  AOBT_NaT 段 bias  A={a_eval['aobt_nat_bias']:+.1f}  "
          f"B={b_eval['aobt_nat_bias']:+.1f}")
    print(f"  AOBT_NaT 段 pred mean A={a_eval['aobt_nat_pred_mean']:.1f}  "
          f"B={b_eval['aobt_nat_pred_mean']:.1f}  Δ={b_eval['aobt_nat_pred_mean']-a_eval['aobt_nat_pred_mean']:+.1f}")
    print(f"  预测 max  A={a_eval['max_pred']:.1f}  B={b_eval['max_pred']:.1f}  Δ={b_eval['max_pred']-a_eval['max_pred']:+.1f}")
    print(f"  预测 p99.9 A={a_eval['p999_pred']:.1f}  B={b_eval['p999_pred']:.1f}")
    print(f"  全样本 SSE A={a_eval['total_sse']:.0f}  B={b_eval['total_sse']:.0f}  Δ%={(b_eval['total_sse']-a_eval['total_sse'])/a_eval['total_sse']*100:+.2f}%")
    print()
    print(f"  Step1 通过? {pass_better}")
    if pass_better:
        print(f"  → 进入 Step2 (全量 12 月重训生成 v20)")
    else:
        print(f"  → 停止, 不进 Step2")

    # ============= 结果 JSON =============
    out = {
        "task": "extreme_restore_step1_holdout",
        "split": "train=2025-02..12 (含TX>10800, 剔TX<=0)  val=2025-01+07 (含TX>10800, 剔TX<=0)",
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "n_train_dep": int(len(tr_dep)),
        "n_train_extreme": int((tr_dep["TAXITIME_SEC_mvt"].values > 10800).sum()),
        "n_holdout_dep": int(len(ho_dep)),
        "n_holdout_extreme": int(n_extreme_ho),
        "global_offset_s": global_off,
        "results": {
            "A_71": {"seeds_rmse": [round(v, 3) for v in seeds_a], "seed_std": float(np.std(seeds_a)),
                     **a_eval},
            "B_71_offset": {"seeds_rmse": [round(v, 3) for v in seeds_b], "seed_std": float(np.std(seeds_b)),
                            **b_eval},
        },
        "per_airport": ap_per,
        "decision": {
            "delta_full_rmse": delta_full,
            "seed_jitter": seed_jitter,
            "delta_aobt_nat_sse_pct": delta_aobt_sse,
            "pass_step1": pass_better,
        },
        "total_time_sec": round(time.time() - t_start, 1),
    }
    fn = D0 / "extreme_restore_step1.json"
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n  结果 JSON: {fn}")
    print(f"  总耗时: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()