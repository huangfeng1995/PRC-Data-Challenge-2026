"""
v19 全量两阶段发版脚本
========================
- 训练: 2025-01..12 全 12 月 DEP (剔 TAXITIME in (0, 10800]), 不剔月
- 特征: 71 + t_unimp_bias_offset + trust_level + iobt_echoes_sched + aobt_iobt_absdiff_sec
        + actl_takeoff_ap_{5,15,30,60}m + actl_congestion_ap
- 模型: 3 seed (42/123/2024) × 3 head
        cls: 真值 > 1800
        short: 真值 <= 1800
        long: 真值 > 1800, 推理时下限 = t_unimp_r7
- 推理: ranking 2026 1+7 → soft fusion, clip [30, 10800]
- gate_check 五项: 行数 / NaN / 范围 / 集+顺序 / 反算 RMSE
- 输出: submissions/jubilant-puzzle_v19.parquet, 不上传

红线: 不改 label / 不 clip 真值训练 / 不建单机场模型 / 不删脏行 / 不上传
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

TASK = "LGB_full_v19_twostage_offset_trust_actl_3seed"
SEEDS = [42, 123, 2024]
MODEL_DIR = ROOT / "models" / "lgb_full_v19"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = D0 / "lgb_full_v19_result.json"
SUB_PATH = ROOT / "submissions" / "jubilant-puzzle_v19.parquet"

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
STEP2_FEATS = ["trust_level", "iobt_echoes_sched", "aobt_iobt_absdiff_sec"]
ACTL_FEATS = ["actl_takeoff_ap_5m", "actl_takeoff_ap_15m", "actl_takeoff_ap_30m",
              "actl_takeoff_ap_60m", "actl_congestion_ap"]
FEATS_FULL = FEATS71 + STEP1_FEATS + STEP2_FEATS + ACTL_FEATS

LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]

LGB_PARAMS_REG = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
LGB_PARAMS_CLS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8,
    "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "verbose": -1, "num_threads": -1,
}
ROUNDS_REG = int(os.environ.get("ROUNDS_REG", 3000))
ROUNDS_CLS = int(os.environ.get("ROUNDS_CLS", 600))
OFFSET_FOLDS = 5
OFFSET_SHRINK = 20.0
OFFSET_MIN_OBS = 20
OFFSET_CLIP = (-900.0, 2400.0)

CAT_MAP_PATH = ROOT / "models" / "feature_lists" / "_rebuild_12m_cat.json"
CAT_MAP = json.loads(CAT_MAP_PATH.read_text())


# ============================================================
# 特征工厂
# ============================================================
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
    mvt_dict = L.compute_mvt_anchor_features(df_full[dep_mask],
                                             t_unimp_vals=t_vals[dep_mask], label=label)
    for k, v in mvt_dict.items():
        df_full[k] = np.nan
        df_full.loc[dep_mask, k] = v
    return df_full


def _ns(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.values.astype("datetime64[ns]").view("int64")
    return pd.to_datetime(s, utc=True, errors="coerce").values.astype("datetime64[ns]").view("int64")


def add_trust(df):
    n = len(df)
    iNaT = np.iinfo(np.int64).min
    a = _ns(df["AOBT_3_flt"])
    i = _ns(df["IOBT_flt"])
    s = _ns(df["SCHED_TIME_UTC_mvt"])
    va, vi, vs = (a != iNaT), (i != iNaT), (s != iNaT)
    diff = np.full(n, np.nan, dtype=np.float64)
    m = va & vi
    diff[m] = np.abs(a[m].astype(np.float64) - i[m].astype(np.float64)) / 1e9
    trust = np.zeros(n, dtype=np.int8)
    trust[m & (diff <= 300.0)] = 2
    trust[m & (diff > 300.0) & (diff <= 1800.0)] = 1
    echo = np.zeros(n, dtype=np.int8)
    me = vi & vs
    echo[me] = (np.abs(i[me].astype(np.float64) - s[me].astype(np.float64)) / 1e9 < 60.0).astype(np.int8)
    df = df.copy()
    df["trust_level"] = trust
    df["iobt_echoes_sched"] = echo
    df["aobt_iobt_absdiff_sec"] = np.where(np.isnan(diff), -1.0, diff).astype(np.float32)
    return df


def add_actl_to_dep(df_all, label=""):
    """actl_* 5 特征 — 复用 ttt 版本实现"""
    n = len(df_all)
    dep_mask = (df_all["PHASE_mvt"].values == "DEP")
    eobt_ns = _ns(df_all["EOBT_1_flt"])
    sched_ns = _ns(df_all["SCHED_TIME_UTC_mvt"])
    mvt_ns = _ns(df_all["MVT_TIME_UTC_mvt"])
    aobt_ns = _ns(df_all["AOBT_3_flt"])
    iNaT = np.iinfo(np.int64).min
    anchor = np.where(eobt_ns != iNaT, eobt_ns, sched_ns).astype("int64")
    aep = df_all["ADEP_mvt"].astype(str).values

    NEW5NAMES = ["actl_takeoff_ap_5m", "actl_takeoff_ap_15m", "actl_takeoff_ap_30m",
                 "actl_takeoff_ap_60m", "actl_congestion_ap"]
    out = {c: np.zeros(n, dtype=np.float32) for c in NEW5NAMES}

    aep_order = np.argsort(aep, kind="mergesort")
    aep_sorted = aep[aep_order]
    unique_ap, starts = np.unique(aep_sorted, return_index=True)
    ends = np.concatenate([starts[1:], [len(aep_sorted)]])
    anchor_sorted = anchor[aep_order]
    mvt_sorted = mvt_ns[aep_order]
    aobt_sorted = aobt_ns[aep_order]
    for i_ap in range(len(unique_ap)):
        s, e = starts[i_ap], ends[i_ap]
        mvt_g = mvt_sorted[s:e]
        aobt_g = aobt_sorted[s:e]
        anchor_g = anchor_sorted[s:e]
        valid_m = mvt_g != iNaT
        valid_a = aobt_g != iNaT
        mvt_valid = np.sort(mvt_g[valid_m])
        aobt_valid = np.sort(aobt_g[valid_a])
        ag = anchor_g
        for win_min, key in [(5, "actl_takeoff_ap_5m"),
                              (15, "actl_takeoff_ap_15m"),
                              (30, "actl_takeoff_ap_30m"),
                              (60, "actl_takeoff_ap_60m")]:
            if len(mvt_valid) == 0:
                continue
            w_ns = win_min * 60 * int(1e9)
            left = np.searchsorted(mvt_valid, ag - w_ns, side="left")
            right = np.searchsorted(mvt_valid, ag, side="left")
            out[key][aep_order[s:e]] = (right - left).astype(np.float32)
        if len(aobt_valid) > 0 and len(mvt_valid) > 0:
            n_a = np.searchsorted(aobt_valid, ag, side="right")
            n_m = np.searchsorted(mvt_valid, ag, side="right")
            out["actl_congestion_ap"][aep_order[s:e]] = np.maximum(n_a - n_m, 0).astype(np.float32)
    return out


def fit_offset_map(tr_df, global_off):
    """Step1: 全量训练时, train 内 5-fold OOF 偏移, 返回 (tr_off, tr_known, fit_map)"""
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
    """对外部 keys (rank_dep) 查 fit_map, fallback 到 global_off"""
    v = keys_df.merge(fit_map, how="left", on=["ADEP_mvt", "RUNWAY_mvt"])["off"].to_numpy()
    return np.where(np.isnan(v), global_off, np.nan_to_num(v)).astype(np.float32), \
        (~np.isnan(v)).astype(np.int8)


def add_bias_offset(tr_df, rank_dep_keys, global_off):
    """Step1: 全量训练时, train 内 5-fold OOF 偏移, ranking DEP 用全量 train 拟合.
    rank_dep_keys: np.ndarray of (ADEP, RUNWAY) 元组, 长度 = rank_dep 行数"""
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

    full = _fit(base)
    rk_df = pd.DataFrame({
        "ADEP_mvt": rank_dep_keys[:, 0].astype(str),
        "RUNWAY_mvt": rank_dep_keys[:, 1].astype(str),
    })
    rk_v = rk_df.merge(full, how="left", on=["ADEP_mvt", "RUNWAY_mvt"])["off"].to_numpy()
    return out.astype(np.float32), known.astype(np.int8), \
        np.where(np.isnan(rk_v), global_off, np.nan_to_num(rk_v)).astype(np.float32), \
        (~np.isnan(rk_v)).astype(np.int8), full


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


def train_reg(X_tr, y_tr, label):
    boosters = []
    for sd in SEEDS:
        ds = lgb.Dataset(X_tr, label=y_tr.astype(np.float32),
                         categorical_feature=LOW_CARD_CATS)
        b = lgb.train(dict(LGB_PARAMS_REG, seed=sd), ds, num_boost_round=ROUNDS_REG)
        boosters.append(b)
        b.save_model(str(MODEL_DIR / f"{label}_seed{sd}.txt"))
    return boosters


def train_cls(X_tr, y_lab, label):
    boosters = []
    for sd in SEEDS:
        ds = lgb.Dataset(X_tr, label=y_lab.astype(np.float32),
                         categorical_feature=LOW_CARD_CATS)
        b = lgb.train(dict(LGB_PARAMS_CLS, seed=sd), ds, num_boost_round=ROUNDS_CLS)
        boosters.append(b)
        b.save_model(str(MODEL_DIR / f"{label}_seed{sd}.txt"))
    return boosters


def main():
    t_start = time.time()
    print("=" * 70); print("[任务] v19 全量两阶段发版"); print("=" * 70)
    print(f"  seeds={SEEDS} rounds_reg={ROUNDS_REG} rounds_cls={ROUNDS_CLS}")
    print(f"  features={len(FEATS_FULL)} (71+2 offset+3 trust+5 actl)")
    print()

    # ============================================================
    # [1] 加载
    # ============================================================
    print("[1] 加载 2025 全 12 月 + ranking")
    train_months = list(range(1, 13))
    train_dfs = [load_month(
        (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
         if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet"),
        columns=TRAIN_COLS) for m in train_months]
    df_tr_raw = pd.concat(train_dfs, ignore_index=True)
    print(f"  train_raw={len(df_tr_raw):,}")
    rank_cols = [c for c in TRAIN_COLS if c != "TAXITIME_SEC_mvt"]
    rank_df = pd.read_parquet(ROOT / "data" / "raw" / "ranking.parquet", columns=rank_cols)
    print(f"  ranking={len(rank_df):,}")

    # ============================================================
    # [2] 训练折 DEP (按月统计 1-12, 不剔月, 剔 TAXITIME in (0, 10800])
    # ============================================================
    print("\n[2] 训练折 DEP 清洗 (TAXITIME in (0, 10800])")
    df_tr_raw_dep = df_tr_raw[df_tr_raw["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tt = df_tr_raw_dep["TAXITIME_SEC_mvt"].values
    clean = (tt > 0) & (tt <= 10800)
    df_tr_dep_clean = df_tr_raw_dep[clean].copy().reset_index(drop=True)
    print(f"  训练 DEP (清洗后): {len(df_tr_dep_clean):,}  (剔 TAXITIME<=0: {(tt<=0).sum():,}, >10800: {(tt>10800).sum():,})")

    # 月份分布
    mo = pd.to_datetime(df_tr_dep_clean["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").dt.month.values
    for m in sorted(np.unique(mo)):
        print(f"    month={m:>2d}: n={(mo==m).sum():,}")

    # LUT (训练折全量)
    lut = L.build_t_unimp_lut(df_tr_dep_clean)
    print(f"\n  LUT L1={len(lut['L1_q'])}, L2={len(lut['L2_q'])}, L3={len(lut['L3_q'])}, "
          f"global={lut['global_q30']:.0f}s")

    # ============================================================
    # [3] 重建训练折特征
    # ============================================================
    print("\n[3] 训练折特征 (全 12 月)...")
    t0 = time.time()
    tr_full = build_features(df_tr_raw, df_tr_raw, lut, label="tr")
    tr_dep = tr_full[tr_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    # 同清洗
    tt2 = tr_dep["TAXITIME_SEC_mvt"].values
    tr_dep = tr_dep[(tt2 > 0) & (tt2 <= 10800)].reset_index(drop=True)
    print(f"  训练 DEP 全套 (清洗后): {tr_dep.shape}  ({time.time()-t0:.0f}s)")

    # ============================================================
    # [4] trust + offset + actl
    # ============================================================
    print("\n[4] trust_level + t_unimp_bias_offset + actl_* 5")
    t0 = time.time()
    tr_dep = add_trust(tr_dep)
    print(f"  trust added {time.time()-t0:.0f}s")

    # actl 用全 train_raw (含 ARR, 用于回填 actl_takeoff_ap_*)
    t0 = time.time()
    actl_tr = add_actl_to_dep(df_tr_raw, label="tr")
    # 训练折 (TAXITIME 清洗后) mask
    clean_tr_full = (df_tr_raw["PHASE_mvt"] == "DEP").values & \
                    (df_tr_raw["TAXITIME_SEC_mvt"].values > 0) & \
                    (df_tr_raw["TAXITIME_SEC_mvt"].values <= 10800)
    for c in ACTL_FEATS:
        tr_dep[c] = actl_tr[c][clean_tr_full]
    print(f"  actl added {time.time()-t0:.0f}s")

    # offset (全 train OOF + 全 train 拟合 ranking)
    t0 = time.time()
    global_off = float(np.median(tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)) -
                       np.median(tr_dep["t_unimp_r7"].values.astype(np.float64)))
    # 先把 ranking 列补齐再做 offset (我们直接传 rank 顺序做拟合, 不构建 rank 特征)
    rank_keys_only = pd.DataFrame({
        "ADEP_mvt": rank_df["ADEP_mvt"].astype(str).values,
        "RUNWAY_mvt": rank_df["RUNWAY_mvt"].astype(str).values,
    })
    tr_off, tr_known, full_off = fit_offset_map(tr_dep, global_off)
    tr_dep["t_unimp_bias_offset"] = tr_off
    tr_dep["t_unimp_bias_offset_known"] = tr_known
    print(f"  offset added (tr OOF) {time.time()-t0:.0f}s  global={global_off:+.1f}s  "
          f"tr_cov={tr_known.mean()*100:.1f}%  "
          f"groups={len(full_off):,}  (rk_offset 计算将在 rank_dep 构建后)")

    # ============================================================
    # [5] ranking 特征 + actl + trust
    # ============================================================
    print("\n[5] ranking 特征...")
    t0 = time.time()
    rank_full = build_features(rank_df, df_tr_raw, lut, label="rank")
    rank_dep = rank_full[rank_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    print(f"  ranking DEP 全套: {rank_dep.shape}  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    rank_dep = add_trust(rank_dep)
    actl_rk = add_actl_to_dep(rank_df, label="rank")
    dep_mask_rk = (rank_df["PHASE_mvt"].values == "DEP")
    for c in ACTL_FEATS:
        rank_dep[c] = actl_rk[c][dep_mask_rk]
    rk_off_vals, rk_known_vals = apply_offset_map(
        pd.DataFrame({
            "ADEP_mvt": rank_dep["ADEP_mvt"].astype(str).values,
            "RUNWAY_mvt": rank_dep["RUNWAY_mvt"].astype(str).values,
        }),
        full_off, global_off,
    )
    rank_dep["t_unimp_bias_offset"] = rk_off_vals
    rank_dep["t_unimp_bias_offset_known"] = rk_known_vals
    print(f"  ranking 附加 {time.time()-t0:.0f}s  rank_dep shape={rank_dep.shape}  "
          f"rk_cov={rk_known_vals.mean()*100:.1f}%")

    # ============================================================
    # [6] 编码 + 矩阵对齐
    # ============================================================
    print("\n[6] cat 编码 + 矩阵对齐")
    X_tr = encode_X(tr_dep, FEATS_FULL)
    X_rk = encode_X(rank_dep, FEATS_FULL)
    y_tr = tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float32)
    print(f"  X_tr shape: {X_tr.shape}")
    print(f"  X_rk shape: {X_rk.shape}")

    # ============================================================
    # [7] 训练 3 head × 3 seed
    # ============================================================
    print("\n[7] 训练 (cls + short + long) × 3 seed")
    t0 = time.time()
    y_lab = (y_tr > 1800).astype(np.int32)
    cls_boosters = train_cls(X_tr, y_lab, label="cls")
    print(f"  cls done {time.time()-t0:.0f}s")

    t0 = time.time()
    idx_s = np.where((y_tr > 0) & (y_tr <= 1800))[0]
    short_boosters = train_reg(X_tr.iloc[idx_s], y_tr[idx_s], label="short")
    print(f"  short done {time.time()-t0:.0f}s n_train={idx_s.size:,}")

    t0 = time.time()
    idx_l = np.where((y_tr > 1800) & (y_tr < 10800))[0]
    long_boosters = train_reg(X_tr.iloc[idx_l], y_tr[idx_l], label="long")
    print(f"  long done {time.time()-t0:.0f}s n_train={idx_l.size:,}")

    # ============================================================
    # [8] ranking 推理 + 软融合
    # ============================================================
    print("\n[8] ranking 推理 + 软融合")
    t0 = time.time()
    P_long = np.mean(np.stack([b.predict(X_rk) for b in cls_boosters]), axis=0)
    p_short = np.mean(np.stack([b.predict(X_rk) for b in short_boosters]), axis=0)
    p_long_raw = np.mean(np.stack([b.predict(X_rk) for b in long_boosters]), axis=0)
    t_rk = np.nan_to_num(rank_dep["t_unimp_r7"].values.astype(np.float64), nan=0.0)
    p_long = np.maximum(p_long_raw, t_rk)
    P_lc = np.clip(P_long, 0.0, 1.0)
    final = p_short * (1.0 - P_lc) + p_long * P_lc
    final = np.clip(final, 30.0, 10800.0)
    print(f"  ranking predict {time.time()-t0:.0f}s  n={len(final):,}")
    print(f"  P_long 分布: p50={np.median(P_lc):.4f}, p95={np.percentile(P_lc, 95):.4f}, "
          f"n_pred@0.5={(P_lc > 0.5).sum():,}")
    print(f"  final 分布: min={final.min():.1f}, p50={np.median(final):.1f}, "
          f"p99={np.percentile(final, 99):.1f}, max={final.max():.1f}")

    # ============================================================
    # [9] gate_check 五项 + 提交
    # ============================================================
    print("\n[9] gate_check 五项 + 提交")
    rank_ids = rank_dep["MVT_ID_mvt"].astype(float).values
    sub_df = pd.DataFrame({
        "MVT_ID_mvt": rank_ids,
        "TAXITIME_SEC_mvt": np.round(final, 0).astype(int),
    })
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_parquet(SUB_PATH, index=False)
    print(f"  提交路径: {SUB_PATH}")
    print(f"  行数: {len(sub_df):,}")

    sub_ref = pd.read_parquet(ROOT / "data" / "raw" / "submitting.parquet", columns=["MVT_ID_mvt"])
    exp_ids = sub_ref["MVT_ID_mvt"].astype(float).values
    set_match = set(rank_ids) == set(exp_ids)
    order_match = np.all(rank_ids == exp_ids)

    print("\n  --- gate_check ---")
    print(f"  CHECK 1 行数:    {len(sub_df):,} == 344,841 ?  {'✅' if len(sub_df) == 344841 else '❌'}")
    n_nan = int(sub_df["TAXITIME_SEC_mvt"].isna().sum())
    print(f"  CHECK 2 NaN:     {n_nan} {'✅' if n_nan == 0 else '❌'}")
    mn, mx = int(sub_df["TAXITIME_SEC_mvt"].min()), int(sub_df["TAXITIME_SEC_mvt"].max())
    n_oob = int(((sub_df["TAXITIME_SEC_mvt"] < 30) | (sub_df["TAXITIME_SEC_mvt"] > 10800)).sum())
    print(f"  CHECK 3 范围:    min={mn}, max={mx}, oob={n_oob} {'✅' if (n_oob == 0 and 30 <= mn and mx <= 10800) else '❌'}")
    print(f"  CHECK 4 集+序:   set={set_match}, order={order_match} {'✅' if (set_match and order_match) else '❌'}")
    # 5: 反算 RMSE (在 holdout 1+7 上, 跳过 - 本脚本只训 ranking, 不留 holdout)
    print(f"  CHECK 5 留 holdout 反算 RMSE - 跳过 (本脚本仅发版, 无 holdout 评估)")

    # ============================================================
    # [10] 结果 JSON
    # ============================================================
    out = {
        "task": TASK,
        "seeds": SEEDS,
        "rounds_reg": ROUNDS_REG,
        "rounds_cls": ROUNDS_CLS,
        "training_months": "2025-01..12 (全量, 剔 TAXITIME<=0 或 >10800, 不剔月)",
        "n_train_dep": int(idx_s.size + idx_l.size),
        "n_train_short": int(idx_s.size),
        "n_train_long": int(idx_l.size),
        "n_train_long_lab_pos": int((y_lab == 1).sum()),
        "features": FEATS_FULL,
        "n_features": len(FEATS_FULL),
        "offset_stats": {
            "global_offset_s": global_off,
            "train_cov_pct": float(tr_known.mean() * 100),
            "rank_cov_pct": float(rk_known_vals.mean() * 100),
            "n_groups_used": int(len(full_off)),
        },
        "ranking_inference": {
            "p_long_p50": float(np.median(P_lc)),
            "p_long_p95": float(np.percentile(P_lc, 95)),
            "n_pred_long@0.5": int((P_lc > 0.5).sum()),
            "final_min": float(final.min()),
            "final_p50": float(np.median(final)),
            "final_p99": float(np.percentile(final, 99)),
            "final_max": float(final.max()),
        },
        "gate_check": {
            "n_rows": int(len(sub_df)),
            "expected_n_rows": 344841,
            "n_nan": n_nan,
            "min": mn,
            "max": mx,
            "n_out_of_range": n_oob,
            "mvt_id_set_match": bool(set_match),
            "mvt_id_order_match": bool(order_match),
            "all_pass": bool(len(sub_df) == 344841 and n_nan == 0 and n_oob == 0
                              and set_match and order_match),
        },
        "model_dir": str(MODEL_DIR),
        "submission_path": str(SUB_PATH),
        "total_time_sec": round(time.time() - t_start, 1),
    }
    RESULT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\n  结果 JSON: {RESULT_PATH}")
    print(f"  模型目录: {MODEL_DIR} (3 head × 3 seed = 9 文件)")
    print(f"  总耗时: {(time.time()-t_start)/60:.1f} min")
    print("\n" + "=" * 70)
    print("[报告完成] 本步不上传, 等用户确认")
    print(f"  v19 提交: {SUB_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()