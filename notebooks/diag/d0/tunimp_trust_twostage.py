"""
Step0-Step4 合并实验: t_unimp_bias_offset + IOBT trust_level + 长尾两阶段
======================================================================
Step0  Jan<->Jul 互验 (Fold A: train=Jan val=Jul / Fold B: train=Jul val=Jan) 出基线
Step1  t_unimp_bias_offset: (ADEP,RUNWAY) 上 offset=median(true)-median(t_unimp)
        -> train 用 5-fold OOF (防自泄漏), holdout 用 train 全量拟合
        -> 作为新特征喂入, 不改原 t_unimp 列
Step2  trust_level: 0=AOBT缺失或|AOBT-IOBT|>1800 / 1=300-1800 / 2=<=300
        + iobt_echoes_sched = |IOBT-SCHED|<60
Step3  长尾两阶段: 分类 head + 短 head + 长 head + 软加权
Step4  四组对照: (1)71特征 (2)+offset (3)+trust (4)完整两阶段

红线: 不改 label / 不 clip 真值 / 不建单机场模型 / 不删脏行 / 不上传
      label 在长尾 head 训练时保持原值, t_unimp_r7 只在推理时做下限
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
WORK = Path("/Users/openclaw/.trae-cn/work/6aaca8f8b524b5e1e3ae7a73")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

SMOKE = "--smoke" in sys.argv
CACHE = WORK / "ttt_cache"
CACHE.mkdir(parents=True, exist_ok=True)

TRAIN_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "EOBT_1_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
    "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_flt", "ARVT_1_flt",
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
FEATS2 = FEATS71 + STEP1_FEATS
FEATS3 = FEATS71 + STEP1_FEATS + STEP2_FEATS

LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]

SEEDS = [42] if SMOKE else [42, 123, 2024]
ROUNDS_REG = 120 if SMOKE else int(os.environ.get("ROUNDS_REG", 3000))
ROUNDS_CLS = 80 if SMOKE else int(os.environ.get("ROUNDS_CLS", 600))
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
FOCUS_AP = ["LIRF", "EGLL", "LTFM", "LFPG"]
N_OFFSET_FOLDS = 5
OFFSET_SHRINK = 20.0
OFFSET_MIN_OBS = 20
OFFSET_CLIP = (-900.0, 2400.0)

CAT_MAP = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())

MONTH_FILE = {m: (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                  if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
              for m in range(1, 13)}
MAIN_TRAIN = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
MAIN_VAL = [1, 7]

SPLITS = {
    "MAIN": (MAIN_TRAIN, MAIN_VAL),
    "FoldA_Jan2Jul": ([1], [7]),
    "FoldB_Jul2Jan": ([7], [1]),
}
RUN_SPLITS = ["FoldB_Jul2Jan"] if SMOKE else ["FoldA_Jan2Jul", "FoldB_Jul2Jan", "MAIN"]


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


def add_trust_features(df):
    """Step2: IOBT 口径 trust_level + iobt_echoes_sched + 连续 absdiff"""
    n = len(df)
    iNaT = np.iinfo(np.int64).min
    a = _ns(df["AOBT_3_flt"])
    i = _ns(df["IOBT_flt"])
    s = _ns(df["SCHED_TIME_UTC_mvt"])
    va, vi, vs = (a != iNaT), (i != iNaT), (s != iNaT)

    diff = np.full(n, np.nan, dtype=np.float64)
    m = va & vi
    diff[m] = np.abs(a[m].astype(np.float64) - i[m].astype(np.float64)) / 1e9

    # trust=0: AOBT 缺失 或 |AOBT-IOBT|>1800 ; 1: 300-1800 ; 2: <=300
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


def _group_offset(keys, y, t, n, fold_ids, global_off, use_folds):
    """按 (ADEP,RUNWAY) 算 offset=median(true)-median(t), 带 shrink 与 min_obs。
    use_folds=True 时只用 fold_ids != f 的行 (OOF)"""
    sub = pd.DataFrame({"k": keys, "y": y, "t": t})
    if use_folds is not None:
        sub = sub[sub["fold"] != use_folds]
    med = sub.groupby("k").agg(med_y=("y", "median"), med_t=("t", "median"), n=("y", "size"))
    med = med[med["n"] >= OFFSET_MIN_OBS]
    off = (med["med_y"] - med["med_t"]).clip(*OFFSET_CLIP)
    off = (off * med["n"] + global_off * OFFSET_SHRINK) / (med["n"] + OFFSET_SHRINK)
    return off, set(off.index)


def add_bias_offset(tr_df, ho_df, tag=""):
    """Step1: t_unimp_bias_offset. train 用 5-fold OOF, holdout 用 train 全量."""
    rng = np.random.RandomState(0)
    fold = rng.randint(0, N_OFFSET_FOLDS, size=len(tr_df))
    base = pd.DataFrame({
        "ADEP_mvt": tr_df["ADEP_mvt"].astype(str).values,
        "RUNWAY_mvt": tr_df["RUNWAY_mvt"].astype(str).values,
        "y": tr_df["TAXITIME_SEC_mvt"].values.astype(np.float64),
        "t": tr_df["t_unimp_r7"].values.astype(np.float64),
        "fold": fold,
    })
    global_off = float(np.median(base["y"]) - np.median(base["t"]))

    def _fit(sub):
        med = sub.groupby(["ADEP_mvt", "RUNWAY_mvt"]).agg(
            med_y=("y", "median"), med_t=("t", "median"), n=("y", "size"))
        med = med[med["n"] >= OFFSET_MIN_OBS]
        off = (med["med_y"] - med["med_t"]).clip(*OFFSET_CLIP)
        med["off"] = (off * med["n"] + global_off * OFFSET_SHRINK) / (med["n"] + OFFSET_SHRINK)
        return med[["off"]].reset_index()

    out = np.full(len(tr_df), global_off, dtype=np.float64)
    known = np.zeros(len(tr_df), dtype=np.int8)
    for f in range(N_OFFSET_FOLDS):
        med = _fit(base[base["fold"] != f])
        pos = np.where(fold == f)[0]
        view = base.iloc[pos][["ADEP_mvt", "RUNWAY_mvt"]].merge(med, how="left",
            on=["ADEP_mvt", "RUNWAY_mvt"])
        v = view["off"].to_numpy()
        out[pos] = np.where(np.isnan(v), global_off, np.nan_to_num(v))
        known[pos] = (~np.isnan(v)).astype(np.int8)

    full = _fit(base)
    ho_keys = pd.DataFrame({
        "ADEP_mvt": ho_df["ADEP_mvt"].astype(str).values,
        "RUNWAY_mvt": ho_df["RUNWAY_mvt"].astype(str).values,
    })
    ho_view = ho_keys.merge(full, how="left", on=["ADEP_mvt", "RUNWAY_mvt"])
    ho_v = ho_view["off"].to_numpy()

    tr_df = tr_df.copy()
    ho_df = ho_df.copy()
    tr_df["t_unimp_bias_offset"] = out.astype(np.float32)
    tr_df["t_unimp_bias_offset_known"] = known
    ho_df["t_unimp_bias_offset"] = np.where(np.isnan(ho_v), global_off,
                                            np.nan_to_num(ho_v)).astype(np.float32)
    ho_df["t_unimp_bias_offset_known"] = (~np.isnan(ho_v)).astype(np.int8)

    lirf_off = full[full["ADEP_mvt"] == "LIRF"]["off"].to_numpy()
    stats = {
        "global_offset": global_off,
        "train_coverage": float(known.mean()),
        "holdout_coverage": float((~np.isnan(ho_v)).mean()),
        "n_groups_total": int(base.groupby(["ADEP_mvt", "RUNWAY_mvt"]).ngroups),
        "n_groups_used": int(len(full)),
        "LIRF_offset": float(np.nanmedian(lirf_off)) if lirf_off.size > 0 else None,
        "n_lirf_rwy_groups": int(lirf_off.size),
    }
    return tr_df, ho_df, stats


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


# ============================================================
# 训练工具
# ============================================================
def train_reg(X_tr, y_tr, X_ho, y_ho, rounds, lo=30.0, hi=10800.0):
    preds, boosters, per_seed = [], [], []
    for sd in SEEDS:
        ds = lgb.Dataset(X_tr, label=y_tr.astype(np.float32), categorical_feature=LOW_CARD_CATS)
        b = lgb.train(dict(LGB_PARAMS_REG, seed=sd), ds, num_boost_round=rounds)
        p = np.clip(b.predict(X_ho).astype(np.float64), lo, hi)
        preds.append(p); boosters.append(b)
        per_seed.append(float(np.sqrt(np.mean((p - y_ho) ** 2))))
        print(f"      seed={sd} RMSE={per_seed[-1]:.2f} ({time.strftime('%H:%M:%S')})", flush=True)
    return np.mean(np.stack(preds), axis=0), per_seed, boosters


def train_cls(X_tr, y_lab, X_ho, rounds):
    preds, boosters = [], []
    for sd in SEEDS:
        ds = lgb.Dataset(X_tr, label=y_lab.astype(np.float32), categorical_feature=LOW_CARD_CATS)
        b = lgb.train(dict(LGB_PARAMS_CLS, seed=sd), ds, num_boost_round=rounds)
        preds.append(b.predict(X_ho).astype(np.float64)); boosters.append(b)
    return np.mean(np.stack(preds), axis=0), boosters


def gain_table(boosters, feats):
    g = np.mean([b.feature_importance(importance_type="gain") for b in boosters], axis=0)
    t = pd.DataFrame({"feat": feats, "gain": g})
    t["gain_pct"] = t["gain"] / t["gain"].sum() * 100
    return t.sort_values("gain", ascending=False).reset_index(drop=True)


# ============================================================
# 单 split 全流程
# ============================================================
def run_split(name, tr_months, ho_months):
    print()
    print("#" * 74)
    print(f"# SPLIT {name}: train={tr_months} val={ho_months}  seeds={SEEDS} rounds={ROUNDS_REG}")
    print("#" * 74)
    out = {"split": name, "train_months": tr_months, "val_months": ho_months,
           "seeds": SEEDS, "rounds_reg": ROUNDS_REG, "rounds_cls": ROUNDS_CLS}

    t0 = time.time()
    df_tr_raw = pd.concat([load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in tr_months],
                          ignore_index=True)
    df_ho_raw = pd.concat([load_month(MONTH_FILE[m], columns=TRAIN_COLS) for m in ho_months],
                          ignore_index=True)
    df_tr_dep_all = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                              (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
                              (df_tr_raw["TAXITIME_SEC_mvt"] < 10800)].copy()
    lut = L.build_t_unimp_lut(df_tr_dep_all)
    print(f"  load {time.time()-t0:.0f}s  train_raw={len(df_tr_raw):,} val_raw={len(df_ho_raw):,}")

    t0 = time.time()
    ho_full = build_features(df_ho_raw, df_tr_raw, lut, label=f"{name}_ho")
    ho_dep = ho_full[ho_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    tr_full = build_features(df_tr_raw, df_tr_raw, lut, label=f"{name}_tr")
    tr_dep = tr_full[(tr_full["PHASE_mvt"] == "DEP") &
                     (tr_full["TAXITIME_SEC_mvt"] > 0) &
                     (tr_full["TAXITIME_SEC_mvt"] < 10800)].copy().reset_index(drop=True)
    print(f"  build {time.time()-t0:.0f}s  train_DEP={len(tr_dep):,} val_DEP={len(ho_dep):,}")

    tr_dep = add_trust_features(tr_dep)
    ho_dep = add_trust_features(ho_dep)
    tr_dep, ho_dep, off_stats = add_bias_offset(tr_dep, ho_dep, tag=name)
    out["offset_stats"] = off_stats
    print(f"  offset: global={off_stats['global_offset']:+.1f}s  "
          f"train_cov={off_stats['train_coverage']*100:.1f}%  "
          f"val_cov={off_stats['holdout_coverage']*100:.1f}%  "
          f"groups={off_stats['n_groups_used']:,}/{off_stats['n_groups_total']:,}  "
          f"LIRF={off_stats['LIRF_offset']}")

    y_tr = tr_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    y_ho = ho_dep["TAXITIME_SEC_mvt"].values.astype(np.float64)
    t_ho = ho_dep["t_unimp_r7"].values.astype(np.float64)
    aps = ho_dep["ADEP_mvt"].astype(str).values

    m_short = (y_ho > 0) & (y_ho <= 1800)
    m_long = (y_ho > 1800) & (y_ho <= 10800)
    m_core = m_short | m_long
    m_full = y_ho > 0

    def sl(pred):
        return {"rmse_short": float(np.sqrt(np.mean((pred[m_short] - y_ho[m_short]) ** 2))),
                "rmse_long": float(np.sqrt(np.mean((pred[m_long] - y_ho[m_long]) ** 2))),
                "rmse_core": float(np.sqrt(np.mean((pred[m_core] - y_ho[m_core]) ** 2))),
                "rmse_full": float(np.sqrt(np.mean((pred[m_full] - y_ho[m_full]) ** 2))),
                "bias_short": float(np.mean(pred[m_short] - y_ho[m_short])),
                "bias_core": float(np.mean(pred[m_core] - y_ho[m_core])),
                "seed_rmse": [round(v, 3) for v in _LAST_SEEDS],
                "seed_std": float(np.std(_LAST_SEEDS)) if len(_LAST_SEEDS) > 1 else 0.0}

    def per_ap(pred):
        d = {}
        for ap in FOCUS_AP:
            mk = (aps == ap) & m_full
            if mk.sum() < 100:
                continue
            e = {"n": int(mk.sum()),
                 "rmse_core": float(np.sqrt(np.mean((pred[m_short & (aps == ap)] -
                                                     y_ho[m_short & (aps == ap)]) ** 2)))
                 if (m_short & (aps == ap)).sum() > 0 else None}
            ml = (aps == ap) & m_long
            e["rmse_long"] = float(np.sqrt(np.mean((pred[ml] - y_ho[ml]) ** 2))) if ml.sum() > 0 else None
            ms = (aps == ap) & m_short
            e["rmse_short"] = float(np.sqrt(np.mean((pred[ms] - y_ho[ms]) ** 2))) if ms.sum() > 0 else None
            d[ap] = e
        return d

    results = {}

    # ---------- Group 1: 71 特征基线 ----------
    X1, X2, X3 = encode_X(tr_dep, FEATS71), encode_X(tr_dep, FEATS2), encode_X(tr_dep, FEATS3)
    V1, V2, V3 = encode_X(ho_dep, FEATS71), encode_X(ho_dep, FEATS2), encode_X(ho_dep, FEATS3)

    print("\n  [G1] 71 特征基线")
    t0 = time.time()
    _LAST = train_reg(X1, y_tr, V1, y_ho, ROUNDS_REG)
    _LAST_SEEDS = _LAST[1]
    p1 = _LAST[0]
    results["g1_base71"] = {**sl(p1), "per_ap": per_ap(p1)}
    print(f"    G1 done {time.time()-t0:.0f}s core={results['g1_base71']['rmse_core']:.2f} "
          f"full={results['g1_base71']['rmse_full']:.2f}")

    # ---------- Group 2: + t_unimp_bias_offset ----------
    print("\n  [G2] 71 + t_unimp_bias_offset")
    t0 = time.time()
    _LAST = train_reg(X2, y_tr, V2, y_ho, ROUNDS_REG)
    _LAST_SEEDS = _LAST[1]
    p2 = _LAST[0]
    gt2 = gain_table(_LAST[2], FEATS2)
    results["g2_offset"] = {**sl(p2), "per_ap": per_ap(p2),
                            "gain_t_unimp_bias_offset": float(gt2.loc[gt2["feat"] == "t_unimp_bias_offset", "gain"].iloc[0]),
                            "gain_pct_t_unimp_bias_offset": float(gt2.loc[gt2["feat"] == "t_unimp_bias_offset", "gain_pct"].iloc[0]),
                            "gain_t_unimp_r7": float(gt2.loc[gt2["feat"] == "t_unimp_r7", "gain"].iloc[0]),
                            "top10_gain": gt2.head(10)[["feat", "gain_pct"]].round(3).to_dict("records")}
    print(f"    G2 done {time.time()-t0:.0f}s core={results['g2_offset']['rmse_core']:.2f} "
          f"| offset gain_pct={results['g2_offset']['gain_pct_t_unimp_bias_offset']:.3f}%")

    # ---------- Group 3: + trust_level ----------
    print("\n  [G3] 71 + offset + trust_level")
    t0 = time.time()
    _LAST = train_reg(X3, y_tr, V3, y_ho, ROUNDS_REG)
    _LAST_SEEDS = _LAST[1]
    p3 = _LAST[0]
    gt3 = gain_table(_LAST[2], FEATS3)
    trust_ho = ho_dep["trust_level"].values
    trust_grp = {}
    for lv in (0, 1, 2):
        mt = (trust_ho == lv) & m_full
        if mt.sum() < 50:
            continue
        trust_grp[str(lv)] = {
            "n": int(mt.sum()),
            "g1": float(np.sqrt(np.mean((p1[mt] - y_ho[mt]) ** 2))),
            "g3": float(np.sqrt(np.mean((p3[mt] - y_ho[mt]) ** 2))),
        }
        trust_grp[str(lv)]["delta"] = trust_grp[str(lv)]["g3"] - trust_grp[str(lv)]["g1"]
    results["g3_trust"] = {**sl(p3), "per_ap": per_ap(p3),
                           "gain_trust_level": float(gt3.loc[gt3["feat"] == "trust_level", "gain"].iloc[0]),
                           "gain_pct_trust_level": float(gt3.loc[gt3["feat"] == "trust_level", "gain_pct"].iloc[0]),
                           "gain_pct_iobt_echoes_sched": float(gt3.loc[gt3["feat"] == "iobt_echoes_sched", "gain_pct"].iloc[0]),
                           "gain_pct_aobt_iobt_absdiff_sec": float(gt3.loc[gt3["feat"] == "aobt_iobt_absdiff_sec", "gain_pct"].iloc[0]),
                           "by_trust_level": trust_grp}
    print(f"    G3 done {time.time()-t0:.0f}s core={results['g3_trust']['rmse_core']:.2f} "
          f"| trust gain_pct={results['g3_trust']['gain_pct_trust_level']:.3f}%")
    for lv, d in trust_grp.items():
        print(f"      trust={lv}: n={d['n']:,} G1={d['g1']:.1f} G3={d['g3']:.1f} Δ={d['delta']:+.1f}")

    # ---------- Group 4: 两阶段 ----------
    print("\n  [G4] 长尾两阶段 (cls + short head + long head)")
    t0 = time.time()
    y_lab_long = (y_tr > 1800).astype(np.int32)
    P_long, _ = train_cls(X3, y_lab_long, V3, ROUNDS_CLS)
    y_ho_lab = (y_ho > 1800).astype(np.int32)
    y_core = y_ho_lab[m_core]; p_core = P_long[m_core]
    auc = float(roc_auc_score(y_core, p_core))
    ap = float(average_precision_score(y_core, p_core))
    print(f"    cls done {time.time()-t0:.0f}s  AUC={auc:.4f}  PR-AUC={ap:.4f}  "
          f"(last full-split run: AUC=0.9319)")

    t0 = time.time()
    m_short_tr = (y_tr > 0) & (y_tr <= 1800)
    idx_s = np.where(m_short_tr)[0]
    _L = train_reg(X3.iloc[idx_s], y_tr[m_short_tr], V3, y_ho, ROUNDS_REG)
    p_short = _L[0]; seeds_short = _L[1]
    print(f"    short head done {time.time()-t0:.0f}s n_train={idx_s.size:,} "
          f"seedRMSE={['%.2f' % v for v in seeds_short]}")

    t0 = time.time()
    m_long_tr = (y_tr > 1800) & (y_tr < 10800)
    idx_l = np.where(m_long_tr)[0]
    # label 保持原值 (不使用任何 floor / 缩放)
    _L = train_reg(X3.iloc[idx_l], y_tr[m_long_tr], V3, y_ho, ROUNDS_REG)
    p_long_raw = _L[0]; seeds_long = _L[1]
    print(f"    long head done {time.time()-t0:.0f}s n_train={idx_l.size:,} "
          f"seedRMSE={['%.2f' % v for v in seeds_long]}")
    # 仅推理时用 t_unimp_r7 做下限
    p_long = np.maximum(p_long_raw, np.nan_to_num(t_ho, nan=0.0))

    P_lc = np.clip(P_long, 0.0, 1.0)
    p4 = np.clip(p_short * (1.0 - P_lc) + p_long * P_lc, 30.0, 10800.0)
    _LAST_SEEDS = seeds_short
    results["g4_twostage"] = {**sl(p4), "per_ap": per_ap(p4),
                              "cls_auc": auc, "cls_pr_auc": ap,
                              "n_pred_long_thr05": int((P_lc > 0.5).sum()),
                              "seed_rmse_long_head": [round(v, 3) for v in seeds_long],
                              "n_train_short": int(idx_s.size), "n_train_long": int(idx_l.size),
                              "per_ap_g3_long": per_ap(p3),
                              "per_ap_g4_long": per_ap(p4)}
    print(f"    G4 core={results['g4_twostage']['rmse_core']:.2f} "
          f"long={results['g4_twostage']['rmse_long']:.2f} "
          f"short={results['g4_twostage']['rmse_short']:.2f}")

    # 汇总表
    print(f"\n  === {name} 四组对照 ===")
    print(f"  {'组':22s} {'短段':>9s} {'长段':>9s} {'核心':>9s} {'全样本':>9s} {'seedσ':>7s}")
    for k, lbl in [("g1_base71", "① 71特征"), ("g2_offset", "② +offset"),
                   ("g3_trust", "③ +trust"), ("g4_twostage", "④ 两阶段")]:
        r = results[k]
        print(f"  {lbl:22s} {r['rmse_short']:>9.2f} {r['rmse_long']:>9.2f} "
              f"{r['rmse_core']:>9.2f} {r['rmse_full']:>9.2f} {r['seed_std']:>7.2f}")

    print(f"\n  === {name} 重点机场 核心RMSE ===")
    print(f"  {'机场':8s} " + " ".join(f"{l:>16s}" for l in ["①base", "②offset", "③trust", "④两阶段"]))
    for apx in FOCUS_AP:
        cells = []
        for k in ["g1_base71", "g2_offset", "g3_trust", "g4_twostage"]:
            v = results[k]["per_ap"].get(apx, {}).get("rmse_core")
            cells.append(f"{v:>16.1f}" if v is not None else f"{'-':>16s}")
        print(f"  {apx:8s} " + " ".join(cells))

    out["results"] = results
    return out


if __name__ == "__main__":
    print("=" * 74)
    print(f"SMOKE={SMOKE} seeds={SEEDS} rounds_reg={ROUNDS_REG} rounds_cls={ROUNDS_CLS}")
    print("=" * 74)
    all_out = {}
    for sp in RUN_SPLITS:
        all_out[sp] = run_split(sp, *SPLITS[sp])
        fn = (WORK / "ttt_smoke.json") if SMOKE else (D0 / "tunimp_trust_twostage_result.json")
        fn.write_text(json.dumps(all_out, ensure_ascii=False, indent=2, default=float))
        print(f"\n  >> partial saved: {fn.name}", flush=True)
    print("\nDONE")
