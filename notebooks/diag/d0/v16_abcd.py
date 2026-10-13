"""
v16 加码实验: 三档信任 + t_unimp 基准偏移 + 双向二阶校正 + PSI 漂移
=====================================================================
基线: v15 71 特征, Jan<->Jul 互验, 3seed LGB
红线: 不改 label / 不 clip 真值 / 不建单机场模型 / 不删脏行

输出: notebooks/diag/d0/v16_abcd_result.json
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3] / "Documents" / "PRC_data_challenge"
if not ROOT.exists():
    ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D

OUT = D0 / "v16_abcd_result.json"

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
    "ref_was_estimated", "mvt_n_ahead", "mvt_mu_rwy_15", "mvt_mu_rwy_30",
    "mvt_w_actual_15", "mvt_w_actual_30",
    "mvt_q_peak", "mvt_q_mean",
    "mvt_na_J", "mvt_na_H", "mvt_na_M", "mvt_na_L",
    "mvt_na_heavy_frac", "mvt_pos_rw", "mvt_na_heavy_x_pos",
    "mvt_mu_rwy_5", "mvt_w_actual_5",
    "dep_cnt_m_5", "dep_cnt_m_15", "dep_cnt_m_30",
    "aobt_cnt_m_5", "aobt_cnt_m_15", "aobt_cnt_m_30",
    "eobt_ahead_cnt_5", "eobt_ahead_cnt_15", "eobt_ahead_cnt_30",
    "dly_aobt_eobt", "dly_aobt_iobt", "dly_aobt_sched", "dly_lobt_aobt",
    "t_unimp_r7", "rwy_month_share", "ap_month_hourly_density",
]
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
HIGH_CARD_CATS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                  "airport_hour", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]

# 加码 A 新特征
A_FEATS = ["trust_level", "block_echoes_sched"]
# 加码 B 新特征
B_FEATS = ["t_unimp_bias_offset"]

SEEDS = [42, 123, 2024]
LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
NR = 1200


def log(*a):
    print(*a, flush=True)


# ============================================================
# 特征构建
# ============================================================
def build_features(df_raw, ref_dep_for_d, lut, label=""):
    df = parse_time_columns(df_raw.copy(), cols=TIME_COLS)
    df_basic = build_basic(df)
    df_full = add_queue_features(df_basic.copy(),
        include_a=True, include_b=True, include_c=True,
        include_d=True, include_e=True)
    df_full["ref_was_estimated"] = df_full["AOBT_3_flt"].isna().astype("int8")
    df_full["is_diverted"] = (df_full["ADES_flt"] != df_full["ADES_FILED_flt"]).astype("int8")

    # ---- 加码 A: 三档信任 ----
    aobt = df_full["AOBT_3_flt"]
    block = df_full["BLOCK_TIME_UTC_mvt"]
    d_ab = (aobt - block).dt.total_seconds()
    trust = np.full(len(df_full), 2, dtype="int8")
    trust[aobt.isna().values] = 0
    m1 = (~aobt.isna().values) & (d_ab.abs() > 1800).values
    trust[m1] = 0
    m2 = (~aobt.isna().values) & (d_ab.abs() >= 300).values & (d_ab.abs() <= 1800).values
    trust[m2] = 1
    df_full["trust_level"] = trust
    d_bs = (block - df_full["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    df_full["block_echoes_sched"] = (d_bs.abs() < 60).astype("int8")

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


def add_bias_offset(df_dep, label=""):
    """加码 B: 按 (ADEP, RUNWAY) 在 ref 上算 t_unimp 中位 - 真值中位 的差。
    df_dep 需含 t_unimp_r7 / TAXITIME_SEC_mvt / ADEP_mvt / RUNWAY_mvt。
    ref_dep: 参考帧(训练折)本身。
    """
    g = df_dep.groupby(["ADEP_mvt", "RUNWAY_mvt"], observed=True)
    med_t = g["t_unimp_r7"].median().rename("med_t")
    med_y = g["TAXITIME_SEC_mvt"].median().rename("med_y")
    off = (med_t - med_y)
    log(f"  bias_offset[{label}]: {len(off):,} 组, "
        f"mean={off.mean():.1f}s, p10={off.quantile(.1):.1f}, p90={off.quantile(.9):.1f}")
    keys = pd.MultiIndex.from_arrays([df_dep["ADEP_mvt"].values, df_dep["RUNWAY_mvt"].values])
    out = off.reindex(keys).values.astype("float32")
    return out


def encode(df, cats, fit=False, maps=None):
    X = df.copy()
    if fit:
        maps = {}
        for c in cats:
            if c not in X.columns:
                continue
            vc = X[c].astype("str").value_counts()
            keep = vc[vc >= 30].index
            maps[c] = {v: i for i, v in enumerate(keep)}
    for c in cats:
        if c not in X.columns or c not in maps:
            continue
        X[c] = X[c].astype("str").map(maps[c]).fillna(-1).astype("int32")
    return X, maps


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def train_ens(Xtr, ytr, Xva, seeds=SEEDS, params=None, nr=NR, cat=None):
    ps, imps = [], None
    p = params or LGB_PARAMS
    for s in seeds:
        pp = dict(p); pp["seed"] = s; pp["bagging_seed"] = s; pp["feature_fraction_seed"] = s
        ds = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat, free_raw_data=True)
        dv = lgb.Dataset(Xva, label=None, reference=ds, categorical_feature=cat, free_raw_data=True)
        m = lgb.train(pp, ds, num_boost_round=nr, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        ps.append(m.predict(Xva, num_iteration=m.best_iteration))
        if imps is None:
            imps = m.feature_importance("gain")
    return np.mean(ps, axis=0), np.array(ps), imps


def psi(ref, cur, bins=10):
    r = np.asarray(ref, dtype="float64"); c = np.asarray(cur, dtype="float64")
    ok = ~(np.isnan(r) | np.isnan(c))
    r, c = r[ok], c[ok]
    if len(r) < 100 or len(c) < 100:
        return float("nan")
    qs = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    pr = np.histogram(r, bins=qs)[0] / len(r)
    pc = np.histogram(c, bins=qs)[0] / len(c)
    pr = np.clip(pr, 1e-6, None); pc = np.clip(pc, 1e-6, None)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


# ============================================================
# [0] 加载
# ============================================================
log("=" * 70); log("[0] 加载 Jan + Jul"); log("=" * 70)
t0 = time.time()
dfs = [load_month(f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet",
                  columns=TRAIN_COLS) for m in (1, 7)]
raw = pd.concat(dfs, ignore_index=True)
del dfs

jan_raw = raw[raw["MVT_TIME_UTC_mvt"].dt.month == 1].reset_index(drop=True)
jul_raw = raw[raw["MVT_TIME_UTC_mvt"].dt.month == 7].reset_index(drop=True)
raw = None

# LUT 用两月合并池 (仅用于特征工程参照, 非 label 泄露: 只用 MVT/BLOCK/SCHED 时间与 ADEP/RUNWAY)
pool = pd.concat([jan_raw, jul_raw], ignore_index=True)
pool = parse_time_columns(pool, cols=TIME_COLS)
pool_dep = pool[(pool["PHASE_mvt"] == "DEP") &
                (pool["TAXITIME_SEC_mvt"] > 0) &
                (pool["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
lut = L.build_t_unimp_lut(pool_dep)
log(f"  pool DEP {len(pool_dep):,}  {time.time()-t0:.0f}s")

log("  build Jan ...")
jan_full = build_features(jan_raw, pool_dep, lut, label="jan")
jan_dep = jan_full[(jan_full["PHASE_mvt"] == "DEP") &
                   (jan_full["TAXITIME_SEC_mvt"] > 0) &
                   (jan_full["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
del jan_full, jan_raw

log("  build Jul ...")
jul_full = build_features(jul_raw, pool_dep, lut, label="jul")
jul_dep = jul_full[(jul_full["PHASE_mvt"] == "DEP") &
                   (jul_full["TAXITIME_SEC_mvt"] > 0) &
                   (jul_full["TAXITIME_SEC_mvt"] < 10800)].reset_index(drop=True)
del jul_full, jul_raw, pool, pool_dep

log(f"  Jan DEP {len(jan_dep):,} | Jul DEP {len(jul_dep):,}  {time.time()-t0:.0f}s")

MONTHS = {"jan": jan_dep, "jul": jul_dep}
Y = {k: v["TAXITIME_SEC_mvt"].values.astype("float64") for k, v in MONTHS.items()}

# 加码 B: bias_offset 用【对方月份】为 ref (模拟推理时用历史池)
for tgt, ref in (("jan", "jul"), ("jul", "jan")):
    MONTHS[tgt]["t_unimp_bias_offset"] = add_bias_offset(MONTHS[ref], label=f"ref={ref}")

RESULT = {"meta": {"n_jan": len(jan_dep), "n_jul": len(jul_dep),
                   "feats71_n": len(FEATS71), "seeds": SEEDS}}

# 类别集合 (两折统一)
ALL_CATS = sorted(set(LOW_CARD_CATS + HIGH_CARD_CATS + A_FEATS + ["pred_bin"]))
CAT_FOR_TRAIN = sorted(set(LOW_CARD_CATS + HIGH_CARD_CATS + A_FEATS))

# ============================================================
# Fold 循环
# ============================================================
FOLDS = [("A", "jan", "jul"), ("B", "jul", "jan")]


def run_fold(fold_name, tr_m, va_m, feats, tag):
    tr, va = MONTHS[tr_m], MONTHS[va_m]
    ytr, yva = Y[tr_m], Y[va_m]
    Xtr, maps = encode(tr[feats], CAT_FOR_TRAIN, fit=True)
    Xva, _ = encode(va[feats], CAT_FOR_TRAIN, fit=False, maps=maps)
    cat = [c for c in CAT_FOR_TRAIN if c in Xtr.columns]
    p_ens, p_seeds, imps = train_ens(Xtr, ytr, Xva, cat=cat)
    per_seed = [rmse(yva, p) for p in p_seeds]
    out = {"rmse": rmse(yva, p_ens), "per_seed": per_seed,
           "seed_jitter": float(max(per_seed) - min(per_seed)),
           "seed_std": float(np.std(per_seed))}
    # gain
    gn = dict(zip(feats, imps.tolist()))
    tot = sum(gn.values()) or 1
    out["gain_top8"] = sorted(
        [{"f": k, "gain": round(v, 1), "share_pct": round(v / tot * 100, 3)}
         for k, v in gn.items() if v > 0], key=lambda x: -x["gain"])[:8]
    out["new_feat_gain"] = {f: {"gain": round(gn.get(f, 0.0), 1),
                                "share_pct": round(gn.get(f, 0.0) / tot * 100, 3)}
                            for f in A_FEATS + B_FEATS}
    # 分 trust_level
    tl = va["trust_level"].values
    out["by_trust"] = {}
    for lv in (0, 1, 2):
        m = tl == lv
        if m.sum() > 100:
            out["by_trust"][str(lv)] = {"n": int(m.sum()), "rmse": round(rmse(yva[m], p_ens[m]), 3),
                                        "bias": round(float(np.mean(p_ens[m] - yva[m])), 3)}
    # 分 pred_bin (用基线预测分档, 5 档)
    qs = np.quantile(p_ens, np.linspace(0, 1, 6)); qs[0], qs[-1] = -np.inf, np.inf
    pb = np.digitize(p_ens, qs[1:-1])
    out["by_predbin"] = {}
    for b in range(5):
        m = pb == b
        if m.sum() > 100:
            out["by_predbin"][str(b)] = {"n": int(m.sum()),
                                         "rmse": round(rmse(yva[m], p_ens[m]), 3),
                                         "bias": round(float(np.mean(p_ens[m] - yva[m])), 3),
                                         "mean_pred": round(float(p_ens[m].mean()), 1),
                                         "mean_true": round(float(yva[m].mean()), 1)}
    # 分机场 top
    ap = va["ADEP_mvt"].values
    out["by_airport"] = {}
    for a in pd.unique(ap):
        m = ap == a
        if m.sum() >= 500:
            out["by_airport"][str(a)] = round(rmse(yva[m], p_ens[m]), 2)
    log(f"  [{tag}/{fold_name}] {tr_m}->{va_m} RMSE={out['rmse']:.3f} "
        f"seeds={[round(x,2) for x in per_seed]} jitter={out['seed_jitter']:.2f}")
    return out, p_ens, pb


log("")
log("=" * 70); log("[基线] v15 71 特征 (3seed LGB, Jan<->Jul 互验)"); log("=" * 70)
base = {}
PRED = {}
for fn, trm, vam in FOLDS:
    o, p, pb = run_fold(fn, trm, vam, FEATS71, "base")
    base[fn] = o; PRED[(fn, "base")] = (p, pb)
BASE_MEAN = float(np.mean([base[f]["rmse"] for f, _, _ in FOLDS]))
log(f"  >>> 基线两折均值 RMSE = {BASE_MEAN:.3f}")

# 加码 A
log("")
log("=" * 70); log("[加码A] 三档信任特征"); log("=" * 70)
resA = {}
for fn, trm, vam in FOLDS:
    o, p, pb = run_fold(fn, trm, vam, FEATS71 + A_FEATS, "A")
    resA[fn] = o; PRED[(fn, "A")] = (p, pb)
A_MEAN = float(np.mean([resA[f]["rmse"] for f, _, _ in FOLDS]))
log(f"  >>> A 两折均值 RMSE = {A_MEAN:.3f}  (Δ={A_MEAN-BASE_MEAN:+.3f})")

# 加码 B (在 A 之上叠加, 也单独跑 B-only)
log("")
log("=" * 70); log("[加码B] t_unimp_bias_offset (A-only / A+B)"); log("=" * 70)
resB_only = {}
for fn, trm, vam in FOLDS:
    o, p, pb = run_fold(fn, trm, vam, FEATS71 + B_FEATS, "B-only")
    resB_only[fn] = o
B_ONLY_MEAN = float(np.mean([resB_only[f]["rmse"] for f, _, _ in FOLDS]))
log(f"  >>> B-only 两折均值 RMSE = {B_ONLY_MEAN:.3f}  (Δ={B_ONLY_MEAN-BASE_MEAN:+.3f})")

resAB = {}
for fn, trm, vam in FOLDS:
    o, p, pb = run_fold(fn, trm, vam, FEATS71 + A_FEATS + B_FEATS, "A+B")
    resAB[fn] = o; PRED[(fn, "AB")] = (p, pb)
AB_MEAN = float(np.mean([resAB[f]["rmse"] for f, _, _ in FOLDS]))
log(f"  >>> A+B 两折均值 RMSE = {AB_MEAN:.3f}  (Δ={AB_MEAN-BASE_MEAN:+.3f})")

# ============================================================
# 加码 C: 二阶校正器 (用 A+B 作基模型)
# ============================================================
log("")
log("=" * 70); log("[加码C] 双向二阶校正器 (基模型 = A+B)"); log("=" * 70)
SECOND_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.02, "num_leaves": 31, "max_depth": 6,
    "min_data_in_leaf": 40, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 1.0, "reg_lambda": 2.0,
    "verbose": -1, "num_threads": -1,
}
SEC_FEATS = ["base_pred", "trust_level", "t_unimp_bias_offset", "pred_bin",
             "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
             "dep_mvt_cnt_airport_15m", "dep_queue_n_airport",
             "dep_queue_n_runway", "mvt_n_ahead", "sched_hour", "sched_dow"]
SEC_CATS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "trust_level", "pred_bin"]

resC = {}
for fn, trm, vam in FOLDS:
    tr, va = MONTHS[trm], MONTHS[vam]
    ytr, yva = Y[trm], Y[vam]
    # --- 训练折内 OOF 基预测 (按月分折, 时间安全) ---
    tr_months = sorted(tr["MVT_TIME_UTC_mvt"].dt.month.unique())
    oof = np.full(len(tr), np.nan)
    for i, m in enumerate(tr_months):
        inner_tr = tr[tr["MVT_TIME_UTC_mvt"].dt.month != m]
        inner_va = tr[tr["MVT_TIME_UTC_mvt"].dt.month == m]
        if len(inner_va) < 500 or len(inner_tr) < 5000:
            continue
        Xi, mp = encode(inner_tr[FEATS71 + A_FEATS + B_FEATS], CAT_FOR_TRAIN, fit=True)
        Xj, _ = encode(inner_va[FEATS71 + A_FEATS + B_FEATS], CAT_FOR_TRAIN, fit=False, maps=mp)
        cj = [c for c in CAT_FOR_TRAIN if c in Xi.columns]
        pj, _, _ = train_ens(Xi, inner_tr["TAXITIME_SEC_mvt"].values.astype("float64"),
                             Xj, cat=cj, nr=600)
        oof[inner_va.index] = pj
    ok = ~np.isnan(oof)
    if ok.sum() < 5000:
        # 单月无法内折, 退化: 用全训练折拟合的基模型在训练折上的预测
        Xi, mp = encode(tr[FEATS71 + A_FEATS + B_FEATS], CAT_FOR_TRAIN, fit=True)
        p_tr, _, _ = train_ens(Xi, ytr, Xi, cat=[c for c in CAT_FOR_TRAIN if c in Xi.columns], nr=600)
        oof = p_tr
        ok = np.ones(len(tr), bool)
    log(f"  [{fn}] OOF 基预测覆盖 {ok.sum():,}/{len(tr):,}")

    # 构造二阶训练集
    Str = tr[SEC_FEATS].copy()
    Str["base_pred"] = oof
    qb = np.quantile(oof[ok], np.linspace(0, 1, 6))
    Str["pred_bin"] = np.digitize(oof, np.clip(qb[1:-1], -np.inf, np.inf)).astype("int8")

    # 验证集用真基模型预测
    pva, _ = PRED[(fn, "AB")]
    Sva = va[SEC_FEATS].copy()
    Sva["base_pred"] = pva
    Sva["pred_bin"] = np.digitize(pva, np.clip(qb[1:-1], -np.inf, np.inf)).astype("int8")

    Str = Str[ok].reset_index(drop=True)
    ytr2 = ytr[ok]
    Xtr2, mp2 = encode(Str, SEC_CATS, fit=True)
    Xva2, _ = encode(Sva, SEC_CATS, fit=False, maps=mp2)
    c2 = [c for c in SEC_CATS if c in Xtr2.columns]
    p2_ens, p2_seeds, _ = train_ens(Xtr2, ytr2, Xva2, params=SECOND_PARAMS, nr=800, cat=c2)

    r_direct = rmse(yva, pva)
    r_corr = rmse(yva, p2_ens)
    per_seed = [rmse(yva, p) for p in p2_seeds]
    # 分档
    pb = Sva["pred_bin"].values
    by_bin = {}
    for b in range(5):
        m = pb == b
        if m.sum() > 100:
            by_bin[str(b)] = {
                "n": int(m.sum()),
                "rmse_direct": round(rmse(yva[m], pva[m]), 2),
                "rmse_corr": round(rmse(yva[m], p2_ens[m]), 2),
                "delta": round(rmse(yva[m], p2_ens[m]) - rmse(yva[m], pva[m]), 2),
                "bias_direct": round(float(np.mean(pva[m] - yva[m])), 2),
                "bias_corr": round(float(np.mean(p2_ens[m] - yva[m])), 2),
            }
    ap = va["ADEP_mvt"].values
    by_ap = {}
    for a in pd.unique(ap):
        m = ap == a
        if m.sum() >= 500:
            by_ap[str(a)] = {"direct": round(rmse(yva[m], pva[m]), 2),
                             "corr": round(rmse(yva[m], p2_ens[m]), 2)}
    resC[fn] = {"rmse_direct": r_direct, "rmse_corr": r_corr,
                "delta": r_corr - r_direct,
                "corr_per_seed": per_seed,
                "corr_seed_jitter": float(max(per_seed) - min(per_seed)),
                "by_predbin": by_bin, "by_airport": by_ap}
    log(f"  [{fn}] 直出 {r_direct:.3f} -> 校正 {r_corr:.3f}  (Δ={r_corr-r_direct:+.3f})")

C_DELTA = float(np.mean([resC[f]["delta"] for f, _, _ in FOLDS]))
log(f"  >>> C 两折平均 Δ = {C_DELTA:+.3f}")

# ============================================================
# 加码 D: PSI 漂移 (2025 vs 2026 ranking)
# ============================================================
log("")
log("=" * 70); log("[加码D] PSI 漂移检查 2025 vs 2026"); log("=" * 70)
try:
    rank = pd.read_parquet("data/raw/ranking.parquet")
    rank_dep = rank[rank["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    rank_full = build_features(rank_dep, pool_dep if False else None, lut, label="rank")
    psi_res = {}
    for f in FEATS71 + A_FEATS:
        if f not in jan_dep.columns or f not in rank_full.columns:
            continue
        a, b = jan_dep[f], rank_full[f]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            v = psi(a.values, b.values)
            if not np.isnan(v):
                psi_res[f] = round(v, 4)
    red = {k: v for k, v in psi_res.items() if abs(v) > 0.1}
    log(f"  PSI 计算 {len(psi_res)} 个特征, |PSI|>0.1 标红 {len(red)} 个")
    for k, v in sorted(red.items(), key=lambda x: -abs(x[1])):
        log(f"    🔴 {k}: PSI={v}")
    top = sorted(psi_res.items(), key=lambda x: -abs(x[1]))[:12]
    log("  PSI 绝对值 top12:")
    for k, v in top:
        log(f"    {'🔴' if abs(v)>0.1 else '  '} {k}: {v}")
    del rank, rank_dep, rank_full
except Exception as e:
    log(f"  ⚠️ PSI 失败: {e}")
    psi_res, red = {}, {}

# ============================================================
# 汇总
# ============================================================
JIT = 0.5
def verdict(delta, jitter):
    if delta >= -JIT:
        return f"不采纳 (Δ={delta:+.2f}s 未超抖动 ±{JIT}s)"
    return f"采纳 (Δ={delta:+.2f}s > 抖动 ±{JIT}s)"

RESULT.update({
    "baseline": {"mean_rmse": round(BASE_MEAN, 3), "folds": base},
    "addA": {"mean_rmse": round(A_MEAN, 3), "delta": round(A_MEAN - BASE_MEAN, 3),
             "verdict": verdict(A_MEAN - BASE_MEAN, max(base[f]["seed_jitter"] for f, _, _ in FOLDS)),
             "folds": resA},
    "addB_only": {"mean_rmse": round(B_ONLY_MEAN, 3), "delta": round(B_ONLY_MEAN - BASE_MEAN, 3),
                  "verdict": verdict(B_ONLY_MEAN - BASE_MEAN, max(resB_only[f]["seed_jitter"] for f, _, _ in FOLDS)),
                  "folds": resB_only},
    "addA_plus_B": {"mean_rmse": round(AB_MEAN, 3), "delta": round(AB_MEAN - BASE_MEAN, 3),
                    "folds": resAB},
    "addC_second_order": {"mean_delta": round(C_DELTA, 3),
                          "verdict": verdict(C_DELTA, max(resC[f]["corr_seed_jitter"] for f, _, _ in FOLDS)),
                          "folds": resC},
    "addD_psi": {"n_computed": len(psi_res), "n_red": len(red),
                 "red_features": red,
                 "all": psi_res},
    "seed_jitter_threshold": JIT,
})
OUT.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log("=" * 70)
log(f"基线           {BASE_MEAN:8.3f}")
log(f"+A 三档信任    {A_MEAN:8.3f}  Δ={A_MEAN-BASE_MEAN:+.3f}")
log(f"+B only        {B_ONLY_MEAN:8.3f}  Δ={B_ONLY_MEAN-BASE_MEAN:+.3f}")
log(f"+A+B           {AB_MEAN:8.3f}  Δ={AB_MEAN-BASE_MEAN:+.3f}")
log(f"+C 二阶校正    Δ={C_DELTA:+.3f}")
log(f"D 漂移: {len(red)} 个特征 |PSI|>0.1")
log(f"→ {OUT}")
log("=" * 70)
