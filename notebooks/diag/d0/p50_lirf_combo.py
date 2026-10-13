"""
p50_lirf_combo.py
  step1: 干净口径 holdout(1+7) 三组对比
         a) P50 alone   b) P50 + LIRF T14400   c) L2 + LIRF T14400
  step2: 全年(2025 12月)重训 P50 3seed → ranking 推理 + LIRF 修复 → parquet
红线: 不改 label、不 clip 真值、LUT 只用训练折、ns 时间戳、分支隔离、不上传
"""
from __future__ import annotations
import json, pickle, sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "p50_lirf_combo.json"
CACHE = D0 / "_qinv_feats.pkl"
RAWCACHE = D0 / "_gate_rawcols.pkl"
PRED1 = D0 / "_combo_step1_preds.pkl"
NS = 1_000_000_000
CAP = 2000
SEEDS = [42, 123, 2024]
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
PARAMS = {
    "metric": "rmse", "learning_rate": 0.0147555939449363, "num_leaves": 127,
    "max_depth": 11, "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, float("inf"))]
BLAB = ["[0,600)", "[600,1800)", "[1800,3600)", "[3600,10800]", ">10800"]
T_LIRF = 14400
_t0 = time.time()
def log(*a): print(" ".join(str(x) for x in a), flush=True)
def lap(m=""): log(f"[{time.time()-_t0:7.1f}s] {m}")


def ev(P, yt, ap_h, mo_h, tag):
    core = (yt > 0) & (yt <= 10800)
    r = lambda msk: float(np.sqrt(np.mean((P[msk] - yt[msk]) ** 2)))
    o = {"tag": tag, "full": round(r(np.ones(len(yt), bool)), 2), "core": round(r(core), 2),
         "jan": round(r(mo_h == 1), 2), "jul": round(r(mo_h == 7), 2), "bins": [],
         "per_airport_core": {}}
    for (lo, hi), lab in zip(BINS, BLAB):
        msk = (yt >= lo) & (yt < hi) if hi != float("inf") else (yt >= lo)
        e = {"bin": lab, "n": int(msk.sum()),
             "rmse": round(r(msk), 2) if msk.sum() >= 5 else None}
        if msk.sum() >= 5:
            pm, tm = float(np.median(P[msk])), float(np.median(yt[msk]))
            e.update(pred_med=round(pm, 1), true_med=round(tm, 1),
                     ratio=round(pm / tm, 4) if tm > 0 else None, bias=round(pm - tm, 1))
        o["bins"].append(e)
    for a in pd.Series(ap_h).value_counts().index:
        msk = (ap_h == a) & core
        if msk.sum() >= 500:
            o["per_airport_core"][str(a)] = {"n": int(msk.sum()), "rmse": round(r(msk), 2)}
    return o


def step1():
    with open(CACHE, "rb") as f:
        blob = pickle.load(f)
    X = blob["X"]; y = blob["y"]; mo = blob["mo"]; ap = blob["ap"]; mvt_s = blob["mvt_s"]
    X = X[[c for c in X.columns if not c.startswith("q_")]]
    rc = pickle.load(open(RAWCACHE, "rb"))
    raw = pd.concat([rc[m].assign(month=m) for m in sorted(rc)], ignore_index=True)
    def tos(s):
        v = pd.to_datetime(s, utc=True, errors="coerce")
        return v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64") // NS
    sched_s = tos(raw["SCHED_TIME_UTC_mvt"]).to_numpy(np.float64)
    mvt2 = tos(raw["MVT_TIME_UTC_mvt"]).to_numpy(np.float64)
    phys_h_all = mvt2 - sched_s
    def _null(s):
        v = s.astype(str).str.strip()
        return s.isna().to_numpy() | v.isin(["", "None", "nan", "NaT"]).to_numpy()
    nm_h_all = _null(raw["ADEP_flt"])
    lap(f"X={X.shape}  物理量区间 [{np.nanmin(phys_h_all):.0f}, {np.nanmax(phys_h_all):.0f}]")

    tr = np.isin(mo, TRAIN_MONTHS); ho = np.isin(mo, HOLD_MONTHS)
    idx_tr = np.where(tr)[0]
    order = idx_tr[np.argsort(mvt_s[idx_tr], kind="stable")]
    cut = int(len(order) * 0.9)
    fit_idx, es_idx = order[:cut], order[cut:]
    es_months = sorted(set(mo[es_idx].tolist()))
    assert 1 not in es_months and 7 not in es_months, f"ES 含 holdout 月 {es_months}"
    ho_idx = np.where(ho)[0]
    lap(f"拟合 {len(fit_idx):,} / ES {len(es_idx):,} {es_months} ✓ / holdout {len(ho_idx):,}")

    if PRED1.exists():
        d = pickle.load(open(PRED1, "rb")); lap("step1 预测缓存命中")
    else:
        d = {}
        for name, extra in [("P50", {"objective": "quantile", "alpha": 0.5}),
                            ("L2", {"objective": "regression"})]:
            ps, infos = [], []
            for sd in SEEDS:
                t1 = time.time()
                p = dict(PARAMS, **extra); p["seed"] = sd
                p["bagging_seed"] = sd + 1; p["feature_fraction_seed"] = sd + 2
                dtr = lgb.Dataset(X.iloc[fit_idx], label=y[fit_idx], categorical_feature=LOW_CARD_CATS)
                dva = lgb.Dataset(X.iloc[es_idx], label=y[es_idx], reference=dtr)
                b = lgb.train(p, dtr, num_boost_round=CAP, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
                ps.append(np.clip(b.predict(X.iloc[ho_idx], num_iteration=b.best_iteration).astype(np.float64),
                                  30.0, 10800.0))
                infos.append({"seed": sd, "best_iter": int(b.best_iteration),
                              "sec": round(time.time() - t1, 1)})
                lap(f"  {name} seed{sd}: best_iter={b.best_iteration} ({time.time()-t1:.0f}s)")
            d[name] = {"preds": ps, "info": infos}
        pickle.dump(d, open(PRED1, "wb"))
    return d, y[ho_idx], ap[ho_idx], mo[ho_idx], phys_h_all[ho_idx], nm_h_all[ho_idx]


def main():
    # ================= STEP 1 =================
    d, yt, ap_h, mo_h, phys_h, nm_h = step1()
    P50 = np.mean(d["P50"]["preds"], axis=0)
    L2 = np.mean(d["L2"]["preds"], axis=0)
    lirf_m = (ap_h == "LIRF") & nm_h & (phys_h > T_LIRF)
    lap(f"LIRF 修复命中 {int(lirf_m.sum())} 行")
    P50r = P50.copy(); P50r[lirf_m] = phys_h[lirf_m]
    L2r = L2.copy();   L2r[lirf_m] = phys_h[lirf_m]

    res = {"n_holdout": int(len(yt)), "T_lirf": T_LIRF, "n_lirf_changed": int(lirf_m.sum()),
           "p50_info": d["P50"]["info"], "l2_info": d["L2"]["info"]}
    c_msk = (yt > 0) & (yt <= 10800)
    for nm, P, seedkey in [("a_P50", P50, "P50"), ("b_P50_LIRF", P50r, "P50"),
                           ("c_L2_LIRF", L2r, "L2")]:
        res[nm] = ev(P, yt, ap_h, mo_h, nm)
        res[nm]["per_seed_core"] = [round(float(np.sqrt(np.mean(
            (p[c_msk] - yt[c_msk]) ** 2))), 2) for p in d[seedkey]["preds"]]
    res["l2_alone"] = ev(L2, yt, ap_h, mo_h, "L2_alone")

    log("\n" + "=" * 100)
    log("STEP 1 · 干净口径 holdout(1+7) 三组对比")
    log("=" * 100)
    log(f"{'组':<14} {'full':>9} {'core':>9} {'Jan':>9} {'Jul':>9} {'各seed core':>22} {'std':>7}")
    for nm, lab in [("l2_alone", "L2 alone"), ("a_P50", "a) P50 alone"),
                    ("b_P50_LIRF", "b) P50+LIRF"), ("c_L2_LIRF", "c) L2+LIRF")]:
        e = res[nm]
        sc = e.get("per_seed_core", [])
        log(f"{lab:<14} {e['full']:>9.2f} {e['core']:>9.2f} {e['jan']:>9.2f} {e['jul']:>9.2f} "
            f"{str(sc):>22} {np.std(sc) if sc else 0:>7.2f}")
    log(f"\nLIRF 修复净收益:  P50 {res['a_P50']['full']:.2f} → {res['b_P50_LIRF']['full']:.2f} "
        f"(Δ={res['b_P50_LIRF']['full']-res['a_P50']['full']:+.2f})")
    log(f"                core {res['a_P50']['core']:.2f} → {res['b_P50_LIRF']['core']:.2f} "
        f"(Δ={res['b_P50_LIRF']['core']-res['a_P50']['core']:+.2f})")
    log(f"P50+LIRF vs L2+LIRF:  full Δ={res['b_P50_LIRF']['full']-res['c_L2_LIRF']['full']:+.2f}  "
        f"core Δ={res['b_P50_LIRF']['core']-res['c_L2_LIRF']['core']:+.2f}")

    log("\n" + "=" * 100); log("分档 (长滑行低估)"); log("=" * 100)
    log(f"{'档':>14} {'n':>8} | {'L2alone':>9} {'P50':>9} {'P50+LIRF':>9} | "
        f"{'L2 ratio':>9} {'P50 ratio':>10} {'P50L ratio':>11}")
    for b1, b2, b3 in zip(res["l2_alone"]["bins"], res["a_P50"]["bins"], res["b_P50_LIRF"]["bins"]):
        if b1["rmse"] is None: continue
        log(f"{b1['bin']:>14} {b1['n']:>8,} | {b1['rmse']:>9.2f} {b2['rmse']:>9.2f} {b3['rmse']:>9.2f} | "
            f"{b1['ratio']:>9.4f} {b2['ratio']:>10.4f} {b3['ratio']:>11.4f}")
    log("\n分机场 core:")
    ks = ["l2_alone", "a_P50", "b_P50_LIRF", "c_L2_LIRF"]
    log(f"{'机场':>6} " + " ".join(f"{k:>12}" for k in ks))
    for a in sorted(res["a_P50"]["per_airport_core"]):
        row = [res[k]["per_airport_core"].get(a, {}).get("rmse", float("nan")) for k in ks]
        log(f"{a:>6} " + " ".join(f"{v:>12.2f}" for v in row))
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
    lap(f"→ {OUT}")
    lap("STEP 1 完成")


if __name__ == "__main__":
    main()
