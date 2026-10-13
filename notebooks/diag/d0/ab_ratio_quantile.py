"""
ab_ratio_quantile.py — 实验A(比率目标 y/ref) + 实验B(分位数门控)

共用: _qinv_feats.pkl 特征缓存, 干净口径 (train 2-6,8-12 / ES 11-12 / holdout 1+7)
红线: 不改 label、不 clip 真值、P20/ref 只用训练折、时间戳 ns
"""
from __future__ import annotations
import json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "ab_ratio_quantile.json"
CACHE = D0 / "_qinv_feats.pkl"
RAWCACHE = D0 / "_gate_rawcols.pkl"
NS = 1_000_000_000
CAP = 2000
SEEDS = [42, 123, 2024]
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
BASE_PARAMS = {
    "metric": "rmse", "learning_rate": 0.0147555939449363, "num_leaves": 127,
    "max_depth": 11, "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, float("inf"))]
BLAB = ["[0,600)", "[600,1800)", "[1800,3600)", "[3600,10800]", ">10800"]
_t0 = time.time()
def log(*a): print(" ".join(str(x) for x in a), flush=True)
def lap(m=""): log(f"[{time.time()-_t0:7.1f}s] {m}")


def main():
    with open(CACHE, "rb") as f:
        blob = pickle.load(f)
    X = blob["X"]; y = blob["y"]; mo = blob["mo"]; ap = blob["ap"]; mvt_s = blob["mvt_s"]
    lap(f"X={X.shape}")

    # ---- 标准列: FEATS71 (去掉 12 个 q_* 反演量, 它们已被判定无增益) ----
    q_cols = [c for c in X.columns if c.startswith("q_")]
    cols71 = [c for c in X.columns if c not in q_cols]
    X = X[cols71]
    lap(f"用 {len(cols71)} 列 (剔除 {len(q_cols)} 个 q_* 反演量)")

    tr = np.isin(mo, TRAIN_MONTHS); ho = np.isin(mo, HOLD_MONTHS)
    idx_tr = np.where(tr)[0]
    order = idx_tr[np.argsort(mvt_s[idx_tr], kind="stable")]
    cut = int(len(order) * 0.9)
    fit_idx, es_idx = order[:cut], order[cut:]
    es_months = sorted(set(mo[es_idx].tolist()))
    assert 1 not in es_months and 7 not in es_months, f"ES 含 holdout 月 {es_months}"
    ho_idx = np.where(ho)[0]
    lap(f"拟合 {len(fit_idx):,} / ES {len(es_idx):,} {es_months} ✓ / holdout {len(ho_idx):,}")

    # ================= unimp_ref (P20, 只用训练折) =================
    rc = pickle.load(open(RAWCACHE, "rb"))
    raw = pd.concat([rc[m].assign(month=m) for m in sorted(rc)], ignore_index=True)
    assert len(raw) == len(mo)
    def tos(s):
        v = pd.to_datetime(s, utc=True, errors="coerce")
        return v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64") // NS
    raw["_stand"] = X["STAND_mvt"].to_numpy()
    raw["_rwy"] = X["RUNWAY_mvt"].to_numpy()
    apm = pd.DataFrame({"ap": ap, "y": y, "mo": mo, "stand": raw["_stand"].to_numpy(),
                        "rwy": raw["_rwy"].to_numpy()})
    TR = apm[apm["mo"].isin(TRAIN_MONTHS)]          # P20 只用训练月
    g3 = TR.groupby(["ap", "stand", "rwy"])["y"]
    p20_3 = g3.quantile(0.20); n3 = g3.count()
    g2 = TR.groupby(["ap", "rwy"])["y"]
    p20_2 = g2.quantile(0.20); n2 = g2.count()
    p20_ap = TR.groupby("ap")["y"].median()
    gmed = TR["y"].median()
    log(f"unimp_ref: 3级组 {len(p20_3):,}  2级组 {len(p20_2):,}  机场 {len(p20_ap)}")

    def lookup_ref(a, st, rw):
        k3 = pd.MultiIndex.from_arrays([a, st, rw])
        k2 = pd.MultiIndex.from_arrays([a, rw])
        v = pd.Series(p20_3.reindex(k3).to_numpy(), index=np.arange(len(a)))
        cnt = pd.Series(n3.reindex(k3).to_numpy(), index=np.arange(len(a))).fillna(0).to_numpy()
        v2 = pd.Series(p20_2.reindex(k2).to_numpy(), index=np.arange(len(a)))
        cnt2 = pd.Series(n2.reindex(k2).to_numpy(), index=np.arange(len(a))).fillna(0).to_numpy()
        va = pd.Series(p20_ap.reindex(a).to_numpy(), index=np.arange(len(a)))
        v = v.where(cnt >= 5)
        v = v.fillna(v2.where(cnt2 >= 5)).fillna(va).fillna(gmed)
        return np.maximum(v.to_numpy(np.float64), 30.0)

    ref = lookup_ref(ap, raw["_stand"].to_numpy(), raw["_rwy"].to_numpy())
    lap(f"unimp_ref 分位: p10={np.percentile(ref,10):.0f} p50={np.percentile(ref,50):.0f} "
        f"p90={np.percentile(ref,90):.0f}")
    ratio = y / ref
    lap(f"ratio 分位: p10={np.percentile(ratio,10):.3f} p50={np.percentile(ratio,50):.3f} "
        f"p90={np.percentile(ratio,90):.3f}")
    # 训练折内自洽性检查 (P20 自身不应被自身标签污染到失真)
    lap(f"  训练月 内 ratio p50={np.median(ratio[tr]):.3f}  真实 y p50={np.median(y[tr]):.0f}")

    # ================= 评估框架 =================
    yt = y[ho_idx]; ap_h = ap[ho_idx]; mo_h = mo[ho_idx]
    core = (yt > 0) & (yt <= 10800)
    def ev(P, tag, ref_h=None):
        r = lambda msk: float(np.sqrt(np.mean((P[msk] - yt[msk]) ** 2)))
        o = {"tag": tag, "full": round(r(np.ones(len(yt), bool)), 2), "core": round(r(core), 2),
             "jan": round(r(mo_h == 1), 2), "jul": round(r(mo_h == 7), 2),
             "bins": [], "per_airport_core": {}}
        for (lo, hi), lab in zip(BINS, BLAB):
            msk = (yt >= lo) & (yt < hi) if hi != float("inf") else (yt >= lo)
            e = {"bin": lab, "n": int(msk.sum()),
                 "rmse": round(r(msk), 2) if msk.sum() >= 5 else None}
            if msk.sum() >= 5:
                pm = float(np.median(P[msk])); tm = float(np.median(yt[msk]))
                e["pred_med"] = round(pm, 1); e["true_med"] = round(tm, 1)
                e["ratio_pred_true"] = round(pm / tm, 4) if tm > 0 else None
                e["bias"] = round(pm - tm, 1)
            o["bins"].append(e)
        for a in pd.Series(ap_h).value_counts().index:
            msk = (ap_h == a) & core
            if msk.sum() >= 500:
                o["per_airport_core"][str(a)] = {"n": int(msk.sum()), "rmse": round(r(msk), 2)}
        return o

    def train_pred(target, params, tag, seed_list=SEEDS, transform=None):
        ps, infos = [], []
        for sd in seed_list:
            t1 = time.time()
            p = dict(params); p["seed"] = sd
            p["bagging_seed"] = sd + 1; p["feature_fraction_seed"] = sd + 2
            dtr = lgb.Dataset(X.iloc[fit_idx], label=target[fit_idx], categorical_feature=LOW_CARD_CATS)
            dva = lgb.Dataset(X.iloc[es_idx], label=target[es_idx], reference=dtr)
            b = lgb.train(p, dtr, num_boost_round=CAP, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            pr = b.predict(X.iloc[ho_idx], num_iteration=b.best_iteration).astype(np.float64)
            if transform is not None:
                pr = transform(pr)
            ps.append(np.clip(pr, 30.0, 10800.0))
            infos.append({"seed": sd, "best_iter": int(b.best_iteration),
                          "sec": round(time.time() - t1, 1)})
            lap(f"  {tag} seed{sd}: best_iter={b.best_iteration} ({time.time()-t1:.0f}s)")
        return ps, infos

    res = {"unimp_ref": {"p10": float(np.percentile(ref, 10)),
                         "p50": float(np.percentile(ref, 50)),
                         "p90": float(np.percentile(ref, 90)),
                         "n_g3": int(len(p20_3)), "n_g2": int(len(p20_2))},
           "ratio": {"p10": float(np.percentile(ratio, 10)),
                     "p50": float(np.percentile(ratio, 50)),
                     "p90": float(np.percentile(ratio, 90))}}

    # ---------- 基线 (L2 on y) ----------
    lap("=== 基线 L2 on y, 3 seed ===")
    Pb, ib = train_pred(y, dict(BASE_PARAMS, objective="regression"), "base")
    res["baseline"] = {"per_seed": ib, "ev": ev(np.mean(Pb, axis=0), "baseline_ens"),
                       "per_seed_core": [round(float(np.sqrt(np.mean((P[core] - yt[core])**2))), 2) for P in Pb]}
    log(f"  基线 ens: full={res['baseline']['ev']['full']:.2f} core={res['baseline']['ev']['core']:.2f}")
    log(f"  各 seed core: {res['baseline']['per_seed_core']}  "
        f"std={np.std(res['baseline']['per_seed_core']):.2f}")

    # ---------- 实验 A: 比率目标 ----------
    lap("=== 实验A: ratio 目标, 3 seed ===")
    ref_h = ref[ho_idx]
    Pa, ia = train_pred(ratio, dict(BASE_PARAMS, objective="regression"), "ratio",
                        transform=lambda pr: pr * ref_h)
    res["expA_ratio"] = {"per_seed": ia, "ev": ev(np.mean(Pa, axis=0), "ratio_ens"),
                         "per_seed_core": [round(float(np.sqrt(np.mean((P[core] - yt[core])**2))), 2) for P in Pa]}
    log(f"  A ens: full={res['expA_ratio']['ev']['full']:.2f} core={res['expA_ratio']['ev']['core']:.2f}")
    log(f"  各 seed core: {res['expA_ratio']['per_seed_core']}  "
        f"std={np.std(res['expA_ratio']['per_seed_core']):.2f}")

    # ---------- 实验 B: 分位数 ----------
    qp = {}
    for name, al in [("P50", 0.5), ("P80", 0.8)]:
        lap(f"=== 实验B: {name} 分位数 (alpha={al}), 3 seed ===")
        pq_, iq = train_pred(y, dict(BASE_PARAMS, objective="quantile", alpha=al), name)
        qp[name] = (np.mean(pq_, axis=0), iq)
        res[f"expB_{name}"] = {"alpha": al, "per_seed": iq,
                               "ev": ev(qp[name][0], f"{name}_ens")}
        log(f"  {name} ens: full={res[f'expB_{name}']['ev']['full']:.2f} "
            f"core={res[f'expB_{name}']['ev']['core']:.2f}")
    P50, P80 = qp["P50"][0], qp["P80"][0]

    # ================= 报告 =================
    log("\n" + "=" * 104)
    log("实验A: 比率目标  y/ref  (ref = 训练折 (ADEP,STAND,RWY) P20)")
    log("=" * 104)
    log(f"{'档':>14} {'n':>8} | {'基线RMSE':>10} {'A_RMSE':>10} {'Δ':>9} | "
        f"{'基线pred/true':>13} {'A_pred/true':>12} {'A改善':>9}")
    for b_, a_ in zip(res["baseline"]["ev"]["bins"], res["expA_ratio"]["ev"]["bins"]):
        if b_["rmse"] is None: continue
        d = a_["ratio_pred_true"] - b_["ratio_pred_true"]
        log(f"{b_['bin']:>14} {b_['n']:>8,} | {b_['rmse']:>10.2f} {a_['rmse']:>10.2f} "
            f"{a_['rmse']-b_['rmse']:>+9.2f} | {b_['ratio_pred_true']:>13.4f} "
            f"{a_['ratio_pred_true']:>12.4f} {d:>+9.4f}")
    bf, af = res["baseline"]["ev"]["full"], res["expA_ratio"]["ev"]["full"]
    bc, ac = res["baseline"]["ev"]["core"], res["expA_ratio"]["ev"]["core"]
    log(f"\nfull  {bf:>8.2f} → {af:>8.2f}  Δ={af-bf:+.2f}")
    log(f"core  {bc:>8.2f} → {ac:>8.2f}  Δ={ac-bc:+.2f}")
    log(f"Jan   {res['baseline']['ev']['jan']:>8.2f} → {res['expA_ratio']['ev']['jan']:>8.2f}   "
        f"Jul {res['baseline']['ev']['jul']:>8.2f} → {res['expA_ratio']['ev']['jul']:>8.2f}")
    pa = res["baseline"]["ev"]["per_airport_core"]; qa = res["expA_ratio"]["ev"]["per_airport_core"]
    com = sorted(set(pa) & set(qa))
    ab_ = sum(1 for a in com if qa[a]["rmse"] < pa[a]["rmse"])
    log(f"\n分机场 core (A 更优 {ab_}/{len(com)}):")
    for a in sorted(com, key=lambda z: qa[z]["rmse"] - pa[z]["rmse"]):
        log(f"  {a:>5} {pa[a]['rmse']:>9.2f} → {qa[a]['rmse']:>9.2f}  Δ={qa[a]['rmse']-pa[a]['rmse']:>+9.2f}")
    res["expA_verdict"] = {"d_full": round(af - bf, 2), "d_core": round(ac - bc, 2),
                           "ap_better": ab_, "ap_total": len(com)}

    log("\n" + "=" * 104)
    log("实验B: 分位数门控  pred = P50 + k*(P80-P50)")
    log("=" * 104)
    res["expB_sweep"] = []
    log(f"{'k':>5} {'full':>10} {'core':>10} {'Jan':>10} {'Jul':>10} {'[1800,3600)':>12} {'[3600,10800]':>13}")
    for k in [0.0, 0.5, 1.0, 1.5, 2.0]:
        P = np.clip(P50 + k * (P80 - P50), 30.0, 10800.0)
        e = ev(P, f"k={k}")
        b18 = next(b for b in e["bins"] if b["bin"] == "[1800,3600)")
        b36 = next(b for b in e["bins"] if b["bin"] == "[3600,10800]")
        log(f"{k:>5.1f} {e['full']:>10.2f} {e['core']:>10.2f} {e['jan']:>10.2f} {e['jul']:>10.2f} "
            f"{b18['rmse']:>12.2f} {b36['rmse']:>13.2f}")
        res["expB_sweep"].append({"k": k, "full": e["full"], "core": e["core"],
                                  "jan": e["jan"], "jul": e["jul"],
                                  "bin_1800_3600": b18["rmse"], "bin_3600_10800": b36["rmse"],
                                  "ev": e})
    best = min(res["expB_sweep"], key=lambda d: d["full"])
    log(f"\n最优 k = {best['k']}  full={best['full']:.2f} (基线 {bf:.2f}, Δ={best['full']-bf:+.2f})")
    res["expB_best_k"] = best["k"]
    res["expB_verdict"] = {"best_k": best["k"], "d_full": round(best["full"] - bf, 2),
                           "d_core": round(best["core"] - bc, 2)}
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
    lap(f"→ {OUT}")

if __name__ == "__main__":
    main()
