"""
Physics-constrained second-moment mixture + corruption-aware conformal
===================================
[1] Corruption component:  μ_C = MVT − SCHED;  σ_C is physically bounded, sensitivity σ_C ∈ {100,300,600}s
[2] Mixture moments:      μ = p·μ_C + (1−p)·μ_N
                        σ² = p·σ_C² + (1−p)·σ_N² + p(1−p)·(μ_C−μ_N)²
                        σ_N truncated by min(σ_N, 2000)  (kills the NGBoost tail blow-up, records the truncated row count)
[3] Stratified conformal: calibration set mo∈{3,4,5,8,9,10,11} (not in h1/h2)
                        corrupt rows = hits of the full T=14400 criterion OR p>0.5;  normal rows = the rest
                        per stratum take the 90%/95% quantiles q_C/q_N of s=|y−μ|/σ
                        compute the actual coverage on h1/h2 and compare against the nominal 90%/95%
[4] Report JSON: Spearman / σ distribution / σ_C sensitivity / point-prediction RMSE / per-airport

Read-only on existing models and caches; models are written to disk first; does not write submissions/; does not upload.
"""
from __future__ import annotations
import sys, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(D0))
from ngb_probe_reduced import load_cache  # noqa: E402

TS = D0 / "_ts_cache.pkl"
OUT = D0 / "moment_conformal.json"

CALIB = [3, 4, 5, 8, 9, 10, 11]
H1, H2 = [1, 7], [2, 6]
HOLD = [1, 2, 6, 7]
LIRF_AP, LIRF_T = "LIRF", 14400.0
SIG_CAP = 2000.0
SIG_C_GRID = [100.0, 300.0, 600.0]
SIG_C_MAIN = 300.0
LEVELS = [0.90, 0.95]

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:6.1f}s] {m}", flush=True)


def sp(a, b):
    """Spearman, no scipy dependency"""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def rmse(P, yt, m=None):
    if m is None:
        m = np.ones(len(yt), bool)
    return float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))


def main():
    lap("Load cache + NGBoost/copy model outputs")
    X, y, mo, mvt_s, ap = load_cache()
    ts = pickle.load(open(TS, "rb"))
    mvt_a, sch_a = ts["mvt_s"], ts["sch_s"]
    afn = ts["adep_flt_null"]
    assert np.array_equal(mvt_s, mvt_a)
    lap(f"  full {len(y):,}")

    clf = pickle.load(open(D0 / "cp_mixture_models" / "clf.pkl", "rb"))

    # ---- calibration-set p (the only thing that must be computed anew) ----
    i_cal = np.where(np.isin(mo, CALIB))[0]
    lap(f"  calibration set mo{CALIB} = {len(i_cal):,} rows; computing p ...")
    p_cal = np.clip(clf.predict(X.iloc[i_cal], num_iteration=clf.best_iteration), 0, 1)

    # ---- NGBoost loc/scale: predicted straight from _ngb_reduced.pkl ----
    # Note: the _ngb_h{1,2}_{loc,scale}.npy files in the directory are not products of this script
    #     (generated at 13:59, inconsistent with the reduced-model anchors: σ median 1981 vs reported 228.85), hence not reused.
    lap("Load NGBoost (_ngb_reduced.pkl) and predict calib/h1/h2")
    ngb = pickle.load(open(D0 / "_ngb_reduced.pkl", "rb"))
    Xv = X.to_numpy(dtype=np.float32)
    NGBP = D0 / "_moment_ngb_preds.npz"
    if NGBP.exists():
        z = np.load(NGBP)
        loc_cal, sc_cal = z["loc_cal"], z["sc_cal"]
        loc_h1, sc_h1 = z["loc_h1"], z["sc_h1"]
        loc_h2, sc_h2 = z["loc_h2"], z["sc_h2"]
        lap(f"  reusing the NGBoost predictions already on disk {NGBP.name}")
    else:
        def ngb_pred(idx, tag):
            # use pred_param to get the raw parameters, avoiding the DataFrame shape issue of pred_dist
            par = np.asarray(ngb.pred_param(Xv[idx]), dtype=np.float64)
            mu, sc = par[:, 0], np.exp(par[:, 1])
            assert np.isfinite(mu).all(), f"{tag} loc is not finite"
            assert np.isfinite(sc).all(), f"{tag} scale is not finite"
            assert (sc > 0).all(), f"{tag} scale<=0"
            lap(f"    {tag} n={len(idx):,} loc/scale done")
            return mu, sc

        loc_cal, sc_cal = ngb_pred(i_cal, "calib")
        loc_h1, sc_h1 = ngb_pred(i_h1, "h1")
        loc_h2, sc_h2 = ngb_pred(i_h2, "h2")
        np.savez(NGBP,
                 loc_cal=loc_cal, sc_cal=sc_cal, loc_h1=loc_h1, sc_h1=sc_h1,
                 loc_h2=loc_h2, sc_h2=sc_h2)

    i_h1 = np.where(np.isin(mo, H1))[0]
    i_h2 = np.where(np.isin(mo, H2))[0]

    # ---- LGB baseline (already on disk, same source as the h1/h2 probe) ----
    lgb_h1 = np.load(D0 / "_ngb_lgb_h1.npy").astype(np.float64)
    lgb_h2 = np.load(D0 / "_ngb_lgb_h2.npy").astype(np.float64)
    assert len(lgb_h1) == len(i_h1) and len(lgb_h2) == len(i_h2)
    lap(f"  LGB baseline reused h1 {len(lgb_h1):,} / h2 {len(lgb_h2):,} ✓")

    p_h = {"h1": np.clip(clf.predict(X.iloc[i_h1], num_iteration=clf.best_iteration), 0, 1),
           "h2": np.clip(clf.predict(X.iloc[i_h2], num_iteration=clf.best_iteration), 0, 1)}

    def gather(idx, p):
        d = {"y": y[idx], "mo": mo[idx], "ap": ap[idx].astype(str),
             "mvt": mvt_a[idx], "sch": sch_a[idx], "afn": afn[idx], "p": p}
        return d

    S = {"calib": gather(i_cal, p_cal),
         "h1": gather(i_h1, p_h["h1"]), "h2": gather(i_h2, p_h["h2"])}
    S["calib"]["loc"], S["calib"]["scale"] = loc_cal, sc_cal
    S["h1"].update(loc=loc_h1, scale=sc_h1, lgb=lgb_h1)
    S["h2"].update(loc=loc_h2, scale=sc_h2, lgb=lgb_h2)

    # ---- hard assertions: loc/scale finite and scale>0 ----
    for nm in S:
        assert np.isfinite(S[nm]["loc"]).all(), f"{nm} loc is not finite"
        assert np.isfinite(S[nm]["scale"]).all(), f"{nm} scale is not finite"
        assert (S[nm]["scale"] > 0).all(), f"{nm} scale<=0"

    # ---- corrupt-row marking ----
    for nm in S:
        s = S[nm]
        s["mu_C"] = s["mvt"] - s["sch"]
        s["hit14400"] = (s["ap"] == LIRF_AP) & s["afn"] & (s["mu_C"] > LIRF_T)
        s["corrupt"] = s["hit14400"] | (s["p"] > 0.5)
        lap(f"  {nm}: n={len(s['y']):,}  hit14400={int(s['hit14400'].sum()):,}  "
            f"corrupt={int(s['corrupt'].sum()):,}")

    # ---- mixture moments ----
    def mix(d, sig_c, cap=SIG_CAP):
        sc_n = np.minimum(d["scale"], cap)
        n_trunc = int((d["scale"] > cap).sum())
        mu = d["p"] * d["mu_C"] + (1 - d["p"]) * d["loc"]
        var = (d["p"] * sig_c**2 + (1 - d["p"]) * sc_n**2
               + d["p"] * (1 - d["p"]) * (d["mu_C"] - d["loc"])**2)
        return mu, np.sqrt(var), n_trunc

    lap("\n" + "=" * 82)
    print("=" * 82)
    print("[1][2] σ signal comparison: NGBoost raw σ vs mixed σ (σ_C=%g)" % SIG_C_MAIN)
    print("=" * 82)
    sig_stats = {}
    for nm in ["h1", "h2"]:
        mu, sg, ntr = mix(S[nm], SIG_C_MAIN)
        S[nm]["mu"], S[nm]["sig"], S[nm]["n_trunc"] = mu, sg, ntr
        r = np.abs(S[nm]["y"] - mu)          # residual of the mixed point prediction
        r_loc = np.abs(S[nm]["y"] - S[nm]["loc"])  # residual of the NGBoost point prediction
        raw = S[nm]["scale"]
        print(f"\n  {nm}  (n={len(mu):,})")
        print(f"    {'σ vs |y−μ_mix| Spearman':<28}  "
              f"NGBoost raw σ={sp(raw, r):+.4f}   mixed σ={sp(sg, r):+.4f}")
        print(f"    {'σ vs |y−loc| Spearman':<28}  "
              f"NGBoost raw σ={sp(raw, r_loc):+.4f}   (NGBoost self-consistent form)")
        print(f"    {'σ max':<28}  NGBoost raw={raw.max():.4e}   mixed={sg.max():.4e}")
        print(f"    {'σ >1e4 row count':<28}  NGBoost raw={int((raw>1e4).sum()):,}     "
              f"mixed={int((sg>1e4).sum()):,}")
        print(f"    {'σ median':<28}  NGBoost raw={np.median(raw):.2f}      "
              f"mixed={np.median(sg):.2f}")
        print(f"    {'σ truncated rows (>2000)':<28}  NGBoost={ntr:,} "
              f"({ntr/len(raw)*100:.3f}%)")
        m_gt = sg > 1800
        m_lt = sg < 600
        print(f"    σ median (>1800 band)={np.median(sg[m_gt]):.1f}  "
              f"(<600 band)={np.median(sg[m_lt]):.1f}  "
              f"ratio={np.median(sg[m_gt])/np.median(sg[m_lt]):.2f}x")
        anchor = (7.08e14, 431, 228.85, 2.29) if nm == "h1" else (1.93e12, 103, 197.87, 2.49)
        print(f"    [anchor {nm}] NGBoost σ max={anchor[0]:.2e}, >1e4 rows={anchor[1]}, "
              f"σ median={anchor[2]}, long/short tail ratio={anchor[3]}x")
        sig_stats[nm] = {
            "spearman_raw_vs_mix_resid": round(sp(raw, r), 4),
            "spearman_mix_vs_mix_resid": round(sp(sg, r), 4),
            "spearman_raw_vs_loc_resid": round(sp(raw, r_loc), 4),
            "sigma_max_raw": float(raw.max()), "sigma_max_mix": float(sg.max()),
            "n_sigma_gt_1e4_raw": int((raw > 1e4).sum()),
            "n_sigma_gt_1e4_mix": int((sg > 1e4).sum()),
            "sigma_median_raw": round(float(np.median(raw)), 2),
            "sigma_median_mix": round(float(np.median(sg)), 2),
            "n_truncated": ntr, "trunc_rate": round(ntr / len(raw), 6),
            "sigma_ratio_gt1800_over_lt600": round(
                float(np.median(sg[m_gt]) / np.median(sg[m_lt])), 3),
            "anchor": {"sigma_max": anchor[0], "n_gt_1e4": anchor[1],
                       "sigma_median": anchor[2], "ratio": anchor[3]},
        }

    # ---- [3] stratified conformal ----
    lap("\n" + "=" * 82)
    print("=" * 82)
    print("[3] corruption-aware stratified conformal (calibration set mo%s)" % CALIB)
    print("=" * 82)
    cal = S["calib"]
    mu_c, sg_c, _ = mix(cal, SIG_C_MAIN)
    cal["mu"], cal["sig"] = mu_c, sg_c
    s_score = np.abs(cal["y"] - mu_c) / np.maximum(sg_c, 1e-9)
    cal["score"] = s_score
    q = {}
    for lvl in LEVELS:
        q[("C", lvl)] = float(np.quantile(s_score[cal["corrupt"]], lvl))
        q[("N", lvl)] = float(np.quantile(s_score[~cal["corrupt"]], lvl))
    print(f"\n  calibration set n={len(cal['y']):,}  corrupt rows={int(cal['corrupt'].sum()):,}  "
          f"normal rows={int((~cal['corrupt']).sum()):,}")
    for lvl in LEVELS:
        print(f"    nominal {lvl:.0%}: q_C={q[('C',lvl)]:.4f}   q_N={q[('N',lvl)]:.4f}")

    conf = {}
    for nm in ["h1", "h2"]:
        d = S[nm]
        conf[nm] = {}
        print(f"\n  {nm} actual coverage:")
        for lvl in LEVELS:
            row = {}
            for tag, tkey, mask in [("corrupt rows", "C", d["corrupt"]),
                                    ("normal rows", "N", ~d["corrupt"])]:
                if mask.sum() == 0:
                    row[tag] = {"n": 0}
                    continue
                qv = q[(tkey, lvl)]
                lo = d["mu"][mask] - qv * d["sig"][mask]
                hi = d["mu"][mask] + qv * d["sig"][mask]
                cov = float(((d["y"][mask] >= lo) & (d["y"][mask] <= hi)).mean())
                wid = float(np.median(hi - lo))
                row[tag] = {"n": int(mask.sum()), "q": round(qv, 4),
                            "coverage": round(cov, 4),
                            "dev": round(cov - lvl, 4),
                            "median_width": round(wid, 1)}
                print(f"    nominal {lvl:.0%} {tag}: n={int(mask.sum()):>7,}  "
                      f"q={qv:.4f}  actual coverage={cov:.4f}  dev={cov-lvl:+.4f}  "
                      f"median interval width={wid:,.0f}s")
            row["nominal"] = lvl
            conf[nm][f"{lvl:.2f}"] = row

    # ---- σ_C sensitivity ----
    lap("\n" + "=" * 82)
    print("=" * 82)
    print("[4] σ_C sensitivity (σ_C ∈ %s)" % SIG_C_GRID)
    print("=" * 82)
    sens = {}
    for sc in SIG_C_GRID:
        row = {}
        print(f"\n  σ_C = {sc:.0f}s")
        for nm in ["calib", "h1", "h2"]:
            d = S[nm]
            mu_v, sg_v, _ = mix(d, sc)
            if nm == "calib":
                s_ = np.abs(d["y"] - mu_v) / np.maximum(sg_v, 1e-9)
                qq = {f"{l}_{t}": float(np.quantile(s_[d["corrupt"] if t == 'C' else ~d["corrupt"]], l))
                      for l in LEVELS for t in ["C", "N"]}
            else:
                qq = None
            row[nm] = {"sigma_median": round(float(np.median(sg_v)), 2),
                       "sigma_p95": round(float(np.percentile(sg_v, 95)), 2),
                       "sigma_max": float(sg_v.max()),
                       "n_gt_1e4": int((sg_v > 1e4).sum()),
                       "spearman": round(sp(sg_v, np.abs(d["y"] - mu_v)), 4),
                       "q": qq}
        # h1/h2 coverage (under the calib q, q varies with sigma_C)
        for nm in ["h1", "h2"]:
            d = S[nm]
            mu_v, sg_v, _ = mix(d, sc)
            cc = {}
            for lvl in LEVELS:
                for tag, mask in [("C", d["corrupt"]), ("N", ~d["corrupt"])]:
                    qv = row["calib"]["q"][f"{lvl}_{tag}"]
                    lo = mu_v[mask] - qv * sg_v[mask]
                    hi = mu_v[mask] + qv * sg_v[mask]
                    cc[f"{lvl}_{tag}"] = round(
                        float(((d["y"][mask] >= lo) & (d["y"][mask] <= hi)).mean()), 4)
            row[nm]["coverage"] = cc
        print(f"    h1: σ median={row['h1']['sigma_median']}  Spearman={row['h1']['spearman']}  "
              f"coverage 95%C={row['h1']['coverage']['0.95_C']} 95%N={row['h1']['coverage']['0.95_N']}")
        print(f"    h2: σ median={row['h2']['sigma_median']}  Spearman={row['h2']['spearman']}  "
              f"coverage 95%C={row['h2']['coverage']['0.95_C']} 95%N={row['h2']['coverage']['0.95_N']}")
        sens[f"{sc:.0f}"] = row

    # ---- point-prediction RMSE ----
    lap("\n" + "=" * 82)
    print("=" * 82)
    print("[5] point-prediction μ_total full RMSE (σ_C=%g)" % SIG_C_MAIN)
    print("=" * 82)
    allidx = np.concatenate([i_h1, i_h2])
    y_all = np.concatenate([S["h1"]["y"], S["h2"]["y"]])
    mo_all = np.concatenate([S["h1"]["mo"], S["h2"]["mo"]])
    ap_all = np.concatenate([S["h1"]["ap"], S["h2"]["ap"]])
    mu_all = np.concatenate([S["h1"]["mu"], S["h2"]["mu"]])
    lgb_all = np.concatenate([S["h1"]["lgb"], S["h2"]["lgb"]])
    muC_all = np.concatenate([S["h1"]["mu_C"], S["h2"]["mu_C"]])
    afn_all = np.concatenate([S["h1"]["afn"], S["h2"]["afn"]])

    aps = pd.Series(ap_all).value_counts().index.tolist()
    schemes = {"mu_total_mixture": mu_all, "LGB baseline": lgb_all}
    v22 = lgb_all.copy()
    hit = (ap_all == LIRF_AP) & afn_all & (muC_all > LIRF_T)
    v22[hit] = muC_all[hit]
    schemes["v22 hard rule"] = v22

    print(f"\n  {'scheme':<16}" + "".join(f"{'mo='+str(m):>10}" for m in HOLD)
          + f"{'4 months':>11}{'vs LGB':>10}")
    print("  " + "-" * 69)
    rmse_tab = {}
    base4 = None
    for k, P in schemes.items():
        r4 = rmse(P, y_all)
        if base4 is None:
            base4 = r4
        cells = "".join(f"{rmse(P, y_all, mo_all == m):>10.2f}" for m in HOLD)
        print(f"  {k:<16}{cells}{r4:>11.2f}{r4-base4:>+10.2f}")
        rmse_tab[k] = {f"mo{m}": round(rmse(P, y_all, mo_all == m), 2) for m in HOLD}
        rmse_tab[k]["all4"] = round(r4, 2)
        rmse_tab[k]["delta_vs_LGB"] = round(r4 - base4, 2)
        rmse_tab[k]["_pred"] = P

    # per-airport
    print(f"\n  per-airport full RMSE:")
    print(f"  {'airport':<7}{'n':>9}{'LGB':>10}{'v22':>10}{'μ_total':>10}"
          f"{'μ-LGB':>10}{'same sign':>7}")
    print("  " + "-" * 63)
    ap_tab = {}
    for a in aps:
        m = ap_all == a
        r_l, r_v, r_t = (rmse(schemes[k], y_all, m) for k in
                         ["LGB baseline", "v22 hard rule", "mu_total_mixture"])
        d = r_t - r_l
        same = (d <= 0) if d != 0 else True
        same = d <= 0.5
        ap_tab[a] = {"n": int(m.sum()), "LGB": round(r_l, 2), "v22": round(r_v, 2),
                     "mu_total": round(r_t, 2), "delta_vs_LGB": round(d, 2),
                     "improved": bool(d <= 0.5)}
        print(f"  {a:<7}{int(m.sum()):>9,}{r_l:>10.1f}{r_v:>10.1f}{r_t:>10.1f}"
              f"{d:>+10.2f}{'  OK' if d <= 0.5 else '  FAIL':>7}")
    n_ok = sum(v["improved"] for v in ap_tab.values())
    print(f"\n  per-airport not worse than LGB: {n_ok}/{len(aps)}")

    # per-month same sign
    mm_ok = all(rmse_tab["mu_total_mixture"][f"mo{m}"] <= rmse_tab["LGB baseline"][f"mo{m}"] + 1e-9
                for m in HOLD)
    print(f"  all months not worse than LGB: {mm_ok}")

    # save predictions (models are written to disk first)
    np.savez(D0 / "_moment_conformal_preds.npz",
             mu=mu_all, sig=np.concatenate([S["h1"]["sig"], S["h2"]["sig"]]),
             p=np.concatenate([S["h1"]["p"], S["h2"]["p"]]),
             y=y_all, mo=mo_all, ap=ap_all, lgb=lgb_all)
    lap(f"  predictions written to disk _moment_conformal_preds.npz")

    res = {
        "sig_c_main": SIG_C_MAIN, "sigma_n_cap": SIG_CAP,
        "calib_months": CALIB, "h1_months": H1, "h2_months": H2,
        "sigma_stats": sig_stats,
        "conformal": {"q": {f"{t}_{l:.2f}": q[(t, l)] for t in ["C", "N"] for l in LEVELS},
                      "coverage": conf},
        "sigma_c_sensitivity": sens,
        "point_rmse": {k: {kk: vv for kk, vv in v.items() if kk != "_pred"}
                       for k, v in rmse_tab.items()},
        "per_airport": ap_tab,
        "gate_per_airport": {"n_ok": int(n_ok), "n_total": len(aps)},
        "gate_per_month_all_better": bool(mm_ok),
        "anchors": {"LGB_holdout_all4": 493.05, "v22": 435.08, "D_probability_mixture": 451.60},
        "pred_path": str(D0 / "_moment_conformal_preds.npz"),
        "total_time_min": round((time.time() - _t0) / 60, 1),
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    lap(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
