"""
[2] Land the innovative method on the final four months -> supplementary/final_uncertainty.parquet
==========================================================================
- Reuse the LGB point prediction (read submissions/jubilant-puzzle_final.parquet directly, no recomputation)
- σ_N: NGBoost (_ngb_reduced.pkl) predicted on FEATS71 rebuilt on final_ranking
- μ_C = MVT − SCHED, σ_C = 100 (best by sensitivity)
- σ² = p·σ_C² + (1−p)·σ_N² + p(1−p)·(μ_C−μ_N)²,  σ_N truncated by min(σ_N, 2000)
- Intervals: 90%/95% stratified q_C/q_N from the calibration set (mo 3,4,5,8,9,10,11)
- Output: MVT_ID_mvt, point_pred, sigma_total, is_damaged, lo_90, hi_90, lo_95, hi_95

This is a repo supplementary artifact, it does not go into the scored parquet; point_pred is not modified.
"""
from __future__ import annotations
import sys, json, time, pickle, os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(D0))
os.chdir(ROOT)

SUB = ROOT / "submissions" / "jubilant-puzzle_final.parquet"
SUPP = ROOT / "supplementary" / "final_uncertainty.parquet"
META = D0 / "final_uncertainty_meta.json"
CALQ = D0 / "_calib_q.json"

SIG_C = 100.0
SIG_CAP = 2000.0
LIRF_AP, LIRF_T = "LIRF", 14400.0

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:6.1f}s] {m}", flush=True)


def main():
    from final_predict_d import (TRAIN_COLS, TIME_COLS, FEATS71, HIGH_CARD_CATS,
                                 LOW_CARD_CATS, CAT_MAP, build_features, encode_X_cp,
                                 to_sec)
    from src.data_loader import load_month
    import _r7_lib as L

    # ---------- [1] stratified q of the calibration set (σ_C=100) ----------
    lap("[1] Compute the stratified q of the calibration set (σ_C=%g)" % SIG_C)
    if CALQ.exists():
        qs = json.loads(CALQ.read_text())
        lap(f"  reusing the file already on disk {CALQ.name}")
    else:
        from ngb_probe_reduced import load_cache
        X, y, mo, mvt_s, ap = load_cache()
        ts = pickle.load(open(D0 / "_ts_cache.pkl", "rb"))
        clf = pickle.load(open(D0 / "cp_mixture_models" / "clf.pkl", "rb"))
        ngb = pickle.load(open(D0 / "_ngb_reduced.pkl", "rb"))
        CAL = [3, 4, 5, 8, 9, 10, 11]
        i_cal = np.where(np.isin(mo, CAL))[0]
        p = np.clip(clf.predict(X.iloc[i_cal], num_iteration=clf.best_iteration), 0, 1)
        z = np.load(D0 / "_moment_ngb_preds.npz")
        mu_C = ts["mvt_s"][i_cal] - ts["sch_s"][i_cal]
        afn = ts["adep_flt_null"][i_cal]
        apc = ap[i_cal].astype(str)
        corrupt = ((apc == LIRF_AP) & afn & (mu_C > LIRF_T)) | (p > 0.5)
        sc_n = np.minimum(z["sc_cal"], SIG_CAP)
        mu = p * mu_C + (1 - p) * z["loc_cal"]
        var = (p * SIG_C**2 + (1 - p) * sc_n**2
               + p * (1 - p) * (mu_C - z["loc_cal"]) ** 2)
        s = np.abs(y[i_cal] - mu) / np.sqrt(var)
        qs = {f"{t}_{int(l*100)}": round(float(np.quantile(
                 s[corrupt if t == "C" else ~corrupt], l)), 6)
              for t in ["C", "N"] for l in [0.90, 0.95]}
        qs["_n_calib"] = int(len(i_cal))
        qs["_n_corrupt"] = int(corrupt.sum())
        qs["_sig_c"] = SIG_C
        qs["_sig_cap"] = SIG_CAP
        CALQ.write_text(json.dumps(qs, indent=2))
        del X, y, ngb
        lap(f"  q = {qs}")
    for k in ["C_90", "N_90", "C_95", "N_95"]:
        assert k in qs, k

    # ---------- [2]/[3] final four-month features + NGBoost σ_N (with cache) ----------
    FCP = D0 / "_final_ngb_sigma.npz"
    XCP = D0 / "_final_Xcp.npy"
    SID = D0 / "_final_ids_timestamps.npz"
    XCP_CACHE_OK = FCP.exists() and XCP.exists() and SID.exists()
    if XCP_CACHE_OK:
        z = np.load(FCP)
        mu_N, sig_N_raw = z["mu_N"], z["sig_N_raw"]
        zs = np.load(SID)
        mvt_id, mvt, sch, afn_np = (zs["mvt_id"], zs["mvt"], zs["sch"],
                                     zs["afn"].astype(bool))
        ap = zs["ap"]
        lap(f"  reusing cache {FCP.name} + {XCP.name} + {SID.name}")
    else:
        lap("[2] Rebuild final FEATS71 (LUT fitted on 2025 fit months only)")
        train_months = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
        tr = []
        for m in train_months:
            fn = (f"data/raw/training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
                  if m < 12 else "data/raw/training_2025-12-01_2026-01-01.parquet")
            tr.append(load_month(fn, columns=TRAIN_COLS))
        df_tr_raw = pd.concat(tr, ignore_index=True)
        del tr
        df_clean = df_tr_raw[(df_tr_raw["PHASE_mvt"] == "DEP") &
                             (df_tr_raw["TAXITIME_SEC_mvt"] > 0) &
                             (df_tr_raw["TAXITIME_SEC_mvt"] <= 10800)].copy()
        lut = L.build_t_unimp_lut(df_clean)
        lap(f"  LUT training-fold DEP = {len(df_clean):,}")

        fin = pd.read_parquet("data/raw/final_ranking.parquet")
        fin_dep = fin[fin["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
        assert fin_dep["TAXITIME_SEC_mvt"].isna().all()
        assert fin_dep["BLOCK_TIME_UTC_mvt"].isna().all()
        lap(f"  final DEP = {len(fin_dep):,}")

        fin_full = build_features(fin_dep, df_tr_raw, lut, label="final")
        fd = fin_full[fin_full["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
        Xcp = encode_X_cp(fd)
        lap(f"  FEATS71 = {Xcp.shape}")
        np.save(XCP, Xcp.to_numpy(dtype=np.float32))

        lap("[3] NGBoost predicts σ_N (final four months)")
        ngb = pickle.load(open(D0 / "_ngb_reduced.pkl", "rb"))
        par = np.asarray(ngb.pred_param(Xcp.to_numpy(dtype=np.float32)),
                         dtype=np.float64)
        mu_N, sig_N_raw = par[:, 0], np.exp(par[:, 1])
        np.savez(FCP, mu_N=mu_N, sig_N_raw=sig_N_raw)

        mvt_id = fd["MVT_ID_mvt"].to_numpy().astype(np.float64)
        mvt = to_sec(fd["MVT_TIME_UTC_mvt"])
        sch = to_sec(fd["SCHED_TIME_UTC_mvt"])
        afn_np = (fd["ADEP_flt"].isna()
                  | (fd["ADEP_flt"].astype(str).str.strip() == "")).to_numpy()
        ap = fd["ADEP_mvt"].astype(str).to_numpy()
        np.savez(SID, mvt_id=mvt_id, mvt=mvt, sch=sch, afn=afn_np, ap=ap)
        del fin_full, fin, fin_dep, df_tr_raw, df_clean
    assert np.isfinite(mu_N).all() and np.isfinite(sig_N_raw).all()
    assert (sig_N_raw > 0).all()
    n_trunc = int((sig_N_raw > SIG_CAP).sum())
    sig_N = np.minimum(sig_N_raw, SIG_CAP)
    lap(f"  σ_N: median={np.median(sig_N):.2f} n_truncated={n_trunc:,} "
        f"({n_trunc/len(sig_N)*100:.3f}%)")

    # ---------- [4] p and corruption marking ----------
    PCP = D0 / "_final_copy_prob.npy"
    if PCP.exists() and XCP_CACHE_OK:
        p = np.load(PCP)
        lap(f"  reusing the file already on disk {PCP.name}")
    else:
        lap("[4] Copy classifier p")
        clf = pickle.load(open(D0 / "cp_mixture_models" / "clf.pkl", "rb"))
        p = np.clip(clf.predict(Xcp, num_iteration=clf.best_iteration), 0, 1)
        np.save(PCP, p)

    mu_C = mvt - sch
    is_hit = (ap == LIRF_AP) & afn_np & (mu_C > LIRF_T)
    is_damaged = is_hit | (p > 0.5)
    lap(f"  T14400 hits={int(is_hit.sum()):,}  corrupt rows (p>0.5 or hit)={int(is_damaged.sum()):,}")

    # ---------- [5] mixed σ ----------
    var = (p * SIG_C**2 + (1 - p) * sig_N**2
           + p * (1 - p) * (mu_C - mu_N) ** 2)
    sig = np.sqrt(var)
    mu_mix = p * mu_C + (1 - p) * mu_N
    lap(f"  σ_total: median={np.median(sig):.2f} max={sig.max():.4e} "
        f">1e4={int((sig>1e4).sum()):,}")

    # ---------- [6] stratified conformal intervals ----------
    qC90, qN90 = qs["C_90"], qs["N_90"]
    qC95, qN95 = qs["C_95"], qs["N_95"]
    qq90 = np.where(is_damaged, qC90, qN90)
    qq95 = np.where(is_damaged, qC95, qN95)

    # ---------- [7] assemble the output (point_pred read from the submission file, unmodified) ----------
    lap("[7] Read in the point predictions and assemble the supplementary output")
    sub = pd.read_parquet(SUB)
    assert len(sub) == len(mvt_id), f"n_rows {len(sub)} vs {len(mvt_id)}"
    assert np.array_equal(sub["MVT_ID_mvt"].to_numpy().astype(np.float64),
                          mvt_id), "MVT_ID is not aligned"
    point = sub["TAXITIME_SEC_mvt"].to_numpy().astype(np.int64)

    out = pd.DataFrame({
        "MVT_ID_mvt": mvt_id.astype(np.int64),
        "point_pred": point,
        "sigma_total": np.round(sig, 3),
        "is_damaged": is_damaged,
        "lo_90": np.round(np.maximum(point - qq90 * sig, 0.0), 1),
        "hi_90": np.round(point + qq90 * sig, 1),
        "lo_95": np.round(np.maximum(point - qq95 * sig, 0.0), 1),
        "hi_95": np.round(point + qq95 * sig, 1),
    })
    SUPP.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SUPP, index=False)
    lap(f"  ✓ written {SUPP} ({SUPP.stat().st_size:,} bytes)")

    # ---------- [8] report ----------
    mo = pd.to_datetime(pd.Series(mvt), unit="s").dt.month.to_numpy()
    print("\n" + "=" * 74)
    print("[report] uncertainty over the final four months")
    print("=" * 74)
    print(f"  n_rows = {len(out):,}")
    print(f"  σ_total: min={sig.min():.1f} p25={np.percentile(sig,25):.1f} "
          f"median={np.median(sig):.1f} p75={np.percentile(sig,75):.1f} "
          f"p95={np.percentile(sig,95):.1f} max={sig.max():.4e}")
    print(f"  σ >1e4 = {int((sig>1e4).sum()):,}   "
          f"(NGBoost raw on h1 was 7.08e14 / 431 rows)")
    print(f"  σ median (>1800 band)={np.median(sig[sig>1800]):.1f}  "
          f"(<600 band)={np.median(sig[sig<600]):.1f}  "
          f"ratio={np.median(sig[sig>1800])/np.median(sig[sig<600]):.2f}x")
    print(f"\n  corrupt rows = {int(is_damaged.sum()):,}  "
          f"(T14400 hits {int(is_hit.sum()):,} + p>0.5 {int((p>0.5).sum()):,})")
    for tag, msk in [("corrupt rows", is_damaged), ("normal rows", ~is_damaged)]:
        w90 = (out["hi_90"] - out["lo_90"])[msk]
        w95 = (out["hi_95"] - out["lo_95"])[msk]
        print(f"    {tag} n={int(msk.sum()):>7,}  "
              f"90% median interval width={w90.median():>9,.0f}s  "
              f"95% median interval width={w95.median():>9,.0f}s  "
              f"σ median={np.median(sig[msk]):.1f}")
    print(f"\n  per-month:")
    for mm in [1, 2, 6, 7]:
        m = mo == mm
        w95 = (out["hi_95"] - out["lo_95"])[m]
        print(f"    mo={mm}: n={int(m.sum()):>7,}  σ median={np.median(sig[m]):>8.1f}  "
              f"σ max={sig[m].max():>10.1f}  95% median interval width={w95.median():>9,.0f}s  "
              f"corrupt={int(is_damaged[m].sum()):>4}")

    assert out["MVT_ID_mvt"].is_unique
    assert not out.isna().any().any()
    print(f"\n  ✓ no NaN / MVT_ID unique / {len(out):,} rows")
    print(f"  [criterion] q_C90={qC90} q_N90={qN90} q_C95={qC95} q_N95={qN95}")

    META.write_text(json.dumps({
        "output": str(SUPP), "n_rows": int(len(out)),
        "sig_c": SIG_C, "sigma_n_cap": SIG_CAP,
        "calib_q": qs,
        "n_damaged": int(is_damaged.sum()),
        "n_t14400_hit": int(is_hit.sum()),
        "n_p_gt_05": int((p > 0.5).sum()),
        "sigma_total": {"median": round(float(np.median(sig)), 2),
                        "p95": round(float(np.percentile(sig, 95)), 2),
                        "max": float(sig.max()),
                        "n_gt_1e4": int((sig > 1e4).sum()),
                        "ratio_gt1800_over_lt600": round(float(
                            np.median(sig[sig > 1800]) / np.median(sig[sig < 600])), 3)},
        "sigma_n_truncated": n_trunc,
        "interval_median_width": {
            "damaged_90": round(float((out["hi_90"] - out["lo_90"])[is_damaged].median()), 1),
            "damaged_95": round(float((out["hi_95"] - out["lo_95"])[is_damaged].median()), 1),
            "normal_90": round(float((out["hi_90"] - out["lo_90"])[~is_damaged].median()), 1),
            "normal_95": round(float((out["hi_95"] - out["lo_95"])[~is_damaged].median()), 1)},
        "per_month": {f"mo{mm}": {"n": int((mo == mm).sum()),
                                 "sigma_median": round(float(np.median(sig[mo == mm])), 1),
                                 "sigma_max": round(float(sig[mo == mm].max()), 1),
                                 "n_damaged": int(is_damaged[mo == mm].sum())}
                      for mm in [1, 2, 6, 7]},
        "note": "repo supplementary artifact, not part of the scored parquet; point_pred directly reuses the point prediction from submissions, unchanged",
        "total_time_min": round((time.time() - _t0) / 60, 1),
    }, indent=2, ensure_ascii=False))
    lap(f"\n→ {META}")


if __name__ == "__main__":
    main()
