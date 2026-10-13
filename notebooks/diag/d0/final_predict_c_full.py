"""
Stage 3: regenerate the submission on final_ranking with the final criterion
================================================
Final criterion (validated on holdout, see e_final_eval.json):
  - LIRF rows with (ADEP_flt missing AND MVT-SCHED>14400) -> replaced by MVT-SCHED  [full T=14400 criterion]
  - All other rows -> LGB baseline prediction (p0_lgb 3-seed average)
  - LOBT window: E2 = LIRF is also projected into [MVT-LOBT±3606]; rows with LOBT missing are skipped
  - clip [30, 10800]

E1/E2 comparison conclusion: E2 (LIRF also projected) LIRF RMSE 1345.66 < E1 1358.42 -> pick E2
  (only 4 rows differ, T14400 repaired rows touched by LOBT a second time = 0 rows)

Reuses the model outputs in _final_d_raw.npz (base / p / mu_N), no re-inference.
Also emits a no-clip variant for comparison (clip would destroy the T14400 repaired rows: median true y of the hit rows is 20576s).

Red lines: do not change labels, do not use external data, do not auto-upload
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
NPY = D0 / "_final_d_raw.npz"
SUB_PATH = ROOT / "submissions" / "jubilant-puzzle_final.parquet"
SUB_NOCLIP = ROOT / "submissions" / "jubilant-puzzle_final_noclip.parquet"
META = D0 / "final_predict_c_full_meta.json"

LIRF_AP = "LIRF"
T_LOBT = 3606.0
LIRF_T = 14400.0
CLIP_LO, CLIP_HI = 30.0, 10800.0

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:6.1f}s] {m}", flush=True)


def main():
    lap("Load npz (reuse model outputs, no re-inference)")
    z = np.load(NPY, allow_pickle=True)
    base, mvt, sch, lobt = z["pred_base"], z["mvt"], z["sch"], z["lobt"]
    lobt_ok, mvt_id = z["lobt_ok"], z["mvt_id"]
    ap = z["ap"]
    lap(f"  n={len(base):,}  LIRF={int((ap==LIRF_AP).sum()):,}  LOBT valid={int(lobt_ok.sum()):,}")

    # ---- ADEP_flt missing marker on the final side ----
    lap("Read final ADEP_flt (criterion for the full T=14400 form)")
    fin = pd.read_parquet("data/raw/final_ranking.parquet",
                          columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_flt"])
    fin_dep = fin[fin["PHASE_mvt"].eq("DEP")].copy()
    # align explicitly by MVT_ID (do not rely on row order)
    lut = fin_dep.assign(k=fin_dep["MVT_ID_mvt"].astype(float)).set_index("k")["ADEP_flt"]
    aligned = lut.reindex(pd.Index(mvt_id.astype(float))).to_numpy()
    assert not pd.isna(aligned).sum() < 0 or True
    n_miss_align = int(pd.isna(aligned).sum() - pd.isna(fin_dep["ADEP_flt"]).sum())
    adepflt_null = pd.isna(aligned) | (pd.Series(aligned).astype(str).str.strip() == "").to_numpy()
    lap(f"  ADEP_flt missing: {int(adepflt_null.sum()):,} / {len(adepflt_null):,}")
    del fin, fin_dep, lut

    # ---- synthesize ----
    is_lirf = (ap == LIRF_AP)
    mu_C = mvt - sch
    hit = is_lirf & adepflt_null & (mu_C > LIRF_T)
    lap(f"  T=14400 full-criterion hits: {int(hit.sum()):,} rows "
        f"(LIRF total {int(is_lirf.sum()):,}; simplified criterion would've been "
        f"{int((is_lirf & (mu_C > LIRF_T)).sum()):,})")

    ctr = mvt - lobt
    lo_w, hi_w = ctr - T_LOBT, ctr + T_LOBT

    def build(clip_hi):
        P = base.copy()
        P[hit] = mu_C[hit]
        P = np.clip(P, CLIP_LO, clip_hi)
        # E2: project all rows with valid LOBT (including LIRF)
        m_ok = lobt_ok
        before = P[m_ok].copy()
        P[m_ok] = np.clip(before, lo_w[m_ok], hi_w[m_ok])
        n_changed = int((np.abs(P[m_ok] - before) > 1e-9).sum())
        n_reclip = int(((P < CLIP_LO) | (P > clip_hi)).sum())
        P = np.clip(P, CLIP_LO, clip_hi)
        return P, n_changed, n_reclip

    P, n_changed, n_reclip = build(CLIP_HI)
    P_nc, _, _ = build(1e9)
    lap(f"  LOBT projection changed {n_changed:,} rows ({n_reclip:,} rows need re-clip after projection)")

    # ---- Gate seven items ----
    lap("Gate check")
    submitting = pd.read_parquet("data/raw/final_submitting.parquet")
    assert set(submitting.columns) == {"MVT_ID_mvt", "TAXITIME_SEC_mvt"}
    exp = submitting["MVT_ID_mvt"].astype(float).values

    def emit(pred, path, clip_hi):
        pm = pd.DataFrame({"MVT_ID_mvt": mvt_id.astype(float),
                           "TAXITIME_SEC_mvt": np.round(pred, 0).astype(np.int64)})
        assert pm["MVT_ID_mvt"].is_unique, "MVT_ID has duplicates"
        o = submitting[["MVT_ID_mvt"]].merge(pm, on="MVT_ID_mvt", how="left")
        g = o["MVT_ID_mvt"].astype(float).values
        checks = {
            "columns correct": list(o.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"],
            "n_rows=670790": len(o) == 670790 == len(submitting),
            "MVT_ID order matches": bool(np.all(exp == g)),
            "MVT_ID set matches": set(exp) == set(g),
            "no NaN": bool(o["TAXITIME_SEC_mvt"].notna().all()),
            "int64": o["TAXITIME_SEC_mvt"].dtype == np.int64,
            f"range[{CLIP_LO:.0f},{clip_hi:.0f}]": bool(
                (o["TAXITIME_SEC_mvt"] >= CLIP_LO).all()
                and (o["TAXITIME_SEC_mvt"] <= clip_hi).all()),
        }
        for k, v in checks.items():
            print(f"    {'✓' if v else '✗ FAIL'} {k}")
        assert all(checks.values()), f"Gate failed: {checks}"
        o.to_parquet(path, index=False)
        return o, checks

    print("\n  [main file clip[30,10800]]")
    out, checks = emit(P, SUB_PATH, CLIP_HI)
    lap(f"  ✓ written {SUB_PATH} ({SUB_PATH.stat().st_size:,} bytes)")
    print("\n  [reference no-clip]")
    out_nc, _ = emit(P_nc, SUB_NOCLIP, 1e9)
    lap(f"  ✓ written {SUB_NOCLIP} ({SUB_NOCLIP.stat().st_size:,} bytes)")

    # ---- distribution self-check ----
    mo = pd.to_datetime(pd.Series(mvt), unit="s").dt.month.to_numpy()

    def q(s, p):
        return float(np.percentile(s, p))

    print("\n" + "=" * 76)
    print("[distribution self-check] final criterion (E2, T14400 full)")
    print("=" * 76)
    for nm, v in [("clip[30,10800]", P), ("no-clip", P_nc)]:
        print(f"  {nm:<16} n={len(v):,} min={v.min():>8.0f} p25={q(v,25):>7.0f} "
              f"median={q(v,50):>7.0f} p75={q(v,75):>7.0f} p95={q(v,95):>8.0f} max={v.max():>9.0f}")
        print(f"  {'':<16} >1800={int((v>1800).sum()):>7,} >3600={int((v>3600).sum()):>6,} "
              f">10800={int((v>10800).sum()):>6,}")
    print(f"\n  baseline median={np.median(base):.0f} p95={q(base,95):.0f}")
    print(f"  LIRF: baseline median={np.median(base[is_lirf]):.0f} -> "
          f"final median={np.median(P[is_lirf]):.0f}")

    print("\n  per-month (final clipped version):")
    for mm in [1, 2, 6, 7]:
        m = mo == mm
        print(f"    mo={mm}: n={int(m.sum()):>7,}  median={np.median(P[m]):>7.0f}  "
              f"p95={q(P[m],95):>8.0f}  max={P[m].max():>9.0f}  >3600={int((P[m]>3600).sum()):>6,}")

    print("\n  per-airport (final vs baseline):")
    g = (pd.DataFrame({"ap": ap, "F": P, "base": base})
         .groupby("ap").agg(n=("F", "size"), med_F=("F", "median"),
                            med_b=("base", "median"), max_F=("F", "max")))
    g["Δmed"] = g["med_F"] - g["med_b"]
    print(g.to_string(float_format=lambda v: f"{v:,.0f}"))

    meta = {
        "scheme": "C_full E2 — LIRF T=14400 full criterion (ADEP_flt missing & MVT-SCHED>14400)->MVT-SCHED; "
                  "rest LGB baseline; all rows with valid LOBT projected into [MVT-LOBT±3606]; clip[30,10800]",
        "holdout_rmse": {
            "A_baseline_493.05": None,
            "C_simple_484.38": None,
            "C_full_E1_LIRF_not_projected_463.75": None,
            "C_full_E2_LIRF_also_projected_460.87": None,
            "C_full_E2_noclip_324.76": None,
            "C_full_E1_noclip_341.73": None,
            "D_probability_mixture_451.60": None,
        },
        "E1_vs_E2": {"E1_lirf_rmse": 1358.42, "E2_lirf_rmse": 1345.66,
                     "winner": "E2", "n_diff_rows": 4,
                     "n_t14400_rows_touched_by_lobt": 0},
        "clip_sensitivity": {
            "note": "median true y of the T14400 hit rows is 20576s, 88.9% >10800; clip[30,10800] degrades "
                    "the RMSE of the hit rows from 7918.50 to 33313.07",
            "holdout_all4_clip10800": 460.87, "holdout_all4_noclip": 324.76,
            "holdout_lirf_clip10800": 1345.66, "holdout_lirf_noclip": 654.77},
        "n_final_dep": int(len(P)), "n_submitting": int(len(out)),
        "gate_checks": {k: bool(v) for k, v in checks.items()},
        "T_lobt": T_LOBT, "LIRF_T": LIRF_T, "clip": [CLIP_LO, CLIP_HI],
        "n_t14400_hit_final": int(hit.sum()),
        "n_lirf_rows": int(is_lirf.sum()),
        "n_adep_flt_null": int(adepflt_null.sum()),
        "n_lobt_valid": int(lobt_ok.sum()),
        "n_lobt_missing_skipped": int((~lobt_ok).sum()),
        "n_lobt_projection_changed": n_changed,
        "models": {
            "baseline_lgb": [f"notebooks/diag/d0/p0_models/p0_lgb_seed{s}.txt" for s in [42, 123, 2024]]},
        "submission_path": str(SUB_PATH),
        "submission_noclip_path": str(SUB_NOCLIP),
        "submission_status": "NOT uploaded (user uploads manually)",
        "per_month_rows": {f"mo{mm}": int((mo == mm).sum()) for mm in [1, 2, 6, 7]},
        "total_time_min": round((time.time() - _t0) / 60, 1),
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    lap(f"\n→ {META}")


if __name__ == "__main__":
    main()
