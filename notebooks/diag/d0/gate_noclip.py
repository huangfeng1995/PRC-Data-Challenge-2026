"""
Gate validation of the noclip official submission file (read-only)
==================================
Range validation falls into two categories (a uniform 30~10800 cannot be used):
  · non-repaired rows          : 30 ~ 10800
  · the 63 rows hit by the full T=14400 criterion: pred == MVT-SCHED (values >10800 allowed, max ~111,666)

full T=14400 criterion = (ADEP==LIRF) AND (ADEP_flt missing) AND (MVT-SCHED > 14400)
Repaired-row marking uses an explicit alignment of final_ranking by MVT_ID (not row order).

Read-only: does not modify files, does not retrain, does not upload.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
SUB = ROOT / "submissions" / "jubilant-puzzle_final.parquet"
BACKUP = ROOT / "submissions" / "jubilant-puzzle_final_clip_backup.parquet"
SUBMIT_TMPL = ROOT / "data/raw/final_submitting.parquet"
FINAL = ROOT / "data/raw/final_ranking.parquet"
OUT = D0 / "gate_noclip.json"

LIRF_AP = "LIRF"
LIRF_T = 14400.0
CLIP_LO, CLIP_HI = 30.0, 10800.0
N_FIX_EXPECT = 63


def sec(s: pd.Series) -> np.ndarray:
    if s.dtype == "datetime64[us, UTC]":
        s = s.dt.tz_localize(None)
    v = s.astype("datetime64[ns]").to_numpy("int64") // 10**9
    v = v.astype(np.float64)
    v[v < 0] = np.nan
    return v


def main():
    checks, info = {}, {}

    print("=" * 78)
    print("[Gate] validation of the noclip official submission file")
    print("=" * 78)
    print(f"  file: {SUB.name}  ({SUB.stat().st_size:,} bytes)")

    sub = pd.read_parquet(SUB)
    checks["columns correct"] = list(sub.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    checks["n_rows=670790"] = len(sub) == 670790
    checks["int64"] = sub["TAXITIME_SEC_mvt"].dtype == np.int64
    checks["no NaN"] = bool(sub["TAXITIME_SEC_mvt"].notna().all())
    checks["MVT_ID unique"] = bool(sub["MVT_ID_mvt"].is_unique)

    tmpl = pd.read_parquet(SUBMIT_TMPL)
    exp = tmpl["MVT_ID_mvt"].astype(float).values
    got = sub["MVT_ID_mvt"].astype(float).values
    checks["MVT_ID order matches"] = bool(np.all(exp == got))
    checks["MVT_ID set matches"] = set(exp) == set(got)
    checks["n_rows matches template"] = len(sub) == len(tmpl)

    for k, v in checks.items():
        print(f"    {'✓' if v else '✗ FAIL'}  {k}")
    assert all(checks.values()), f"basic Gate failed: {checks}"

    # ---- repaired-row marking (final_ranking aligned by MVT_ID) ----
    fn = pq.read_table(FINAL, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt",
                                      "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                      "LOBT_flt"]).to_pandas()
    f = fn[fn["PHASE_mvt"].eq("DEP")].copy()
    f["mvt"] = sec(f["MVT_TIME_UTC_mvt"])
    f["sch"] = sec(f["SCHED_TIME_UTC_mvt"])
    f["lobt"] = sec(f["LOBT_flt"])
    muC = f["mvt"] - f["sch"]
    afn = f["ADEP_flt"].isna() | (f["ADEP_flt"].astype(str).str.strip() == "")
    hit = (f["ADEP_mvt"].eq(LIRF_AP).to_numpy() & afn.to_numpy()
           & (muC.to_numpy() > LIRF_T))
    key = pd.Index(f["MVT_ID_mvt"].astype(float))
    hit_s = pd.Series(hit, index=key)
    fix_flag = hit_s.reindex(pd.Index(got), fill_value=False).to_numpy(bool)
    muC_s = pd.Series(muC.to_numpy(), index=key)
    fix_muC = muC_s.reindex(pd.Index(got)).to_numpy()   # aligned to the submission file row order
    mv_s = pd.Series(f["mvt"].to_numpy(), index=key)
    mvt_r = mv_s.reindex(pd.Index(got)).to_numpy()
    lo_s = pd.Series(f["lobt"].to_numpy(), index=key)
    lobt_r = lo_s.reindex(pd.Index(got)).to_numpy()
    lobt_ok_r = ~np.isnan(lobt_r)
    mo = pd.to_datetime(pd.Series(mvt_r), unit="s").dt.month.to_numpy()
    del f, fn

    n_fix = int(fix_flag.sum())
    pred = sub["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    is_fix = fix_flag
    non_fix = ~fix_flag

    print("\n  [category range validation]")
    checks[f"n_repaired_rows={N_FIX_EXPECT}"] = n_fix == N_FIX_EXPECT
    print(f"    {'✓' if checks[f'n_repaired_rows={N_FIX_EXPECT}'] else '✗ FAIL'}  "
          f"n_repaired_rows = {n_fix} (expected {N_FIX_EXPECT})")

    # repaired rows: pred must == MVT-SCHED (exactly equal, no clip)
    d = pred[is_fix] - fix_muC[is_fix]
    n_eq = int((np.abs(d) < 0.5).sum())
    fix_ok = (n_eq == n_fix)
    checks["repaired row pred==MVT-SCHED(no clip)"] = fix_ok
    print(f"    {'✓' if fix_ok else '✗ FAIL'}  the 63 repaired rows have pred == MVT-SCHED: "
          f"{n_eq}/{n_fix}  (max|d|={np.abs(d).max():.3f})")
    n_fix_gt = int((pred[is_fix] > CLIP_HI).sum())
    n_fix_at = int(np.isclose(pred[is_fix], CLIP_HI).sum())
    checks["no clip traces in repaired rows"] = n_fix_at == 0
    print(f"    {'✓' if checks['no clip traces in repaired rows'] else '✗ FAIL'}  "
          f"repaired rows: {n_fix_gt} rows >10800, {n_fix_at} rows ==10800 (should be 0)")
    print(f"    repaired row preds: min={pred[is_fix].min():.0f} "
          f"p50={np.median(pred[is_fix]):.0f} max={pred[is_fix].max():.0f}")

    # non-repaired rows: 30 ~ 10800; the exception is rows legitimately projected by the LOBT window to >10800
    nf = pred[non_fix]
    ok_lo = bool((nf >= CLIP_LO).all())
    over = non_fix & (pred > CLIP_HI)
    n_over = int(over.sum())
    # these rows must fall inside their LOBT window (proving window projection rather than a missing/out-of-range clip)
    ctr = mvt_r - lobt_r
    lo_w, hi_w = ctr - 3606.0, ctr + 3606.0
    in_win = (pred[over] >= lo_w[over] - 0.5) & (pred[over] <= hi_w[over] + 0.5)
    ok_over = bool(lobt_ok_r[over].all() and in_win.all())
    checks["non-repaired rows range[30,10800]"] = ok_lo and ok_over
    checks["non-repaired rows >10800 all come from the LOBT window"] = ok_over
    print(f"    {'✓' if ok_lo else '✗ FAIL'}  non-repaired rows ({len(nf):,}) "
          f"lower bound ≥30: {nf.min():.0f}")
    print(f"    {'✓' if ok_over else '✗ FAIL'}  non-repaired rows >10800 total {n_over} rows, "
          f"all inside their LOBT window [lo,hi] (window projection, not out of range)")
    if n_over:
        det = pd.DataFrame({"MVT_ID": got[over], "pred": pred[over],
                            "lo_w": lo_w[over], "hi_w": hi_w[over]})
        print("      detail:")
        for _, r_ in det.iterrows():
            print(f"        MVT_ID={int(r_.MVT_ID)}  pred={r_.pred:.0f}  "
                  f"window=[{r_.lo_w:.0f}, {r_.hi_w:.0f}]  in_window="
                  f"{r_.lo_w - 0.5 <= r_.pred <= r_.hi_w + 0.5}")
    print(f"    non-repaired rows max={nf.max():.0f}  "
          f"({int((nf<=CLIP_HI).sum()):,} rows in [30,10800], {n_over} rows are window projections)")

    for k, v in checks.items():
        if k not in ("columns correct", "n_rows=670790", "int64", "no NaN", "MVT_ID unique",
                     "MVT_ID order matches", "MVT_ID set matches", "n_rows matches template"):
            continue
    assert all(checks.values()), f"Gate failed: {[k for k,v in checks.items() if not v]}"

    # ---- report ----
    print("\n" + "=" * 78)
    print("[report]")
    print("=" * 78)
    print(f"  official file: {SUB.name}  ({SUB.stat().st_size:,} bytes)")
    print(f"  backup file: {BACKUP.name}  ({BACKUP.stat().st_size:,} bytes)  "
          f"exists={BACKUP.exists()}")

    print(f"\n  per-month (n_rows / pred_median / pred_max):")
    mon = {}
    for mm in [1, 2, 6, 7]:
        m = mo == mm
        v = pred[m]
        mon[f"mo{mm}"] = {"n": int(m.sum()), "median": round(float(np.median(v)), 1),
                          "max": round(float(v.max()), 1),
                          "n_gt_10800": int((v > CLIP_HI).sum())}
        print(f"    mo={mm}: n={int(m.sum()):>7,}  median={np.median(v):>8.0f}  "
              f"max={v.max():>9.0f}  >10800={int((v>CLIP_HI).sum()):>3}")

    q = lambda p: float(np.percentile(pred, p))
    print(f"\n  global: min={pred.min():.0f}  p25={q(25):.0f}  median={q(50):.0f}  "
          f"p75={q(75):.0f}  p95={q(95):.0f}  max={pred.max():.0f}")
    print(f"  >1800={int((pred>1800).sum()):,}  >3600={int((pred>3600).sum()):,}  "
          f">10800={int((pred>CLIP_HI).sum()):,}")

    # comparison with the clip backup
    b = pd.read_parquet(BACKUP)
    bmap = pd.Series(b["TAXITIME_SEC_mvt"].to_numpy(),
                     index=b["MVT_ID_mvt"].astype(float))
    bpred = bmap.reindex(pd.Index(got)).to_numpy()
    diff = np.abs(pred - bpred) > 0.5
    print(f"\n  [diff vs the clip backup] n_diff_rows={int(diff.sum()):,} "
          f"(should = 63 repaired rows)  n_nonfix_diff={int((diff & non_fix).sum())}")
    print(f"    clip backup max={bpred.max():.0f}  |  noclip max={pred.max():.0f}")

    res = {"file": str(SUB), "size_bytes": SUB.stat().st_size,
           "backup": str(BACKUP), "backup_size_bytes": BACKUP.stat().st_size,
           "all_checks_passed": bool(all(checks.values())),
           "checks": {k: bool(v) for k, v in checks.items()},
           "n_fixed_rows": n_fix, "n_fixed_gt_10800": n_fix_gt,
           "n_fixed_eq_10800": n_fix_at,
           "n_nonfix_gt_10800": n_over,
           "nonfix_over_10800_detail": [
               {"MVT_ID": int(got[i]), "pred": float(pred[i]),
                "lobt_window": [float(lo_w[i]), float(hi_w[i])],
                "in_window": bool(lo_w[i] - 0.5 <= pred[i] <= hi_w[i] + 0.5)}
               for i in np.where(over)[0]],
           "fixed_pred": {"min": float(pred[is_fix].min()),
                          "p50": float(np.median(pred[is_fix])),
                          "max": float(pred[is_fix].max())},
           "per_month": mon,
           "global": {"min": float(pred.min()), "p25": q(25), "median": q(50),
                      "p75": q(75), "p95": q(95), "max": float(pred.max()),
                      "n_gt_1800": int((pred > 1800).sum()),
                      "n_gt_3600": int((pred > 3600).sum()),
                      "n_gt_10800": int((pred > CLIP_HI).sum())},
           "vs_clip_backup": {"n_diff_rows": int(diff.sum()),
                              "n_nonfix_diff": int((diff & non_fix).sum()),
                              "clip_max": float(bpred.max())},
           "upload_status": "NOT uploaded (user uploads manually)"}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")
    print(f"\n✅ all Gates passed. Not uploaded.")


if __name__ == "__main__":
    main()
