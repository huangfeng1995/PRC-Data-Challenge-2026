"""
Recompute the full RMSE of the noclip E2 criterion over the four holdout months (read-only, verifies ≈324.76)
==================================================================
Criterion (exactly the same as submissions/jubilant-puzzle_final.parquet):
  - LIRF with (ADEP_flt missing AND MVT-SCHED>14400) -> replaced by MVT-SCHED (no clip)
  - All other rows -> LGB baseline
  - All rows with valid LOBT projected into [MVT-LOBT±3606]
  - E2: LIRF is also projected
  - clip [30, 10800]  (E2 does not relax further; repaired rows were already clipped before projection, see below)

Note: the noclip version in final is build(1e9), i.e. no upper bound is set on values other than the repaired rows;
      this script follows exactly the same criterion.
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "holdout_noclip_e2.json"

HOLD = [1, 2, 6, 7]
LIRF_AP, LIRF_T = "LIRF", 14400.0
T_LOBT = 3606.0
CLIP_LO = 30.0


def main():
    ts = pickle.load(open(D0 / "_ts_cache.pkl", "rb"))
    y, mo = ts["y"], ts["mo"]
    ap = ts["ap"].astype(str)
    mvt, sch = ts["mvt_s"], ts["sch_s"]
    lobt, lok = ts["lobt_s"], ts["lobt_valid"]
    afn = ts["adep_flt_null"]

    i = np.where(np.isin(mo, HOLD))[0]
    yy, mm, aa = y[i], mo[i], ap[i]
    mvt_h, sch_h, lobt_h, lok_h, afn_h = mvt[i], sch[i], lobt[i], lok[i], afn[i]

    l1 = np.load(D0 / "_ngb_lgb_h1.npy").astype(np.float64)
    l2 = np.load(D0 / "_ngb_lgb_h2.npy").astype(np.float64)
    m1 = np.isin(mm, [1, 7])
    base = np.empty(len(i))
    base[m1] = l1
    base[~m1] = l2

    mu_C = mvt_h - sch_h
    is_lirf = aa == LIRF_AP
    hit = is_lirf & afn_h & (mu_C > LIRF_T)

    def build(clip_hi):
        P = base.copy()
        P[hit] = mu_C[hit]          # hard replacement, no clip
        P = np.clip(P, CLIP_LO, clip_hi)
        # E2: project all rows with valid LOBT (including LIRF)
        P[lok_h] = np.clip(P[lok_h], (mvt_h - lobt_h)[lok_h] - T_LOBT,
                           (mvt_h - lobt_h)[lok_h] + T_LOBT)
        return np.clip(P, CLIP_LO, clip_hi)

    def rm(P, m=None):
        if m is None:
            return float(np.sqrt(np.mean((P - yy) ** 2)))
        return float(np.sqrt(np.mean((P[m] - yy[m]) ** 2)))

    variants = {
        "noclip_E2 (official submission criterion)": build(1e9),
        "clip10800_E2 (backup criterion)": build(10800.0),
        "LGB baseline": base,
    }
    res, print_tbl = {}, {}
    print("=" * 74)
    print("[holdout four months] recompute noclip E2 criterion (n=%d)" % len(yy))
    print("=" * 74)
    print(f"{'criterion':<26}" + "".join(f"{'mo='+str(m):>10}" for m in HOLD)
          + f"{'4 months':>11}{'vs base':>10}")
    print("-" * 74)
    b4 = None
    for k, P in variants.items():
        r4 = rm(P)
        if b4 is None:
            b4 = r4
        cells = "".join(f"{rm(P, mm==m):>10.2f}" for m in HOLD)
        print(f"{k:<26}{cells}{r4:>11.2f}{r4-b4:>+10.2f}")
        res[k] = {f"mo{m}": round(rm(P, mm == m), 2) for m in HOLD}
        res[k]["all4"] = round(r4, 2)
        res[k]["delta_vs_LGB"] = round(r4 - b4, 2)
        print_tbl[k] = r4

    P = variants["noclip_E2 (official submission criterion)"]
    print(f"\n  [stratified] LIRF n={int(is_lirf.sum()):,}")
    for k in variants:
        print(f"    {k:<26} LIRF RMSE={rm(variants[k], is_lirf):>9.2f}")
    print(f"  repaired rows n={int(hit.sum())}: "
          f"pred==MVT-SCHED all equal={bool(np.all(np.abs(P[hit]-mu_C[hit])<0.5))}  "
          f"max={P[hit].max():.0f}")
    print(f"  non-repaired rows >10800 = {int(((~hit)&(P>10800)).sum())}")

    res["n"] = int(len(yy))
    res["n_fix"] = int(hit.sum())
    res["fix_pred_max"] = float(P[hit].max())
    res["n_nonfix_gt_10800"] = int(((~hit) & (P > 10800)).sum())
    res["lirf_rmse"] = {k: round(rm(v, is_lirf), 2) for k, v in variants.items()}
    res["anchors"] = {"E2_clip10800": 460.87, "E2_noclip_expected": 324.76,
                      "LGB_493.05": 493.05, "v22_full_342.40": 342.40,
                      "v22_simple_435.08": 435.08, "D_451.60": 451.60}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
