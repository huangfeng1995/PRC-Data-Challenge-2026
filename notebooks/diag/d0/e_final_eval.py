"""
Final criterion validation: C_full (LIRF T=14400 full spec) + E1/E2 LOBT projection comparison
================================================================
Criterion:
  - rows where LIRF and (ADEP_flt missing and MVT-SCHED>14400) -> replaced by MVT-SCHED
  - all other rows -> LGB baseline prediction
  - LOBT window: non-LIRF valid rows are projected to [MVT-LOBT±3606]
  - E1: LIRF not projected;  E2: LIRF also projected
  - clip[30,10800]

Reuses already-persisted models (p0 baseline predictions + clf/reg), no retraining.
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
OUT = D0 / "e_final_eval.json"

HOLD = [1, 2, 6, 7]
LIRF_AP = "LIRF"
T_LOBT = 3606.0
LIRF_T = 14400.0
CLIP_LO, CLIP_HI = 30.0, 10800.0

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:6.1f}s] {m}", flush=True)


def rmse(P, yt, m=None):
    if m is None:
        m = np.ones(len(yt), bool)
    return float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))


def main():
    lap("Loading cache")
    X, y, mo, mvt_s, ap = load_cache()
    ts = pickle.load(open(TS, "rb"))
    mvt_h_all, sch_all = ts["mvt_s"], ts["sch_s"]
    lobt_all, lobt_ok_all = ts["lobt_s"], ts["lobt_valid"]
    adepflt_null_all = ts["adep_flt_null"]
    assert np.array_equal(mvt_s, mvt_h_all)

    i_hold = np.where(np.isin(mo, HOLD))[0]
    yh, moh, ap_h = y[i_hold], mo[i_hold], ap[i_hold]
    mvt, sch = mvt_h_all[i_hold], sch_all[i_hold]
    lobt, lobt_ok = lobt_all[i_hold], lobt_ok_all[i_hold]
    adepflt_null = adepflt_null_all[i_hold]
    lap(f"  holdout 4 months n={len(i_hold):,}")

    # LGB baseline (already persisted)
    lgb_h1 = np.load(D0 / "_ngb_lgb_h1.npy")
    lgb_h2 = np.load(D0 / "_ngb_lgb_h2.npy")
    m_h1 = np.isin(moh, [1, 7])
    base = np.empty(len(i_hold))
    base[m_h1] = lgb_h1
    base[~m_h1] = lgb_h2

    is_lirf = (ap_h == LIRF_AP)
    mu_C = mvt - sch
    ctr = mvt - lobt
    lo_w, hi_w = ctr - T_LOBT, ctr + T_LOBT

    # the two T=14400 criteria
    hit_simple = is_lirf & (mu_C > LIRF_T)
    hit_full = is_lirf & adepflt_null & (mu_C > LIRF_T)
    lap(f"  LIRF n={int(is_lirf.sum()):,}  simplified spec hits {int(hit_simple.sum()):,}  "
        f"full spec hits {int(hit_full.sum()):,}")

    def build(use_full: bool, lirf_project: bool):
        hit = hit_full if use_full else hit_simple
        P = base.copy()
        P[hit] = mu_C[hit]
        P = np.clip(P, CLIP_LO, CLIP_HI)
        # LOBT projection: non-LIRF valid rows are always projected; LIRF depends on the switch
        m_proj = lobt_ok & (~is_lirf if not lirf_project else lobt_ok)
        before = P[m_proj].copy()
        P[m_proj] = np.clip(before, lo_w[m_proj], hi_w[m_proj])
        n_changed = int((np.abs(P[m_proj] - before) > 1e-9).sum())
        n_reclip = int(((P < CLIP_LO) | (P > CLIP_HI)).sum())
        P = np.clip(P, CLIP_LO, CLIP_HI)
        return P, hit, m_proj, n_changed, n_reclip

    # ---- scheme set ----
    A = base.copy()
    P_cs = build(False, False)[0]
    P_e1, hit_f, m_proj_e1, nch_e1, nrc_e1 = build(True, False)
    P_e2, _, m_proj_e2, nch_e2, nrc_e2 = build(True, True)
    combos = {
        "C_simple": P_cs,
        "C_full_E1_no_proj": P_e1,
        "C_full_E2_proj": P_e2,
    }

    # ---- per-month RMSE over the 4 months ----
    print("\n" + "=" * 84)
    print("[Per-month full RMSE]")
    print("=" * 84)
    print(f"{'scheme':<22}" + "".join(f"{'mo='+str(mm):>10}" for mm in HOLD)
          + f"{'4months':>11}{'D vs A':>10}")
    print("-" * 84)
    base4 = np.sqrt(np.mean((A - yh) ** 2))
    res = {}
    for k, P in combos.items():
        r4 = np.sqrt(np.mean((P - yh) ** 2))
        print(f"{k:<22}" + "".join(f"{rmse(P, yh, moh==mm):>10.2f}" for mm in HOLD)
              + f"{r4:>11.2f}{r4-base4:>+10.2f}")
        res[k] = {f"mo{mm}": round(rmse(P, yh, moh == mm), 2) for mm in HOLD}
        res[k]["all4"] = round(float(r4), 2)
        res[k]["delta_vs_A"] = round(float(r4 - base4), 2)

    # ---- LIRF-only RMSE ----
    print("\n" + "=" * 84)
    print("[LIRF-only RMSE]")
    print("=" * 84)
    m = is_lirf
    rA = rmse(A, yh, m)
    print(f"  LIRF n={int(m.sum()):,}")
    for k, P in combos.items():
        print(f"    {k:<22} {rmse(P, yh, m):>9.2f}   (vs A {rmse(P,yh,m)-rA:+.2f})")
    print(f"    {'A baseline':<22} {rA:>9.2f}")

    # ---- E1 vs E2 difference ----
    print("\n" + "=" * 84)
    print("[E1 vs E2] whether LIRF also gets LOBT projection")
    print("=" * 84)
    d_e1_lirf = rmse(P_e1, yh, m) - rmse(P_e2, yh, m)
    d_e1_all = np.sqrt(np.mean((P_e1 - yh) ** 2)) - np.sqrt(np.mean((P_e2 - yh) ** 2))
    diff_rows = np.where(np.abs(P_e1 - P_e2) > 1e-9)[0]
    n_t14400_touched = int((np.isin(diff_rows, np.where(hit_full)[0])).sum())
    print(f"  LIRF RMSE:      E1={rmse(P_e1, yh, m):.2f}  E2={rmse(P_e2, yh, m):.2f}  "
          f"D(E1-E2)={d_e1_lirf:+.2f}s")
    print(f"  4-month RMSE:   E1={np.sqrt(np.mean((P_e1-yh)**2)):.2f}  "
          f"E2={np.sqrt(np.mean((P_e2-yh)**2)):.2f}  D={d_e1_all:+.2f}s")
    print(f"  E1/E2 actually different rows = {len(diff_rows):,}")
    print(f"  of which T14400 repaired rows modified again by LOBT = {n_t14400_touched:,}")
    lap(f"    (E1 non-LIRF projection changed {nch_e1:,} / E2 additionally changed LIRF {nch_e2-nch_e1:,})")
    winner = "E1" if d_e1_lirf <= 0 else "E2"
    print(f"  -> lower LIRF RMSE: {winner}  (D={d_e1_lirf:+.2f}s)")

    # ---- per-airport delta ----
    print("\n" + "=" * 84)
    print("[Per-airport full RMSE -- gate: no degradation anywhere]")
    print("=" * 84)
    aps = pd.Series(ap_h).value_counts().index.tolist()
    print(f"{'ap':<7}{'n':>9}{'A':>9}{'C_simple':>10}{'E1':>9}{'E2':>9}"
          f"{'E1-A':>9}{'E2-A':>9}{'verdict':>7}")
    print("-" * 84)
    ap_tab = {}
    gate_ok = True
    for a in aps:
        mk = ap_h == a
        rA_, rs = rmse(A, yh, mk), rmse(combos["C_simple"], yh, mk)
        r1, r2 = rmse(P_e1, yh, mk), rmse(P_e2, yh, mk)
        ok = r1 <= rA_ + 1e-6
        if not ok:
            gate_ok = False
        ap_tab[a] = {"n": int(mk.sum()), "A": round(rA_, 2), "C_simple": round(rs, 2),
                     "E1": round(r1, 2), "E2": round(r2, 2),
                     "E1_minus_A": round(r1 - rA_, 2), "E2_minus_A": round(r2 - rA_, 2),
                     "not_worse_than_A": bool(ok)}
        print(f"{a:<7}{int(mk.sum()):>9,}{rA_:>9.1f}{rs:>10.1f}{r1:>9.1f}{r2:>9.1f}"
              f"{r1-rA_:>+9.2f}{r2-rA_:>+9.2f}{'  OK' if ok else '  FAIL':>7}")
    n_ok = sum(v["not_worse_than_A"] for v in ap_tab.values())
    print(f"\n  E1 per-airport not worse than A: {n_ok}/{len(aps)}  -> {'PASS' if gate_ok else 'FAIL'}")

    # ---- repaired-row quality ----
    print("\n" + "=" * 84)
    print("[T=14400 repaired-row evidence]")
    print("=" * 84)
    print(f"{'spec':<22}{'hits':>7}{'true_y_med':>12}{'MVT-SCHED_med':>15}"
          f"{'RMSE_after_fix':>12}{'RMSE_baseline':>11}")
    for nm, hit in [("simplified (>14400)", hit_simple), ("full (null & >14400)", hit_full)]:
        if hit.sum() == 0:
            continue
        print(f"{nm:<22}{int(hit.sum()):>7,}{np.median(yh[hit]):>12.1f}"
              f"{np.median(mu_C[hit]):>15.1f}{rmse(mu_C, yh, hit):>12.2f}"
              f"{rmse(A, yh, hit):>11.2f}")

    result = {
        "hit_simple": int(hit_simple.sum()), "hit_full": int(hit_full.sum()),
        "n_lirf": int(is_lirf.sum()), "combos": res,
        "lirf_rmse": {"A": round(rA, 2),
                      **{k: round(rmse(P, yh, m), 2) for k, P in combos.items()}},
        "E1_vs_E2": {"lirf_delta_E1_minus_E2": round(float(d_e1_lirf), 2),
                     "all4_delta": round(float(d_e1_all), 2),
                     "n_diff_rows": int(len(diff_rows)),
                     "n_t14400_rows_touched_by_lobt": n_t14400_touched,
                     "winner": winner},
        "per_airport": ap_tab,
        "gate_E1_not_worse_per_airport": {"pass": bool(gate_ok), "n_ok": int(n_ok),
                                          "n_total": len(aps)},
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  selected: {winner}  ->  {OUT}")
    lap(f"total time {(time.time()-_t0)/60:.1f} min")


if __name__ == "__main__":
    main()