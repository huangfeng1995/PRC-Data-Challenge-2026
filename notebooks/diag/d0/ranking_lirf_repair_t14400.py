"""
ranking_lirf_repair_t14400.py — T=14400 version of ranking post-processing (read-only inference, labels unchanged)

Base: submissions/jubilant-puzzle_v15_lirf_repair.parquet (v21, leaderboard 363.51)
Rule: ADEP=='LIRF' AND ADEP_flt missing AND MVT-SCHED > 14400 -> prediction becomes MVT-SCHED
      all other rows stay identical to v21 row by row
Output: exactly the same column structure as v21
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
BASE = ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair.parquet"   # v21
OUT = ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair_t14400.parquet"
RPT = D0 / "ranking_lirf_repair_t14400.json"
NS = 1_000_000_000
T = 14400.0
AP = "LIRF"

_t0 = time.time()
def log(*a): print(f"[{time.time()-_t0:6.1f}s]", " ".join(str(x) for x in a), flush=True)

# ---------- 1. Baseline (v21) ----------
base = pd.read_parquet(BASE)
log(f"baseline {BASE.name}: {len(base):,} rows, cols={list(base.columns)}")
assert list(base.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
assert len(base) == 344_841
v21 = base["TAXITIME_SEC_mvt"].to_numpy(np.float64).copy()
log(f"  v21 preds: min={v21.min():.2f} max={v21.max():.2f} mean={v21.mean():.2f}")

# ---------- 2. ranking visible columns ----------
rk = pd.read_parquet(RAW / "ranking.parquet",
                     columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt",
                              "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"])
for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]:
    v = pd.to_datetime(rk[c], utc=True, errors="coerce")
    # R5: explicitly normalize to ns before taking int64 seconds
    rk[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    rk[c + "_s"] = rk[c].astype("int64") // NS
log(f"ranking read {len(rk):,} rows, timestamps unified to datetime64[ns]")

# ---------- 3. Rule ----------
is_dep = (rk["PHASE_mvt"] == "DEP").to_numpy()
is_lirf = (rk["ADEP_mvt"].astype(str) == AP).to_numpy()
adep_flt = rk["ADEP_flt"]
adep_null = (adep_flt.isna() | adep_flt.astype(str).str.strip().isin(["", "None", "nan"])).to_numpy()
aobt_s = rk["AOBT_3_flt_s"].to_numpy(np.float64)
aobt_bad = (aobt_s == -9223372037) | np.isnan(aobt_s)
cand = (rk["MVT_TIME_UTC_mvt_s"] - rk["SCHED_TIME_UTC_mvt_s"]).to_numpy(np.float64)
cand = np.where(np.isfinite(cand), cand, -np.inf)

hit = is_dep & is_lirf & adep_null & (cand > T)
log("")
log(f"rule breakdown (T={int(T)}):")
log(f"  DEP and {AP}          : {int((is_dep & is_lirf).sum()):,}")
log(f"  ADEP_flt missing      : {int(adep_null.sum()):,}")
log(f"  ADEP_flt blank == AOBT sentinel: {bool((adep_null == aobt_bad).all())}")
log(f"  LIRF & ADEP_flt null  : {int((is_dep & is_lirf & adep_null).sum()):,}")
log(f"  >>> hit (add thr>{int(T)})  : {int(hit.sum()):,} rows")

# ---------- 4. Apply ----------
rk["_hit"] = hit
rk["_cand"] = cand
m = base.merge(rk[["MVT_ID_mvt", "_hit", "_cand"]], on="MVT_ID_mvt", how="left")
assert len(m) == len(base)
assert m["_hit"].notna().all()

n_hit = int(m["_hit"].sum())
newp = v21.copy()
hitmask = m["_hit"].to_numpy()
newp[hitmask] = m.loc[hitmask, "_cand"].to_numpy(np.float64)
changed = ~np.isclose(newp, v21, rtol=0, atol=1e-9)

print()
print("=" * 92)
print("Newly hit rows (v21 unchanged -> changed this run)")
print("=" * 92)
prev = pd.read_parquet(ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair.parquet")
hit_ids = set(m.loc[hitmask, "MVT_ID_mvt"].astype(int))
v21_map = dict(zip(base["MVT_ID_mvt"].astype(int), v21))
new_map = dict(zip(base["MVT_ID_mvt"].astype(int), newp))
added = [i for i in hit_ids if not np.isclose(v21_map[i], min(
    [v21_map[i]], default=0), atol=1e-9)]
# the real "new" = hit this run but v21 prediction != MVT-SCHED (i.e. rows v21 never changed)
newly = [i for i in sorted(hit_ids)
         if abs(new_map[i] - v21_map[i]) > 1e-9]
print(f"hit this run {n_hit} rows | actually new vs v21 {len(newly)} rows")
print(f"\n{'MVT_ID':>12} {'v21 value':>10} {'T14400 new':>12} {'delta':>11}")
for i in newly:
    print(f"{i:>12} {v21_map[i]:>10.2f} {new_map[i]:>12.1f} {new_map[i]-v21_map[i]:>+11.1f}")

print()
print("=" * 92)
print("Summary")
print("=" * 92)
print(f"  hit rows (LIRF+ADEP_flt null+>{int(T)}) : {n_hit}")
print(f"  newly changed vs v21 rows          : {len(newly)}")
print(f"  total changed vs v21 rows          : {int(changed.sum())}")
print(f"  v21   range     : [{v21.min():.2f}, {v21.max():.2f}]  mean={v21.mean():.2f}")
print(f"  T14400 range    : [{newp.min():.2f}, {newp.max():.2f}]  mean={newp.mean():.2f}")
print(f"  unchanged rows  : {int((~changed).sum()):,}")

assert np.array_equal(newp[~changed], v21[~changed]), "non-hit rows were modified unexpectedly"
assert len(newp) == 344_841
print("\n✓ verification passed: non-hit rows identical to v21 row by row; total rows 344,841")

# ---------- 5. Persist ----------
out = pd.DataFrame({"MVT_ID_mvt": base["MVT_ID_mvt"].to_numpy(),
                    "TAXITIME_SEC_mvt": newp})
assert list(out.columns) == list(base.columns)
out.to_parquet(OUT, index=False)
log(f"\n→ {OUT}  ({OUT.stat().st_size:,} bytes)")

chk = pd.read_parquet(OUT)
assert len(chk) == 344_841 and list(chk.columns) == list(base.columns)
assert chk["MVT_ID_mvt"].equals(base["MVT_ID_mvt"]), "MVT_ID order mismatch"
d = np.abs(chk["TAXITIME_SEC_mvt"].to_numpy() - v21)
print(f"✓ readback check: {len(chk):,} rows, {int((d>1e-9).sum())} rows differ from v21 (expected={len(newly)})")
print(f"  rows below 30: {int((chk['TAXITIME_SEC_mvt']<30).sum())}  NaN: {int(chk['TAXITIME_SEC_mvt'].isna().sum())}")

json.dump({"base_file": BASE.name, "out_file": OUT.name,
           "n_rows": int(len(out)), "n_cols": list(out.columns),
           "rule": {"adep": AP, "cond": "ADEP_flt missing", "T": T,
                    "new_pred": "MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt"},
           "n_dep_lirf": int((is_dep & is_lirf).sum()),
           "n_adep_flt_null": int(adep_null.sum()),
           "adep_null_equals_aobt_bad": bool((adep_null == aobt_bad).all()),
           "n_hit": n_hit, "n_newly_changed_vs_v21": len(newly),
           "n_changed_vs_v21": int(changed.sum()),
           "v21_stats": {"min": float(v21.min()), "max": float(v21.max()), "mean": float(v21.mean())},
           "new_stats": {"min": float(newp.min()), "max": float(newp.max()), "mean": float(newp.mean())},
           "newly_changed": [{"MVT_ID": int(i), "v21": float(v21_map[i]), "new": float(new_map[i])}
                             for i in newly]},
          open(RPT, "w"), ensure_ascii=False, indent=2)
print(f"→ {RPT}")