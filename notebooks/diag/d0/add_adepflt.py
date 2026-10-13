"""
Append an ADEP_flt validity flag to _ts_cache.pkl (for 2025 four-month holdout validation)
=======================================================================
The full T=14400 spec on final = (ADEP==LIRF) AND (ADEP_flt missing) AND (MVT-SCHED>14400).
Stage 1's C used the simplified spec (no ADEP_flt condition); this script adds that condition to align with final.

Explicitly aligns to the existing _ts_cache row order by MVT_ID and asserts a 1-to-1 match over 2,085,047 rows.
"""
from __future__ import annotations
import pickle, glob
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D0 = Path("/Users/openclaw/Documents/PRC_data_challenge/notebooks/diag/d0")
ROOT = D0.parents[2]
TS = D0 / "_ts_cache.pkl"
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_flt"]


def main():
    ts = pickle.load(open(TS, "rb"))
    mvt_id_c = ts["mvt_id"]
    n = len(mvt_id_c)
    print(f"cache rows = {n:,}")

    parts = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))):
        d = pq.read_table(f, columns=COLS).to_pandas()
        parts.append(d[d["PHASE_mvt"].eq("DEP")])
    raw = pd.concat(parts, ignore_index=True)
    del parts
    print(f"raw DEP = {len(raw):,}")
    assert len(raw) == n, f"row count mismatch {len(raw)} vs {n}"

    mvt_id_r = raw["MVT_ID_mvt"].to_numpy()
    assert np.array_equal(mvt_id_r, mvt_id_c), "MVT_ID order mismatch (row order has drifted)"
    print("  ✓ MVT_ID matches exactly row by row")

    s = raw["ADEP_flt"]
    # null test: None / NaN / empty string / whitespace-only all count as missing
    adepflt_null = (s.isna() | (s.astype(str).str.strip() == "")).to_numpy()
    print(f"  ADEP_flt missing: {int(adepflt_null.sum()):,} / {n:,} "
          f"({adepflt_null.mean()*100:.3f}%)")

    ts["adep_flt_null"] = adepflt_null
    pickle.dump(ts, open(TS, "wb"), protocol=4)
    print(f"→ appended adep_flt_null to {TS.name}")

    # LIRF subset stats (aligned with final)
    ap = ts["ap"]
    mo = ts["mo"]
    is_lirf = ap == "LIRF"
    mu_C = ts["mvt_s"] - ts["sch_s"]
    for label, months in [("4mo(1/2/6/7)", [1, 2, 6, 7])]:
        m = np.isin(mo, months)
        lirf = is_lirf & m
        hit_simple = lirf & (mu_C > 14400)
        hit_full = lirf & adepflt_null & (mu_C > 14400)
        print(f"\n  [{label}] LIRF n={int(lirf.sum()):,}")
        print(f"    simplified (MVT-SCHED>14400)   hits {int(hit_simple.sum()):,}")
        print(f"    full (ADEP_flt null & >14400)  hits {int(hit_full.sum()):,}")
        print(f"    ADEP_flt missing but <=14400   {int((lirf & adepflt_null & (mu_C<=14400)).sum()):,}")


if __name__ == "__main__":
    main()