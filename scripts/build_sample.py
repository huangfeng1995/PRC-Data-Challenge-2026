"""
Generate small sample data (data/sample/) for quick feature-pipeline validation after cloning in the cloud
==================================================================
- Sample a small number of rows per airport from data/raw/training_2025_*.parquet x 12 + ranking.parquet
- Keep a few DEP rows per airport per month + the matching ARR rows (same FLIGHT_ID or same MVT time +/-5 minutes)
- Keep the total at a few MB ~ tens of MB so the repo grows by < 50 MB
- Never touch the sensitive directories listed in .gitignore
- Outputs:
    data/sample/
        training_2025-{MM}-{MM+1}.parquet  (12 files, each < 1 MB)
        ranking.parquet
        submitting.parquet
        sample_meta.json  (parameters of this sampling script + per-file row counts)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SAMPLE = ROOT / "data/sample"

# Airports (10, same as the competition)
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL",
            "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]

# Training: N_DEP rows per airport per month (DEP) + N_ARR rows for the same airport/month (ARR), with the linked FLIGHT_ID context
N_DEP_PER_AP_PER_MONTH = 40
N_ARR_PER_AP_PER_MONTH = 30  # sample slightly fewer ARR rows

# ranking + submitting: full copies (they are small anyway, ~45MB in total)
KEEP_RANKING_FULL = True


def sample_month(month: int) -> pd.DataFrame:
    """Sample a given month: take a few rows from each PHASE group per airport, preserving the time order as much as possible"""
    if month < 12:
        fp = RAW / f"training_2025-{month:02d}-01_2025-{month+1:02d}-01.parquet"
    else:
        fp = RAW / "training_2025-12-01_2026-01-01.parquet"
    if not fp.exists():
        print(f"⚠️ missing {fp.name}, skipping", flush=True)
        return pd.DataFrame()
    print(f"  [m={month:02d}] reading {fp.name} ...", flush=True)
    # All columns (column count is stable, not large)
    df = pd.read_parquet(fp)
    print(f"    original {len(df):,} rows", flush=True)

    out_parts = []
    for ap in AIRPORTS:
        sub = df[df["ADEP_mvt"] == ap] if ap in df["ADEP_mvt"].unique() else df.iloc[:0]
        if len(sub) == 0:
            continue
        dep = sub[sub["PHASE_mvt"] == "DEP"].sample(
            n=min(N_DEP_PER_AP_PER_MONTH, len(sub[sub["PHASE_mvt"] == "DEP"])),
            random_state=42)
        arr = sub[sub["PHASE_mvt"] == "ARR"].sample(
            n=min(N_ARR_PER_AP_PER_MONTH, len(sub[sub["PHASE_mvt"] == "ARR"])),
            random_state=42)
        out_parts.append(pd.concat([dep, arr], ignore_index=True))
    if not out_parts:
        return pd.DataFrame()
    out = pd.concat(out_parts, ignore_index=True)
    print(f"    after sampling {len(out):,} rows ({len(out_parts)} airports)", flush=True)
    return out


def main():
    if not RAW.exists():
        print(f"❌ missing data source {RAW}/, run scripts/download_data.py first", flush=True)
        return 1
    SAMPLE.mkdir(parents=True, exist_ok=True)

    meta = {"version": "v1", "n_dep_per_ap_per_month": N_DEP_PER_AP_PER_MONTH,
            "n_arr_per_ap_per_month": N_ARR_PER_AP_PER_MONTH, "airports": AIRPORTS,
            "files": {}}

    print("=" * 60)
    print("Generating small sample (data/sample/)")
    print("=" * 60)
    total_bytes = 0
    for m in range(1, 13):
        df = sample_month(m)
        if len(df) == 0:
            continue
        if m < 12:
            out_name = f"training_2025-{m:02d}-01_2025-{m+1:02d}-01.parquet"
        else:
            out_name = "training_2025-12-01_2026-01-01.parquet"
        out_path = SAMPLE / out_name
        df.to_parquet(out_path, index=False)
        sz = out_path.stat().st_size
        total_bytes += sz
        meta["files"][out_name] = {"rows": int(len(df)), "bytes": sz}
        print(f"  wrote {out_name}: {len(df):,} rows, {sz/1024:.1f} KB", flush=True)

    # ranking + submitting sampling (no need for the full set in the cloud, 5000 rows is enough to test the feature pipeline)
    N_RANK_SAMPLE = 5000
    for name, n_sample in [("ranking.parquet", N_RANK_SAMPLE), ("submitting.parquet", N_RANK_SAMPLE)]:
        src = RAW / name
        if not src.exists():
            print(f"⚠️ missing {name}, skipping", flush=True)
            continue
        df = pd.read_parquet(src)
        if len(df) > n_sample:
            df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        out_path = SAMPLE / name
        df.to_parquet(out_path, index=False)
        sz2 = out_path.stat().st_size
        total_bytes += sz2
        meta["files"][name] = {"rows": int(len(df)), "bytes": sz2, "sampled": n_sample < len(pd.read_parquet(src)) if src.exists() else False}
        print(f"  wrote {name}: {len(df):,} rows, {sz2/1024:.1f} KB", flush=True)

    meta["total_bytes"] = total_bytes
    (SAMPLE / "sample_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n✅ total {total_bytes/1024:.1f} KB ({total_bytes/1e6:.2f} MB)")
    print(f"   written to {SAMPLE}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())