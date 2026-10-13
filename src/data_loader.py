"""
Data loading module
===================

Reads the monthly parquet files from data/raw/, merges them into a full-year dataset, and applies basic filtering.

Design notes:
- Only read DEP rows (taxi-out is only meaningful for departing flights)
- Keep only the necessary columns to reduce memory usage
- Convert all time columns to pandas datetime + UTC
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Field definitions (from https://prc-data-challenge-2026.netlify.app/data.html)
# ---------------------------------------------------------------------------

# The columns we care about in the training data. The _mvt suffix is airport-side, _flt is NM flight-side.
TRAIN_COLS_MVT = [
    "MVT_ID_mvt",        # unique identifier
    "FLIGHT_ID_mvt",
    "ADEP_mvt",          # departure airport ICAO
    "ADES_mvt",          # arrival airport ICAO
    "PHASE_mvt",         # DEP / ARR, we need DEP
    "MVT_TIME_UTC_mvt",  # actual takeoff time (note: this is the label endpoint, must not be used as a feature)
    "BLOCK_TIME_UTC_mvt",  # actual pushback time (note: this is the taxi-out starting point, it leaks, use with care)
    "SCHED_TIME_UTC_mvt",  # scheduled take-off/landing time (usable, no leakage)
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",  # target variable (seconds)
]

TRAIN_COLS_FLT = [
    "LOBT_flt",
    "CALLSIGN_flt",
    "ADES_flt",
    "ADES_FILED_flt",    # different from ADES_flt = flight re-routing
    "MARKET_SEGMENT_flt",
    "IOBT_flt",          # initial scheduled pushback time (usable, no leakage)
    "EOBT_1_flt",        # FPL estimated pushback time (usable, no leakage)
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",    # wake turbulence category: L/M/H/J
    "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt",        # WARNING: actual pushback time, leaks the target starting point, only usable for post-hoc aggregation
    "ARVT_3_flt",        # WARNING: actual arrival time, also carries leakage information for DEP rows
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def list_monthly_files(data_dir: Path | str, prefix: str = "training_") -> list[Path]:
    """List all parquet files under data_dir matching prefix, sorted by file name."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"{prefix}*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No {prefix}*.parquet files found under {data_dir}."
            f"Please make sure the data has been downloaded and placed in this directory."
        )
    return files


def load_month(file_path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a single monthly parquet file.

    Passing columns explicitly through pyarrow greatly reduces memory usage (the original file is ~25 MB, a full read would load everything).
    """
    file_path = Path(file_path)
    table = pq.read_table(file_path, columns=columns)
    return table.to_pandas()


def load_all_months(
    data_dir: Path | str,
    prefix: str = "training_",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load all monthly files and concatenate them.

    Estimated memory usage: 4.17M rows in total, ~280 MB of parquet expands to ~1.5 GB as a pandas DataFrame.
    If the machine has less than 8 GB of RAM, consider training month by month or processing the data directly with pyarrow.
    """
    files = list_monthly_files(data_dir, prefix=prefix)
    print(f"Found {len(files)} parquet files, starting to load...")

    dfs = []
    for f in files:
        df = load_month(f, columns=columns)
        dfs.append(df)
        print(f"  loaded {f.name}: {len(df):,} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"total rows after merge: {len(df):,}")
    return df


def filter_departures(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only DEP rows, which is what taxi-out prediction targets."""
    n_before = len(df)
    df = df[df["PHASE_mvt"] == "DEP"].copy()
    print(f"filtered DEP: {n_before:,} -> {len(df):,} rows ({(len(df) / n_before * 100):.1f}%)")
    return df


def parse_time_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Convert UTC time string columns into pandas datetime (UTC)."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Main entry: load + filter + time column parsing
# ---------------------------------------------------------------------------

def load_training(data_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load training data (full year 2025), keep only DEP rows and parse time columns.

    The returned DataFrame has already been filtered to PHASE=DEP, and all time columns are datetime64[ns, UTC].
    """
    cols = TRAIN_COLS_MVT + TRAIN_COLS_FLT
    df = load_all_months(data_dir, prefix="training_", columns=cols)
    df = filter_departures(df)
    df = parse_time_columns(
        df,
        cols=[
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "IOBT_flt",
            "EOBT_1_flt",
            "AOBT_3_flt",
            "ARVT_3_flt",
            "LOBT_flt",
        ],
    )
    return df


def load_ranking(data_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load the ranking dataset (2026-01 + 2026-07, target variable masked).

    BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are null on DEP rows, which is exactly the part we have to predict.
    """
    cols = TRAIN_COLS_MVT + TRAIN_COLS_FLT
    df = load_month(Path(data_dir) / "ranking.parquet", columns=cols)
    df = filter_departures(df)
    df = parse_time_columns(
        df,
        cols=[
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "IOBT_flt",
            "EOBT_1_flt",
            "AOBT_3_flt",
            "ARVT_3_flt",
            "LOBT_flt",
        ],
    )
    return df


def load_submission_template(data_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load the submission template (only two columns: MVT_ID_mvt and TAXITIME_SEC_mvt)."""
    return pd.read_parquet(Path(data_dir) / "submitting.parquet")


if __name__ == "__main__":
    # Module self-test: python -m src.data_loader
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    print("=== Training data ===")
    df = load_training(data_dir)
    print(f"\nshape: {df.shape}")
    print(f"\ncolumn dtypes:\n{df.dtypes}")
    print(f"\ntarget variable description:\n{df['TAXITIME_SEC_mvt'].describe()}")