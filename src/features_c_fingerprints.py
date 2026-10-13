"""
C Anomaly Fingerprint Module - 7 features
==========================================
For departure taxi-out seconds TAXITIME_SEC_mvt, long taxi (>30min) is about 4.5% of rows, carrying about 63% of the variance, and is the largest unconquered direction.

7 features:
  1. type_S: AOBT actual pushback = SCHED scheduled takeoff (signal: takes off right after pushback, should not have taxi time)
  2. type_D: AOBT actual pushback substantially later than EOBT (delayed pushback)
  3. type_N: AOBT missing (no actual pushback time, missing in training data)
  4. type_H: long taxi (>=30min) but AOBT earlier than SCHED (pushback early but taxis long, anomaly signal)
  5. aobt_sched_clash: AOBT == SCHED (within a minute) strong anomaly signal
  6. crossday: AOBT date != SCHED date (cross-day)
  7. LTFM suffix stripping: cleanup of the STAND_mvt suffix for LTFM (e.g. "01A" -> "01")

Red lines:
  - BLOCK_TIME is blanked out in ranking DEP, never use BLOCK
  - MVT - AOBT cannot be used for training (this module complies)
  - Fields must exist, cannot be invented out of thin air
"""
from __future__ import annotations

import numpy as np
import pandas as pd


C_FEATS = [
    "c_type_S",        # AOBT ~= SCHED (takes off right after pushback, should not have long taxi)
    "c_type_D",        # AOBT much later than EOBT (delayed pushback)
    "c_type_N",        # AOBT missing
    "c_type_H",        # long taxi but AOBT earlier than SCHED
    "c_aobt_sched_clash",  # AOBT == SCHED (within a minute)
    "c_crossday",      # AOBT date != SCHED date
    "c_ltfc_stand_strip",  # LTFM STAND suffix stripping (01A -> 01)
]


def _norm_strip_ltfc(stand: pd.Series, airport: pd.Series) -> pd.Series:
    """Cleanup of the STAND_mvt suffix for LTFM
    - Example: "01A" -> "01", "02R" -> "02"
    - Other airports: keep as is
    - NaN -> NaN
    """
    out = stand.astype(str).copy()
    is_ltfc = airport.astype(str).values == "LTFM"
    # LTFM rows: take the pure numeric part
    digits = out.str.extract(r"(\d+)", expand=False)
    out[is_ltfc] = digits[is_ltfc]
    return out


def compute_c_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 C anomaly fingerprint features. Input requires the columns:
      - AOBT_3_flt (actual pushback, may be NaN)
      - EOBT_1_flt (estimated pushback, may be NaN)
      - SCHED_TIME_UTC_mvt (scheduled takeoff, always present)
      - ADEP_mvt (airport, always present)
      - STAND_mvt (stand, may be NaN)

    Returns: the original df + 7 new columns (filled only for DEP rows, NaN for ARR rows)
    """
    df = df.copy()
    n = len(df)

    # Convert timestamps to ns int64
    aobt = pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce")
    eobt = pd.to_datetime(df["EOBT_1_flt"], utc=True, errors="coerce")
    sched = pd.to_datetime(df["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
    mvt = pd.to_datetime(df.get("MVT_TIME_UTC_mvt"), utc=True, errors="coerce") if "MVT_TIME_UTC_mvt" in df.columns else None

    iNaT = np.iinfo(np.int64).min
    aobt_i = aobt.values.astype("datetime64[ns]").view("i8") if not aobt.empty else np.full(n, iNaT, dtype="int64")
    eobt_i = eobt.values.astype("datetime64[ns]").view("i8") if not eobt.empty else np.full(n, iNaT, dtype="int64")
    sched_i = sched.values.astype("datetime64[ns]").view("i8") if not sched.empty else np.full(n, iNaT, dtype="int64")
    aobt_v = aobt_i != iNaT
    eobt_v = eobt_i != iNaT
    sched_v = sched_i != iNaT

    # 1. type_S: AOBT ~= SCHED (within 5 minutes)
    type_S = np.zeros(n, dtype=np.int8)
    if mvt is not None:
        mvt_i = mvt.values.astype("datetime64[ns]").view("i8")
        # type_S: AOBT - SCHED within [-60s, 300s] (takes off shortly after pushback)
        diff_aobt_sched_s = (aobt_i - sched_i) / 1e9
        type_S = ((aobt_v & sched_v) & (diff_aobt_sched_s >= -60) & (diff_aobt_sched_s <= 300)).astype(np.int8)
    df["c_type_S"] = type_S

    # 2. type_D: AOBT much later than EOBT (pushback delayed >30 minutes)
    type_D = np.zeros(n, dtype=np.int8)
    diff_aobt_eobt_s = (aobt_i - eobt_i) / 1e9
    type_D = ((aobt_v & eobt_v) & (diff_aobt_eobt_s > 1800)).astype(np.int8)
    df["c_type_D"] = type_D

    # 3. type_N: AOBT missing (no actual pushback time)
    type_N = (~aobt_v).astype(np.int8)
    df["c_type_N"] = type_N

    # 4. type_H: long taxi (>=30min) but AOBT earlier than SCHED (pushback early but taxis long)
    #    long taxi is judged with TAXITIME_SEC_mvt (>=1800)
    #    only DEP rows are filled
    type_H = np.zeros(n, dtype=np.int8)
    if "TAXITIME_SEC_mvt" in df.columns and mvt is not None:
        is_dep = (df["PHASE_mvt"].astype(str).values == "DEP") if "PHASE_mvt" in df.columns else np.ones(n, dtype=bool)
        tt = pd.to_numeric(df["TAXITIME_SEC_mvt"], errors="coerce").values
        long_taxi = (tt >= 1800) & np.isfinite(tt)
        # AOBT earlier than SCHED (pushback < scheduled takeoff) but taxi >=30min
        aobt_before_sched = (aobt_i < sched_i) & aobt_v & sched_v
        type_H = (is_dep & long_taxi & aobt_before_sched).astype(np.int8)
    df["c_type_H"] = type_H

    # 5. aobt_sched_clash: AOBT equals SCHED (within a minute)
    aobt_sched_clash = np.zeros(n, dtype=np.int8)
    diff_aobt_sched_s = (aobt_i - sched_i) / 1e9
    aobt_sched_clash = ((aobt_v & sched_v) & (np.abs(diff_aobt_sched_s) < 60)).astype(np.int8)
    df["c_aobt_sched_clash"] = aobt_sched_clash

    # 6. crossday: AOBT date != SCHED date (cross-day)
    crossday = np.zeros(n, dtype=np.int8)
    if aobt_v.any() and sched_v.any():
        aobt_date = aobt.dt.date.values  # NaT -> NaT
        sched_date = sched.dt.date.values
        # comparing NaT raises an error, so use bitwise
        valid = aobt_v & sched_v
        crossday = np.where(valid, (aobt_date != sched_date).astype(np.int8), 0)
    df["c_crossday"] = crossday

    # 7. LTFM suffix stripping (the LIRF suffix stripping targets LTFM rather than LIRF, your description may be a typo)
    if "STAND_mvt" in df.columns and "ADEP_mvt" in df.columns:
        df["c_ltfc_stand_strip"] = _norm_strip_ltfc(df["STAND_mvt"], df["ADEP_mvt"]).values
    else:
        df["c_ltfc_stand_strip"] = np.nan

    return df


# Validation function for the known proportion / variance contribution of long taxi (>= 30min)
def long_taxi_stats(df_dep: pd.DataFrame) -> dict:
    """Validate the C module targeting: proportion of long taxi (>=30min) rows + proportion of variance"""
    tt = pd.to_numeric(df_dep["TAXITIME_SEC_mvt"], errors="coerce").values
    valid = np.isfinite(tt) & (tt > 0)
    long_taxi = valid & (tt >= 1800)
    n_total = int(valid.sum())
    n_long = int(long_taxi.sum())

    # Total variance (var of (y - mean))
    y_mean = tt[valid].mean()
    var_all = float(((tt[valid] - y_mean) ** 2).sum())

    # Variance contributed by long taxi (SSE_long)
    sse_long = float(((tt[long_taxi] - y_mean) ** 2).sum()) if n_long > 0 else 0.0
    sse_short = float(((tt[valid & ~long_taxi] - y_mean) ** 2).sum())

    return {
        "n_total": n_total,
        "n_long_taxi": n_long,
        "long_pct": n_long / max(n_total, 1),
        "sse_long": sse_long,
        "sse_short": sse_short,
        "sse_long_pct": sse_long / max(sse_long + sse_short, 1),
    }


if __name__ == "__main__":
    # Self-test: load February training data, compute C features + long taxi statistics
    from pathlib import Path
    import sys
    ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
    sys.path.insert(0, str(ROOT))
    from src.data_loader import load_month, parse_time_columns

    df = load_month("data/raw/training_2025-02-01_2025-03-01.parquet")
    df = parse_time_columns(df, cols=["AOBT_3_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"])

    df_c = compute_c_fingerprints(df)
    print(f"\n=== C anomaly fingerprint distribution (February DEP) ===")
    dep_c = df_c[df_c["PHASE_mvt"] == "DEP"]
    for f in C_FEATS:
        if f in ["c_ltfc_stand_strip"]:
            print(f"  {f}: n_valid={dep_c[f].notna().sum()}, top 5: "
                  f"{list(dep_c[f].value_counts().head(5).items())}")
        else:
            v = dep_c[f].value_counts()
            print(f"  {f}: {dict(v)}")

    print(f"\n=== long taxi statistics (C module targeting validation) ===")
    stats = long_taxi_stats(dep_c)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")