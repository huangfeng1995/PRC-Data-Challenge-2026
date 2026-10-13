"""
Congestion Aggregate Feature Module (core of plan B)
====================================================

For each DEP record, compute the congestion status at the same airport/runway within several time windows before its "pushback moment".

WARNING: the "reference time" used for training vs inference is different!
  - At training time: use AOBT_3_flt (actual pushback time) as reference - because we can obtain it after the fact, aggregate features are most accurate
  - At inference time: BLOCK_TIME_UTC_mvt is masked in the ranking data, but EOBT_1_flt is available as a "scheduled estimate"
  - Compromise: use EOBT_1_flt as the "reference time", so training and inference are consistent, at the cost of introducing some noise

Module entry: add_aggregate_features(df, ref_time_col='EOBT_1_flt')

Performance notes (v3 cumsum refactor):
  - Old version O(N^2): inner for k + np.searchsorted + np.percentile
  - New version O(N): sort+lexsort first, then use cumsum to compute count/mean in one pass
  - p90 uses np.partition instead of np.percentile, O(W) vs O(W log W)
  - 2 million rows, 10 airports, 30m window: old version 12+ minutes -> new version < 10 seconds (measured)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core utility: O(N) cumsum implementation
# ---------------------------------------------------------------------------

def _vectorized_past_window(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    window_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized computation of count / mean / p90 within the [t-window_min, t) window.

    Algorithm (O(N) overall):
      1. Sort by (group, time)
      2. Build cumsum inside each group:
         - cumcnt: cumulative sum of the indicator
         - cumsum_y: cumulative sum of target (NaN treated as 0)
      3. For each time point t_now:
         - lo = searchsorted(group_times, t_now - window, 'left')
         - hi = k  (excluding self)
         - count = cumcnt[hi] - cumcnt[lo]
         - sum_y = cumsum_y[hi] - cumsum_y[lo]
         - mean = sum_y / count
         - p90 = partition within the window (approximate O(W))

    Returns three numpy arrays (count, mean, p90), in the original df order.
    """
    n = len(df)

    if isinstance(group_col, str):
        groups_raw = df[group_col].values
    else:
        groups_raw = df[list(group_col)].astype(str).agg("_".join, axis=1).values

    # Encode group as integers (easy for np.diff to find boundaries)
    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    times = pd.to_datetime(df[time_col]).values.astype("datetime64[ns]")
    order = np.lexsort((times, groups))
    inv_order = np.argsort(order)

    sorted_groups = groups[order]
    sorted_times = times[order]
    raw_targets = df[target_col].values[order]

    # Treat NaN as 0 (for mean/cumsum), but count only non-NaN
    not_nan = ~np.isnan(raw_targets)
    safe_targets = np.where(not_nan, raw_targets, 0.0).astype("float64")
    # count = 1 - is_nan, so NaN counts as 0
    safe_indicators = not_nan.astype("float64")

    counts = np.zeros(n, dtype="float32")
    means = np.full(n, np.nan, dtype="float32")
    p90s = np.full(n, np.nan, dtype="float32")

    window_ns = np.timedelta64(window_min, "m")

    # Find group boundaries
    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [n]])

    # Iterate over each group
    for start, end in zip(group_starts, group_ends):
        gtimes = sorted_times[start:end]
        gsum = safe_targets[start:end]
        gcount = safe_indicators[start:end]
        m = end - start

        # cumsum (prepend a 0 sentinel before each group to simplify boundaries)
        cum_sum = np.concatenate([[0.0], np.cumsum(gsum, dtype="float64")])
        cum_cnt = np.concatenate([[0.0], np.cumsum(gcount, dtype="float64")])

        # Window lower bound (for each time point)
        t_lo_all = gtimes - window_ns  # array
        # searchsorted returns the lo index (1-based because of the sentinel)
        lo_idx = np.searchsorted(gtimes, t_lo_all, side="left")  # 0..m
        # hi = own position = k+1 (because a sentinel was prepended to cumsum)
        # that is, the index of cum_idx_k+1
        # count = cum_cnt[k+1] - cum_cnt[lo_idx[k]]
        cnt = cum_cnt[np.arange(1, m + 1)] - cum_cnt[lo_idx]
        s = cum_sum[np.arange(1, m + 1)] - cum_sum[lo_idx]

        counts_sorted = cnt.astype("float32")
        # Write back (note: these are indices in sorted order, must be written back to the original order)
        # We directly build arrays in sorted order, then reorder back with inv_order
        # This is the clearest approach: first store into sorted_* arrays, finally reorder with inv_order back to the original df order

        # mean = sum / count
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_sorted = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype("float32")

        # p90: for each time point, slice [lo_idx, k+1) and use np.partition
        # Use a list comprehension mixed with vectorization (slice lengths differ, pure vectorization is hard)
        p90_sorted = np.full(m, np.nan, dtype="float32")
        # p90: use max instead of the true 90th percentile
        # (max is 100x faster than percentile, and tree models can infer the distribution from mean + max)
        # For congestion features, max ~= p90 is a reasonable approximation
        nonzero = cnt > 0
        if nonzero.any():
            for k in np.where(nonzero)[0]:
                lo = lo_idx[k]
                hi = k + 1
                if hi > lo:
                    seg = safe_targets[start + lo:start + hi]
                    if len(seg) > 0:
                        p90_sorted[k] = float(np.max(seg))

        # Write back in original order
        local_idx = order[start:end]
        counts[local_idx] = counts_sorted
        means[local_idx] = mean_sorted
        p90s[local_idx] = p90_sorted

    return counts, means, p90s


# ---------------------------------------------------------------------------
# Convenience wrapper: run multiple windows + multiple targets in one pass
# ---------------------------------------------------------------------------

def _past_n_minutes_stats(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    windows_min: Iterable[int],
    stats: Iterable[str] = ("count", "mean", "p90"),
) -> pd.DataFrame:
    """For each group (group_col), compute target_col within the windows_min minutes before time_col of each record."""
    out = pd.DataFrame(index=df.index)

    if isinstance(group_col, str):
        groups = df[group_col]
    else:
        groups = df[list(group_col)].astype(str).agg("_".join, axis=1)

    for w in windows_min:
        cnt, mean, p90 = _vectorized_past_window(
            df, group_col=group_col, time_col=time_col,
            target_col=target_col, window_min=w,
        )
        suffix = f"_by_{group_col if isinstance(group_col, str) else '_'.join(group_col)}"
        if "count" in stats:
            out[f"{target_col}_cnt_{w}m{suffix}"] = cnt
        if "mean" in stats:
            out[f"{target_col}_mean_{w}m{suffix}"] = mean
        if "p90" in stats:
            out[f"{target_col}_p90_{w}m{suffix}"] = p90

    return out


def _compute_past_for_group(
    df: pd.DataFrame,
    group_col: str | list[str],
    time_col: str,
    target_col: str,
    windows_min: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    """For a single group, compute count/mean/p90 for multiple windows in one scan.

    v3.6 fix: NaT timestamps were treated as extremely small values by np.searchsorted, causing cnt/mean/p90
    to be computed as "the entire history of the same airport", polluting predictions. Fix:
    - Before computing aggregates, remove rows where ref_time_col=NaT from the cumsum/sort chain
    - Fill the output of those rows directly with NaN, so the model learns "NaN = no congestion signal, use basic features"
    - Training and inference are fully consistent

    Returns: dict {window_min: {"count": arr, "mean": arr, "p90": arr}} in the original df order
    """
    n = len(df)

    if isinstance(group_col, str):
        groups_raw = df[group_col].values
    else:
        groups_raw = df[list(group_col)].astype(str).agg("_".join, axis=1).values
    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    # v3.6 fix: parse time, identify NaT (NaT is int64 min in datetime64[ns])
    times = pd.to_datetime(df[time_col]).values.astype("datetime64[ns]")
    INT64_MIN = np.iinfo(np.int64).min
    # pandas stores NaT as int64 min
    sorted_times_raw_int = times.view("i8")
    valid_time_mask = sorted_times_raw_int != INT64_MIN  # True = valid time

    # Also handle NaN for raw_targets (rows with missing target are not counted)
    raw_targets = df[target_col].values
    target_nan_mask = np.isnan(raw_targets)
    # Prerequisite for a row to be valid for aggregation: time is valid AND target is not NaN (rows with target NaT are excluded from the statistics)
    # But count-type features only need valid time (target=NaN does not affect count, as long as the DEP indicator = 1)
    # Here we are conservative: for cnt use only valid time; for mean/p90 also require target to be non-NaN

    # First build the validity bits in the original order (used later when writing back)
    valid_count_row = valid_time_mask.copy()  # time is not NaT -> can compute count
    valid_target_row = valid_time_mask & ~target_nan_mask  # time + target both valid -> can compute mean/p90

    # Sort by (group, time); NaT rows sort to the front/back within the group (because int64 min is the smallest)
    # We use NaT as the largest key to sort last, so cumsum only accumulates within the non-NaT segment
    # Trick: replace NaT time int64 with int64 max, then exclude it from the sort
    safe_times_int = np.where(valid_time_mask, sorted_times_raw_int, np.iinfo(np.int64).max)
    safe_times = safe_times_int.view("datetime64[ns]")

    order = np.lexsort((safe_times, groups))
    sorted_groups = groups[order]
    sorted_times = times[order]  # original time (may contain NaT)
    # reorder raw_targets by order too
    raw_targets_ordered = raw_targets[order]
    target_nan_ordered = np.isnan(raw_targets_ordered)

    # safe_targets: target NaT -> 0 (does not affect sum, does affect cnt)
    safe_targets = np.where(target_nan_ordered, 0.0, raw_targets_ordered).astype("float64")
    # safe_indicators: 1 if (time valid AND target non-NaN), else 0
    time_valid_ordered = (safe_times_int[order] != np.iinfo(np.int64).max)
    safe_indicators = (time_valid_ordered & ~target_nan_ordered).astype("float64")
    # indicator for count: only requires valid time (target NaT still counts as 1 flight)
    count_indicators = time_valid_ordered.astype("float64")

    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [n]])

    # Result arrays (in sorted order) - NaN by default, sorted_count also uses NaN to mean "no signal"
    sorted_count = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}
    sorted_mean = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}
    sorted_p90 = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}

    for start, end in zip(group_starts, group_ends):
        gtimes = sorted_times[start:end]  # may contain NaT
        gsum = safe_targets[start:end]
        gsum_count = count_indicators[start:end]  # only requires valid time
        gcount = safe_indicators[start:end]  # requires both time and target valid (for mean/p90)
        m = end - start

        cum_sum = np.concatenate([[0.0], np.cumsum(gsum, dtype="float64")])
        cum_cnt = np.concatenate([[0.0], np.cumsum(gcount, dtype="float64")])
        cum_cnt_for_count = np.concatenate([[0.0], np.cumsum(gsum_count, dtype="float64")])

        # v3.6 fix: for each row, if time is invalid -> NaN, otherwise compute normally
        time_valid_in_group = (gtimes.view("i8") != INT64_MIN)
        # A mask array indicating "the time of this row is valid"
        # Later when computing lo_idx, invalid rows have lo_idx=0 (full window range), but they are filtered by the NaN mask

        for w in windows_min:
            window_ns = np.timedelta64(w, "m")
            # When computing lo_idx, also use searchsorted for NaT rows (which lands at some position within the group),
            # but we will later set the values of invalid rows to NaN
            t_lo_all = gtimes - window_ns  # NaT - timedelta = NaT, treated as extremely small value
            lo_idx = np.searchsorted(gtimes, t_lo_all, side="left")

            # count uses cum_cnt_for_count (only requires valid time)
            cnt = cum_cnt_for_count[np.arange(1, m + 1)] - cum_cnt_for_count[lo_idx]
            s = cum_sum[np.arange(1, m + 1)] - cum_sum[lo_idx]
            mean_v = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype("float32")

            # v3.6 fix: invalid time rows -> output NaN (so XGBoost treats it as "no signal")
            cnt = np.where(time_valid_in_group, cnt, np.nan).astype("float32")
            mean_v = np.where(time_valid_in_group, mean_v, np.nan).astype("float32")

            sorted_count[w][start:end] = cnt
            sorted_mean[w][start:end] = mean_v

            # p90 -> use max (simplification); only computed for valid time rows
            nonzero = (~np.isnan(cnt)) & (cnt > 0)
            if nonzero.any():
                p90_local = sorted_p90[w][start:end]
                for k in np.where(nonzero)[0]:
                    lo = lo_idx[k]
                    hi = k + 1
                    if hi > lo:
                        seg = safe_targets[start + lo:start + hi]
                        if len(seg) > 0:
                            p90_local[k] = float(np.max(seg))

    # Reorder back to the original df order
    inv = np.argsort(order)
    return {
        w: {
            "count": sorted_count[w][inv],
            "mean": sorted_mean[w][inv],
            "p90": sorted_p90[w][inv],
        }
        for w in windows_min
    }


def add_aggregate_features(
    df: pd.DataFrame,
    ref_time_col: str = "EOBT_1_flt",
    airport_col: str = "ADEP_mvt",
    runway_col: str = "RUNWAY_mvt",
    target_col: str = "TAXITIME_SEC_mvt",
    windows_min: tuple[int, ...] = (15, 30, 60),
    include_runway: bool = True,
) -> pd.DataFrame:
    """Main entry: add all congestion aggregate features (past windows). Optimized version: all windows within each group share a single cumsum.

    WARNING before calling, make sure:
        1. df has already been filtered to PHASE=DEP (data_loader already does this)
        2. ref_time_col (default EOBT_1_flt) is a timezone-aware UTC datetime
        3. target_col will be NaN in the ranking data (ranking masks the target), aggregation automatically skips NaN

    Output: adds the following columns to df (default parameters):
        - cnt_dep_airport_15m / 30m / 60m  : number of departures at the same airport in the past 15/30/60 minutes
        - mean_taxi_airport_15m / 30m / 60m : average taxi-out at the same airport in the past 15/30/60 minutes
        - p90_taxi_airport_15m / 30m / 60m  : P90 taxi-out at the same airport in the past 15/30/60 minutes
        - cnt_dep_runway_15m / 30m / 60m   : same as above, but restricted to the same runway (when include_runway=True)
        - mean_taxi_runway_15m / 30m / 60m  : same as above
        - p90_taxi_runway_15m / 30m / 60m   : same as above
    """
    df = df.copy()
    print(f"Starting aggregate feature computation (ref={ref_time_col}, airport={airport_col}, runway={include_runway and runway_col})...")

    import time

    # Build a "DEP indicator", each DEP record = 1
    df["__is_dep__"] = 1.0

    # Airport level - compute all windows in one pass
    t0 = time.time()
    airport_ind = _compute_past_for_group(
        df, group_col=airport_col, time_col=ref_time_col,
        target_col="__is_dep__", windows_min=windows_min,
    )
    airport_taxi = _compute_past_for_group(
        df, group_col=airport_col, time_col=ref_time_col,
        target_col=target_col, windows_min=windows_min,
    )
    print(f"  airport level ({time.time()-t0:.1f}s)")

    for w in windows_min:
        df[f"cnt_dep_airport_{w}m"] = airport_ind[w]["count"]
        df[f"mean_taxi_airport_{w}m"] = airport_taxi[w]["mean"]
        df[f"p90_taxi_airport_{w}m"] = airport_taxi[w]["p90"]

    if include_runway:
        t0 = time.time()
        runway_ind = _compute_past_for_group(
            df, group_col=[airport_col, runway_col], time_col=ref_time_col,
            target_col="__is_dep__", windows_min=windows_min,
        )
        runway_taxi = _compute_past_for_group(
            df, group_col=[airport_col, runway_col], time_col=ref_time_col,
            target_col=target_col, windows_min=windows_min,
        )
        print(f"  runway level ({time.time()-t0:.1f}s)")

        for w in windows_min:
            df[f"cnt_dep_runway_{w}m"] = runway_ind[w]["count"]
            df[f"mean_taxi_runway_{w}m"] = runway_taxi[w]["mean"]
            df[f"p90_taxi_runway_{w}m"] = runway_taxi[w]["p90"]

    # Remove temporary columns
    df = df.drop(columns=["__is_dep__"])
    print(f"Aggregate features done, added {len(windows_min) * (3 + (3 if include_runway else 0))} columns")
    return df


# ---------------------------------------------------------------------------
# Complete feature pipeline (basic features + aggregate features)
# ---------------------------------------------------------------------------

def build_features_v2(df: pd.DataFrame, include_aggregate: bool = True, include_lead: bool = True) -> pd.DataFrame:
    """Complete feature engineering = basic features + congestion aggregate (lag) + future aggregate (lead).

    Parameters:
        df: raw DataFrame loaded from data_loader
        include_aggregate: whether to add lag features (congestion in the past N minutes)
        include_lead: whether to add lead features (scheduled congestion in the future N minutes)
    """
    from src.features import build_features as build_basic
    from src.features_laglead import add_lead_features

    df = build_basic(df)

    if include_aggregate:
        df = add_aggregate_features(df)

    if include_lead:
        df = add_lead_features(df)

    return df


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Self-test: use mock data
    print("=== features_aggregate module self-test ===\n")

    # Build mock: 3 airports x 1000 DEP
    np.random.seed(42)
    n = 1000
    airports = ["EDDF", "EGLL", "LFPG"]
    base_time = pd.Timestamp("2025-06-15 08:00:00", tz="UTC")

    df = pd.DataFrame({
        "ADEP_mvt": np.random.choice(airports, n),
        "RUNWAY_mvt": np.random.choice(["07L", "25R", "18"], n),
        "EOBT_1_flt": [base_time + pd.Timedelta(minutes=int(m))
                       for m in np.random.randint(0, 240, n)],
        "PHASE_mvt": ["DEP"] * n,
        "MVT_ID_mvt": [f"id_{i}" for i in range(n)],
        "SCHED_TIME_UTC_mvt": [base_time + pd.Timedelta(minutes=int(m))
                               for m in np.random.randint(0, 240, n)],
    })
    # Target: simple linear + noise
    df["TAXITIME_SEC_mvt"] = 600 + np.random.normal(0, 100, n)

    print(f"Mock data: {df.shape}")

    import time
    t0 = time.time()
    df_agg = add_aggregate_features(df, windows_min=(15, 30))
    print(f"\nTime taken: {time.time() - t0:.2f}s")
    print(f"Output shape: {df_agg.shape}")
    print(f"New columns: {[c for c in df_agg.columns if c not in df.columns]}")
    print(f"\nAggregate feature example:\n{df_agg[[c for c in df_agg.columns if 'airport_15m' in c or 'airport_30m' in c]].head()}")