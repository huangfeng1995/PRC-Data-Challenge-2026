"""
Lag / Lead feature module
=========================

Learned from the 2nd place of 2025 jubilant-vase:
"context-Aware Features: Implementation of Lag/Lead features to capture flight dynamics"

Concretely for taxi-out:
- **Lag**: statistics of the same airport/runway over the past N minutes (in features_aggregate.py)
- **Lead**: density of "confirmed/scheduled" flights at the same airport/runway in the next N minutes

Lead features reflect "I already know it will get congested next, so I should predict now that taxi-out will grow".
But note: Lead can only use "known/scheduled" information (EOBT_flt), it must not introduce leakage.

Note on the feasibility of Lead features:
- the ranking data contains information on all flights with PHASE=DEP, so we can count how many depart in the next 30 minutes
- no leakage is introduced, because these "future flights" are known at prediction time

Performance notes (v3 cumsum refactor):
  - same as features_aggregate, O(N²) replaced with O(N) cumsum
  - p90 uses np.percentile (sliced version)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _vectorized_future_window(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    window_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized computation of count / mean / p90 inside the (t, t + window_min] window.

    Algorithm (O(N)):
      1. Sort by (group, time)
      2. cumsum
      3. For each t_now:
         - lo = k+1 (excluding itself)
         - hi = searchsorted(group_times, t_now + window, 'left')
         - count = cumcnt[hi] - cumcnt[lo]

    Returns three numpy arrays (count, mean, p90), in the original order of df.
    """
    n = len(df)

    if isinstance(group_col, str):
        groups_raw = df[group_col].values
    else:
        groups_raw = df[list(group_col)].astype(str).agg("_".join, axis=1).values

    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    times = pd.to_datetime(df[time_col]).values.astype("datetime64[ns]")
    order = np.lexsort((times, groups))

    sorted_groups = groups[order]
    sorted_times = times[order]
    raw_targets = df[target_col].values[order]

    not_nan = ~np.isnan(raw_targets)
    safe_targets = np.where(not_nan, raw_targets, 0.0).astype("float64")
    safe_indicators = not_nan.astype("float64")

    counts = np.zeros(n, dtype="float32")
    means = np.full(n, np.nan, dtype="float32")
    p90s = np.full(n, np.nan, dtype="float32")

    window_ns = np.timedelta64(window_min, "m")

    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [n]])

    for start, end in zip(group_starts, group_ends):
        gtimes = sorted_times[start:end]
        gsum = safe_targets[start:end]
        gcount = safe_indicators[start:end]
        m = end - start

        cum_sum = np.concatenate([[0.0], np.cumsum(gsum, dtype="float64")])
        cum_cnt = np.concatenate([[0.0], np.cumsum(gcount, dtype="float64")])

        # Window upper bound (for each timestamp)
        t_hi_all = gtimes + window_ns
        # searchsorted returns the hi index (1-based because of the sentinel)
        hi_idx = np.searchsorted(gtimes, t_hi_all, side="left")  # 0..m
        # lo = own position k+1 (excluding itself)
        cnt = cum_cnt[hi_idx] - cum_cnt[np.arange(1, m + 1)]
        s = cum_sum[hi_idx] - cum_sum[np.arange(1, m + 1)]

        counts_sorted = cnt.astype("float32")
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_sorted = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype("float32")

        p90_sorted = np.full(m, np.nan, dtype="float32")
        # Use max instead of p90 (performance vs. effect trade-off)
        nonzero = cnt > 0
        if nonzero.any():
            for k in np.where(nonzero)[0]:
                lo = k + 1
                hi = hi_idx[k]
                if hi > lo:
                    seg = safe_targets[start + lo:start + hi]
                    if len(seg) > 0:
                        p90_sorted[k] = float(np.max(seg))

        local_idx = order[start:end]
        counts[local_idx] = counts_sorted
        means[local_idx] = mean_sorted
        p90s[local_idx] = p90_sorted

    return counts, means, p90s


def _compute_future_for_group(
    df: pd.DataFrame,
    group_col: str | list[str],
    time_col: str,
    target_col: str,
    windows_min: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    """For a single group, compute count/mean/p90 for multiple windows in one scan (lead = future window).

    v3.6 fix: same as _compute_past_for_group, rows with a NaT timestamp output NaN instead of "all history".

    Returns: dict {window_min: {"count": arr, "mean": arr, "p90": arr}} in the original df order
    """
    n = len(df)

    if isinstance(group_col, str):
        groups_raw = df[group_col].values
    else:
        groups_raw = df[list(group_col)].astype(str).agg("_".join, axis=1).values
    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    # v3.6 fix: identify NaT rows
    times = pd.to_datetime(df[time_col]).values.astype("datetime64[ns]")
    INT64_MIN = np.iinfo(np.int64).min
    times_int = times.view("i8")
    valid_time_mask = times_int != INT64_MIN

    raw_targets = df[target_col].values
    target_nan_mask = np.isnan(raw_targets)

    # Replace NaT with the max key so it sorts last
    safe_times_int = np.where(valid_time_mask, times_int, np.iinfo(np.int64).max)

    order = np.lexsort((safe_times_int, groups))
    sorted_groups = groups[order]
    sorted_times = times[order]
    raw_targets_ordered = raw_targets[order]
    target_nan_ordered = np.isnan(raw_targets_ordered)

    safe_targets = np.where(target_nan_ordered, 0.0, raw_targets_ordered).astype("float64")
    time_valid_ordered = (safe_times_int[order] != np.iinfo(np.int64).max)
    safe_indicators = (time_valid_ordered & ~target_nan_ordered).astype("float64")
    count_indicators = time_valid_ordered.astype("float64")

    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [n]])

    sorted_count = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}
    sorted_mean = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}
    sorted_p90 = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}

    for start, end in zip(group_starts, group_ends):
        gtimes = sorted_times[start:end]
        gsum = safe_targets[start:end]
        gsum_count = count_indicators[start:end]
        gcount = safe_indicators[start:end]
        m = end - start

        cum_sum = np.concatenate([[0.0], np.cumsum(gsum, dtype="float64")])
        cum_cnt = np.concatenate([[0.0], np.cumsum(gcount, dtype="float64")])
        cum_cnt_for_count = np.concatenate([[0.0], np.cumsum(gsum_count, dtype="float64")])

        time_valid_in_group = (gtimes.view("i8") != INT64_MIN)

        for w in windows_min:
            window_ns = np.timedelta64(w, "m")
            t_hi_all = gtimes + window_ns
            hi_idx = np.searchsorted(gtimes, t_hi_all, side="left")
            cnt = cum_cnt_for_count[hi_idx] - cum_cnt_for_count[np.arange(1, m + 1)]
            s = cum_sum[hi_idx] - cum_sum[np.arange(1, m + 1)]
            mean_v = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype("float32")

            # v3.6 fix: invalid time rows -> NaN
            cnt = np.where(time_valid_in_group, cnt, np.nan).astype("float32")
            mean_v = np.where(time_valid_in_group, mean_v, np.nan).astype("float32")

            sorted_count[w][start:end] = cnt
            sorted_mean[w][start:end] = mean_v

            nonzero = (~np.isnan(cnt)) & (cnt > 0)
            if nonzero.any():
                p90_local = sorted_p90[w][start:end]
                for k in np.where(nonzero)[0]:
                    lo = k + 1
                    hi = hi_idx[k]
                    if hi > lo:
                        seg = safe_targets[start + lo:start + hi]
                        if len(seg) > 0:
                            p90_local[k] = float(np.max(seg))

    inv = np.argsort(order)
    return {
        w: {
            "count": sorted_count[w][inv],
            "mean": sorted_mean[w][inv],
            "p90": sorted_p90[w][inv],
        }
        for w in windows_min
    }


def add_lead_features(
    df: pd.DataFrame,
    ref_time_col: str = "EOBT_1_flt",
    airport_col: str = "ADEP_mvt",
    runway_col: str = "RUNWAY_mvt",
    target_col: str = "TAXITIME_SEC_mvt",
    windows_min: tuple[int, ...] = (15, 30, 60),
    include_runway: bool = True,
) -> pd.DataFrame:
    """Add all lead (future) features. Optimized version: share one cumsum across all windows per group.

    New output columns (default):
        - cnt_lead_airport_15m / 30m / 60m: number of departures at the same airport in the next N minutes
        - mean_lead_taxi_airport_15m / 30m / 60m: average taxi-out at the same airport in the next N minutes
        - p90_lead_taxi_airport_15m / 30m / 60m: P90 at the same airport in the next N minutes
        - same at the runway level
    """
    df = df.copy()
    print(f"Starting lead feature computation (ref={ref_time_col}, airport={airport_col}, runway={include_runway and runway_col})...")

    import time

    df["__is_dep__"] = 1.0

    t0 = time.time()
    airport_ind = _compute_future_for_group(
        df, group_col=airport_col, time_col=ref_time_col,
        target_col="__is_dep__", windows_min=windows_min,
    )
    airport_taxi = _compute_future_for_group(
        df, group_col=airport_col, time_col=ref_time_col,
        target_col=target_col, windows_min=windows_min,
    )
    print(f"  Airport level ({time.time()-t0:.1f}s)")

    for w in windows_min:
        df[f"cnt_lead_airport_{w}m"] = airport_ind[w]["count"]
        df[f"mean_lead_taxi_airport_{w}m"] = airport_taxi[w]["mean"]
        df[f"p90_lead_taxi_airport_{w}m"] = airport_taxi[w]["p90"]

    if include_runway:
        t0 = time.time()
        runway_ind = _compute_future_for_group(
            df, group_col=[airport_col, runway_col], time_col=ref_time_col,
            target_col="__is_dep__", windows_min=windows_min,
        )
        runway_taxi = _compute_future_for_group(
            df, group_col=[airport_col, runway_col], time_col=ref_time_col,
            target_col=target_col, windows_min=windows_min,
        )
        print(f"  Runway level ({time.time()-t0:.1f}s)")

        for w in windows_min:
            df[f"cnt_lead_runway_{w}m"] = runway_ind[w]["count"]
            df[f"mean_lead_taxi_runway_{w}m"] = runway_taxi[w]["mean"]
            df[f"p90_lead_taxi_runway_{w}m"] = runway_taxi[w]["p90"]

    df = df.drop(columns=["__is_dep__"])
    print(f"Lead features done, added {len(windows_min) * (3 + (3 if include_runway else 0))} columns")
    return df


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== features_laglead module self-test ===\n")

    np.random.seed(42)
    base_time = pd.Timestamp("2025-06-15 08:00:00", tz="UTC")
    times = [base_time + pd.Timedelta(minutes=int(m))
             for m in np.random.randint(0, 240, 500)]
    df = pd.DataFrame({
        "ADEP_mvt": np.random.choice(["EDDF", "EGLL"], 500),
        "RUNWAY_mvt": np.random.choice(["07L", "25R"], 500),
        "EOBT_1_flt": times,
        "PHASE_mvt": ["DEP"] * 500,
        "TAXITIME_SEC_mvt": np.random.normal(600, 100, 500),
    })

    df_out = add_lead_features(df, windows_min=(15, 30))
    print(f"\nOutput shape: {df_out.shape}")
    new_cols = [c for c in df_out.columns if c not in df.columns]
    print(f"New columns: {new_cols}")

    # Sanity check: lead cnt should be proportional to the number of flights
    print("\nSample:")
    sample = df_out.sort_values("EOBT_1_flt").head(3)
    print(sample[["EOBT_1_flt", "ADEP_mvt", "cnt_lead_airport_15m", "cnt_lead_airport_30m"]])
