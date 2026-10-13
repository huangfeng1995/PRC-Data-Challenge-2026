"""
Stage 1 Real Queue Feature Module
================================

Definitions (all leak-free, past-only, exclude self i):
- Reference time point ref_i = AOBT_3_flt preferred, falls back to EOBT_1_flt when missing
- New ref_was_estimated indicator column (fallback = 1)
- Only handles DEP departure rows (filtered by data_loader)

Feature set:
A. dep_mvt_cnt_airport_{5,10,15}m: number of other DEP flights at the same airport whose MVT_j falls in the window (ref_i - w, ref_i]
   = real departure throughput (flights that already took off, reflects runway throughput / queue drain speed)
B. dep_queue_n_airport: number of other DEP flights at the same airport at time ref_i that "have AOBT but not yet MVT"
   = real departure queue length
C. dep_queue_heavy_n_airport: count in B with WK_TBL_CAT in {H, J} (wake turbulence composition, heavy aircraft taxi slowly)
D (control, non-primary): runway-level dep_mvt_cnt_runway_* + dep_queue_n_runway
   - RUNWAY_mvt may be the actually used runway (ex post), carries a slight ex-post information risk, excluded from main features by default
E (optional): arr_mvt_cnt_airport_{10,15}m: number of actual ARR landings at the same airport before ref (inbound pressure)

Red lines (all satisfied):
- Own MVT/BLOCK/TAXITIME are never available
- Only other flights are counted (MVT_ID != i)
- Window strictly < ref_i (dep_mvt_cnt uses (ref_i - w, ref_i), i.e. MVT_j in (ref_i - w, ref_i])
- Reconstruct the world state at time ref_i using other flights j's MVT_j / AOBT_j (legitimate historical reconstruction)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reference time point construction
# ---------------------------------------------------------------------------

def add_ref_time(df: pd.DataFrame) -> pd.DataFrame:
    """Build reference time point ref_dt:
    - AOBT_3_flt not missing -> ref_dt = AOBT_3_flt (actual pushback, highest priority)
    - otherwise -> ref_dt = EOBT_1_flt (scheduled estimate, FPL), and ref_was_estimated=1
    """
    df = df.copy()
    aobt = pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce")
    eobt = pd.to_datetime(df["EOBT_1_flt"], utc=True, errors="coerce")
    df["ref_was_estimated"] = aobt.isna().astype("int8")
    df["ref_dt"] = aobt.fillna(eobt)
    return df


# ---------------------------------------------------------------------------
# A. dep_mvt_cnt_airport_{w}m: actual departures at same airport in past w minutes
# ---------------------------------------------------------------------------

def _past_dep_mvt_cnt(
    df: pd.DataFrame,
    airport_col: str,
    mvt_col: str,
    ref_col: str,
    windows_min: tuple[int, ...],
    exclude_self_col: str | None = None,
    runway_col: str | None = None,
) -> pd.DataFrame:
    """For each ref_i, count MVT_j of other DEP flights at the same airport inside the window (ref_i - w, ref_i].

    Excluded:
    - Self (MVT_ID != i, when exclude_self_col is provided)
    - Rows with NaT timestamps

    Returns a DataFrame with index=df.index, one column per window.
    """
    n = len(df)
    airports = df[airport_col].astype(str).values
    mvt_times = pd.to_datetime(df[mvt_col], utc=True, errors="coerce").values.astype("datetime64[ns]")
    ref_times = pd.to_datetime(df[ref_col], utc=True, errors="coerce").values.astype("datetime64[ns]")
    INT64_MIN = np.iinfo(np.int64).min
    INT64_MAX = np.iinfo(np.int64).max
    mvt_valid = mvt_times.view("i8") != INT64_MIN
    ref_valid = ref_times.view("i8") != INT64_MIN

    # Exclude self (if MVT_ID exists)
    if exclude_self_col is not None and exclude_self_col in df.columns:
        own_mvt_id = df[exclude_self_col].astype(str).values
    else:
        own_mvt_id = None

    # Runway grouping (optional)
    if runway_col is not None and runway_col in df.columns:
        runways = df[runway_col].astype(str).values
        groups_raw = np.char.add(np.char.add(airports, "_"), runways)
    else:
        groups_raw = airports

    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    # Safe sort key (NaT -> INT64_MAX sorted last, not participating in cumsum)
    safe_mvt_int = np.where(mvt_valid, mvt_times.view("i8"), INT64_MAX)

    order = np.lexsort((safe_mvt_int, groups))
    sorted_groups = groups[order]
    sorted_mvt_valid = mvt_valid[order]
    sorted_own_mvt_id = own_mvt_id[order] if own_mvt_id is not None else None

    # cumsum counts (number of "flights that already have MVT" at each sorted position)
    cum_count = np.concatenate([[0], np.cumsum(sorted_mvt_valid.astype("float64"))])

    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [n]])

    out = {f"{w}m": np.full(n, np.nan, dtype="float32") for w in windows_min}

    for start, end in zip(group_starts, group_ends):
        m = end - start
        if m == 0:
            continue
        ref_in_group = ref_times[order[start:end]]
        ref_valid_in_group = ref_valid[order[start:end]]
        own_in_group = sorted_own_mvt_id[start:end] if sorted_own_mvt_id is not None else None

        # In this group, the original row of the k-th sorted position = order[start + k]
        # searchsorted operates within [start, end)
        local_mvt_valid = sorted_mvt_valid[start:end]
        cum_local = np.concatenate([[0.0], np.cumsum(local_mvt_valid.astype("float64"))])  # length m+1

        for w in windows_min:
            window_ns = np.timedelta64(w, "m")
            t_lo = ref_in_group - window_ns  # window lower bound (> this)
            # Interval (ref_i - w, ref_i] half-open half-closed: lo = first > ref_i - w, hi = first > ref_i
            t_lo_safe = np.where(ref_valid_in_group, t_lo, np.datetime64("NaT"))
            lo_idx = np.searchsorted(
                mvt_times[order[start:end]],
                t_lo_safe,
                side="right",  # strictly > ref_i - w
            )
            # hi = first MVT > ref_i, i.e. searchsorted(ref, side='right') - 1? No, we use the last one <= ref_i
            # In practice: dep_mvt_cnt = cum_local[hi] - cum_local[lo], where hi is the index of the first MVT > ref_i
            t_hi_safe = np.where(ref_valid_in_group, ref_in_group, np.datetime64("NaT"))
            hi_idx = np.searchsorted(
                mvt_times[order[start:end]],
                t_hi_safe,
                side="right",  # strictly > ref_i, so hi is the first position > ref (excluding ref_i itself)
            )
            cnt = cum_local[hi_idx] - cum_local[lo_idx]

            # Exclude self: if own_in_group[k] == own_in_group[order[k]] ...
            # Simplification: for each k, within range [lo_idx[k], hi_idx[k]), if own_mvt_id[j] == own_in_group[k], then -1
            if own_in_group is not None:
                own_sub = own_in_group  # length m, own_mvt_id of each row in this group (sorted by mvt_time)
                cnt_adj = cnt.copy()
                for k in np.where(ref_valid_in_group & (cnt > 0))[0]:
                    lo, hi = int(lo_idx[k]), int(hi_idx[k])
                    if hi > lo:
                        # Check whether own_sub[lo:hi] equals own_in_group[k]
                        in_window = own_sub[lo:hi]
                        # Self appears in the window -> -1 (only decremented if MVT_i truly falls in the window, but the user red line forbids using MVT_i)
                        # Since we exclude via MVT_ID (not MVT_i), and MVT_i is usually not in the (ref_i-w, ref_i] window (ref_i is w minutes before own takeoff moment),
                        # MVT_i is most likely > ref_i (not yet departed) or << ref_i-w (departed long ago). Rarely MVT_i falls in -> subtract 1.
                        if own_in_group[k] in in_window:
                            cnt_adj[k] -= 1.0
                cnt = cnt_adj

            # Invalid ref -> NaN
            cnt = np.where(ref_valid_in_group, cnt, np.nan).astype("float32")
            # Write back in original order
            local_idx = order[start:end]
            out[f"{w}m"][local_idx] = cnt

    return pd.DataFrame(out, index=df.index)


def add_dep_mvt_cnt(
    df: pd.DataFrame,
    ref_col: str = "ref_dt",
    airport_col: str = "ADEP_mvt",
    runway_col: str = "RUNWAY_mvt",
    mvt_col: str = "MVT_TIME_UTC_mvt",
    windows_min: tuple[int, ...] = (5, 10, 15),
    include_runway: bool = True,
    exclude_self_col: str | None = "MVT_ID_mvt",
) -> pd.DataFrame:
    """Feature A: dep_mvt_cnt_airport_{5,10,15}m (optionally plus runway level)."""
    df = df.copy()
    print(f"  [A] dep_mvt_cnt_airport w={windows_min}...")

    out = _past_dep_mvt_cnt(
        df, airport_col=airport_col, mvt_col=mvt_col, ref_col=ref_col,
        windows_min=windows_min, exclude_self_col=exclude_self_col,
    )
    for w in windows_min:
        df[f"dep_mvt_cnt_airport_{w}m"] = out[f"{w}m"]

    if include_runway:
        out_r = _past_dep_mvt_cnt(
            df, airport_col=airport_col, mvt_col=mvt_col, ref_col=ref_col,
            windows_min=windows_min, exclude_self_col=exclude_self_col,
            runway_col=runway_col,
        )
        for w in windows_min:
            df[f"dep_mvt_cnt_runway_{w}m"] = out_r[f"{w}m"]

    return df


# ---------------------------------------------------------------------------
# B. dep_queue_n_airport: number of other DEP flights at same airport at time ref that "have AOBT but not yet MVT"
# ---------------------------------------------------------------------------

def add_dep_queue_n(
    df: pd.DataFrame,
    ref_col: str = "ref_dt",
    airport_col: str = "ADEP_mvt",
    runway_col: str = "RUNWAY_mvt",
    mvt_col: str = "MVT_TIME_UTC_mvt",
    aobt_col: str = "AOBT_3_flt",
    exclude_self_col: str | None = "MVT_ID_mvt",
    heavy_cats: tuple[str, ...] = ("H", "J"),
    include_runway: bool = True,
) -> pd.DataFrame:
    """Features B/C: queue length of "pushed back but not yet departed" flights at same airport at time ref.

    Queue definition: AOBT_j <= ref_i AND (MVT_j NaT OR MVT_j > ref_i) AND MVT_ID_j != i
    = #{j: AOBT_j <= ref_i} - #{j: AOBT_j <= ref_i AND MVT_j valid AND MVT_j <= ref_i} - self_excluded

    Implementation: for each group (airport/runway), also sort ref, then scan per ref + binary search.
    """
    df = df.copy()
    print(f"  [B] dep_queue_n_airport (heavy={heavy_cats})...", flush=True)

    airports = df[airport_col].astype(str).values
    ref_times = pd.to_datetime(df[ref_col], utc=True, errors="coerce").values.astype("datetime64[ns]")
    mvt_times = pd.to_datetime(df[mvt_col], utc=True, errors="coerce").values.astype("datetime64[ns]")
    aobt_times = pd.to_datetime(df[aobt_col], utc=True, errors="coerce").values.astype("datetime64[ns]")
    INT64_MIN = np.iinfo(np.int64).min
    INT64_MAX = np.iinfo(np.int64).max
    ref_valid = ref_times.view("i8") != INT64_MIN
    aobt_valid = aobt_times.view("i8") != INT64_MIN
    mvt_valid = mvt_times.view("i8") != INT64_MIN
    if exclude_self_col is not None and exclude_self_col in df.columns:
        own_mvt_id = df[exclude_self_col].astype(str).values
    else:
        own_mvt_id = None

    if "WK_TBL_CAT_flt" in df.columns:
        heavy_mask = df["WK_TBL_CAT_flt"].astype(str).isin(list(heavy_cats)).values
    else:
        heavy_mask = np.zeros(len(df), dtype=bool)

    aobt_int_safe = np.where(aobt_valid, aobt_times.view("i8"), INT64_MAX)  # NaT -> max, sorted last
    mvt_int_safe = np.where(mvt_valid, mvt_times.view("i8"), INT64_MAX)

    def _compute_for_group(group_arr, mask, key_label):
        """For each group, compute queue_n, queue_heavy_n (fully numpy-vectorized sweep line).
        """
        import time
        t_total0 = time.time()
        n_total = len(df)
        out_queue_n = np.full(n_total, np.nan, dtype="float32")
        out_queue_heavy = np.full(n_total, np.nan, dtype="float32")

        ref_safe = np.where(ref_valid, ref_times.view("i8"), INT64_MAX)

        # Encode group_arr as int (fast group by)
        _, group_int = np.unique(group_arr, return_inverse=True)
        unique_groups = np.unique(group_int[mask])
        print(f"    [{key_label}] groups={len(unique_groups)}", flush=True)
        for g_int in unique_groups:
            t_g0 = time.time()
            g_idx = np.where(group_int == g_int)[0]
            m = len(g_idx)
            if m == 0:
                continue
            ref_g = ref_safe[g_idx]
            ref_v_g = ref_valid[g_idx]
            aobt_g = aobt_int_safe[g_idx]
            aobt_v_g = aobt_valid[g_idx]
            mvt_g = mvt_int_safe[g_idx]
            mvt_v_g = mvt_valid[g_idx]
            own_g = own_mvt_id[g_idx] if own_mvt_id is not None else None
            heavy_g = heavy_mask[g_idx]

            # ===== Fully vectorized sweep line =====
            # events: 3 kinds (aobt enter, mvt leave, query ref)
            # Sort all at once with np.argsort, then cumsum
            n_aobt = int(aobt_v_g.sum())
            n_mvt = int(mvt_v_g.sum())
            n_query = int(ref_v_g.sum())

            times = np.concatenate([
                aobt_g[aobt_v_g],       # aobt moment
                mvt_g[mvt_v_g],          # mvt moment
                ref_g[ref_v_g],          # ref moment
            ])
            # kind: +1 (aobt), -1 (mvt), 0 (query)
            # Sort key: (time, -kind) makes +1 come before 0 come before -1
            kinds = np.concatenate([
                np.ones(n_aobt, dtype="int8"),     # +1
                -np.ones(n_mvt, dtype="int8"),     # -1
                np.zeros(n_query, dtype="int8"),   # 0
            ])
            # sort by (time, -kind) -> asc
            sort_key = times.astype("int64") * 2 + (-kinds.astype("int64") + 1) * 1
            # Simplification: use lexsort
            # numpy sort priority aobt > query > mvt achieved via sort_key (larger kind goes first)
            # sort by time asc, kind desc: +1, 0, -1
            # For each event, record idx_in_g_idx (row index within g_idx)
            idx_in_g = np.concatenate([
                np.where(aobt_v_g)[0],
                np.where(mvt_v_g)[0],
                np.where(ref_v_g)[0],
            ])
            # Use lexsort: primary = times, secondary = -kinds
            # np.lexsort([secondary, primary])  primary is the last one, sorted by secondary first
            # No, np.lexsort accepts [last_key, ..., first_key], primary is last_key (last column)
            # We want primary = times, secondary = -kinds
            order = np.lexsort((-kinds, times))
            sorted_times = times[order]
            sorted_kinds = kinds[order]
            sorted_idx = idx_in_g[order]

            # cumsum (kind is already +1/-1/0, accumulate directly)
            cum = np.cumsum(sorted_kinds.astype("int64"))
            cum_heavy = np.cumsum((sorted_kinds == 1) & heavy_g[sorted_idx]).astype("int64")

            # Position of query in the sorted sequence:
            # query kind=0 corresponds to sorted_kinds==0
            # query_idx_in_sorted = np.where(sorted_kinds == 0)[0]
            query_mask = sorted_kinds == 0
            query_positions = np.where(query_mask)[0]

            # For each query, queue = cum[query_pos]
            # But cum[query_pos] = "accumulated at query time" = cumulative +1 up to that moment - cumulative -1
            # = #{j: aobt_j <= ref AND mvt_j <= ref} + #{j: aobt_j <= ref AND mvt NaT} - #{j: mvt_j <= ref AND aobt NaT}
            # Wait, the semantics of cum are:
            #   +1 event: occurs at aobt moment (enters queue)
            #   -1 event: occurs at mvt moment (leaves queue)
            #   0 event: query (does not change cum)
            # At query time, cum = (all cumulative +1 at that moment) - (all cumulative -1 at that moment)
            # cumulative +1 = #{j: aobt_j <= ref}
            # cumulative -1 = #{j: mvt_j <= ref}  (mvt_j valid)
            # cum = #{aobt_j <= ref} - #{mvt_j <= ref}
            # = #{j: aobt_j <= ref AND (mvt NaT OR mvt_j > ref)} + #{j: aobt NaT AND mvt_j <= ref} (NaT always exists)
            # What we want is #{j: aobt_j <= ref AND (mvt NaT OR mvt > ref)}
            # So we need to exclude those j with "aobt NaT AND mvt <= ref"
            # For these j, aobt NaT -> no +1 event exists (we only add events for aobt_v_g)
            # But mvt valid + aobt NaT -> -1 event exists (because mvt_v_g)
            # And aobt NaT should not let j enter the queue - this part is handled by the aobt_v_g mask
            # Also exclude aobt_j > ref but mvt_j <= ref (never happens after sorting by ref - wait)
            # When sorting by time, +1 comes before query comes before -1. So cum[query_pos] includes all +1 and -1 before query
            # That is: cum[query] = #{j: aobt_j <= ref} - #{j: mvt_j <= ref}
            # But if mvt_j <= ref yet aobt_j > ref (not yet pushed back, already gone) - in that case cum[mvt] = -1, aobt = +0, one subtraction
            # We should exclude: when aobt NaT, -1 is still valid (because we keep the mvt_v_g mask), but aobt never "entered" the queue -> this j should not be in the queue
            # This corresponds to cum minus 1 (because mvt -1 subtracted one but aobt added none)
            # Similar to self_in_queue: for each j, if aobt NaT AND mvt <= ref -> cum subtracted 1 too many
            # But for j with aobt NaT AND mvt <= ref, under the queue definition: aobt NaT -> not counted as "has AOBT", not in queue OK
            # So cum subtracted 1 too many, we must add that 1 back
            # But this differs from self_in_queue - self_in_queue excludes "self" being in the queue
            # Summary: cum is #{aobt <= ref} - #{mvt <= ref}
            # The true queue = #{aobt_v AND aobt <= ref AND (mvt NaT OR mvt > ref)} - self
            # = #{aobt <= ref AND mvt NaT} + #{aobt <= ref AND mvt > ref} - self
            # = #{aobt <= ref} - #{aobt <= ref AND mvt_v AND mvt <= ref} - self
            # While #{aobt <= ref AND mvt_v AND mvt <= ref} != #{mvt <= ref} (the latter includes mvt with aobt NaT)
            # Difference = #{aobt NaT AND mvt_v AND mvt <= ref}
            # So queue = cum + #{aobt NaT AND mvt_v AND mvt <= ref} - self
            # = cum + cnt_aobt_nat_mvt_le_ref - self

            # Now compute cnt_aobt_nat_mvt_le_ref:
            # In the sorted sequence, aobt NaT does not generate a +1 event (already masked), but mvt NaT does not generate a -1 event
            # So for j with aobt NaT, its mvt -1 is still there - this is the reason for "cum subtracted 1 too much" above
            # Use cum_aobt_nat_mvt_le = #{k: aobt NaT AND mvt_v AND mvt <= ref_k}
            # = for each query, within range [0, query_pos) in sorted order, count mvt events (idx where kind=-1) whose aobt is NaT
            # Vectorized: for -1 events before the query in sorted order: positions where kind=-1
            # For each query_pos, find count of -1 within [0, query_pos) AND aobt_v_g[idx]=False
            # Use cum count
            # Simple: for -1 events before query, count how many have idx_in_g's aobt_v_g == False
            # Use prefix sums

            # Simpler: use the comparison of (aobt_v AND kind=+1) vs (mvt_v AND kind=-1) inside the sweep
            # But when aobt NaT, aobt_v=False, kind does not add an event - this part is already masked
            # So for each j:
            #   aobt_v=T, mvt_v=T: +1 then -1, net 0, contributes 0 to cum but "aobt <= ref AND mvt <= ref" should be subtracted - this part is inaccurate
            #   aobt_v=T, mvt_v=F: only +1, net +1 OK
            #   aobt_v=F, mvt_v=T: only -1, net -1 X (wrong deduction)
            #   aobt_v=F, mvt_v=F: 0 events, net 0 OK

            # For case (aobt_v=F, mvt_v=T): such j should not be in the queue (not pushed back), but cum subtracted 1
            # These "phantom" -1 must be added back
            # Equivalent to: queue = cum + #{j: aobt_v=F AND mvt_v=T AND mvt <= ref} - self

            # Compute phantom_mvt_le_ref:
            # In the sorted sequence, -1 events before the query whose idx has aobt_v=False -> phantom
            phantom_mask_in_sorted = (sorted_kinds == -1) & (~aobt_v_g[sorted_idx])
            # prefix sum: cum_phantom[query_pos] = #phantom up to (and including) this position
            cum_phantom = np.cumsum(phantom_mask_in_sorted.astype("int64"))
            # At query, queue = cum[query] + cum_phantom[query] - self
            query_cum = cum[query_positions] + cum_phantom[query_positions]
            query_heavy_v = cum_heavy[query_positions]

            # Write back: but sorted is by (time, -kind), query's position in sorted is query_positions
            # But query_idx_in_g is ref's row index within g_idx
            query_idx_in_g = sorted_idx[query_positions]

            # self_in_queue: self (aobt_v AND aobt <= ref AND (mvt_v=F OR mvt > ref))
            # Use vectorized comparison of aobt_g / mvt_g / ref_g
            self_in_queue = np.zeros(m, dtype=bool)
            self_heavy = np.zeros(m, dtype=bool)
            # Only compute for ref_v rows
            valid_ref_pos = np.where(ref_v_g)[0]
            self_aobt_ok = aobt_v_g[valid_ref_pos] & (aobt_g[valid_ref_pos] <= ref_g[valid_ref_pos])
            self_mvt_after = (~mvt_v_g[valid_ref_pos]) | (mvt_g[valid_ref_pos] > ref_g[valid_ref_pos])
            self_in_queue[valid_ref_pos] = self_aobt_ok & self_mvt_after
            self_heavy[valid_ref_pos] = self_in_queue[valid_ref_pos] & heavy_g[valid_ref_pos]

            # Write back (fully vectorized): query_idx_in_g is ref's row index within g_idx
            # query_cum / query_heavy_v correspond one-to-one with query_idx_in_g
            result_in_g = np.full(m, np.nan, dtype="float32")
            heavy_in_g = np.full(m, np.nan, dtype="float32")
            if len(query_idx_in_g) > 0:
                base = query_cum.astype("float32")
                base_h = query_heavy_v.astype("float32")
                # Exclude self (self_in_queue at query_idx_in_g)
                self_at_q = self_in_queue[query_idx_in_g]
                base = base - self_at_q.astype("float32")
                base_h = base_h - (self_at_q & self_heavy[query_idx_in_g]).astype("float32")
                result_in_g[query_idx_in_g] = base
                heavy_in_g[query_idx_in_g] = base_h

            out_queue_n[g_idx] = result_in_g
            out_queue_heavy[g_idx] = heavy_in_g
            if m > 100000:
                print(f"      [{key_label}] group {g_int} done in {time.time()-t_g0:.1f}s (m={m:,})", flush=True)
        print(f"    [{key_label}] total: {time.time()-t_total0:.1f}s", flush=True)
        return out_queue_n, out_queue_heavy

    # Airport level
    mask = np.ones(len(df), dtype=bool)
    queue_n_airport, queue_heavy_n_airport = _compute_for_group(airports, mask, "airport")
    df["dep_queue_n_airport"] = queue_n_airport
    df["dep_queue_heavy_n_airport"] = queue_heavy_n_airport

    # Runway level
    if include_runway:
        print(f"  [D] dep_queue_n_runway (control)...", flush=True)
        runway_str = df[runway_col].astype(str).values
        runway_groups = np.char.add(np.char.add(airports, "_"), runway_str)
        queue_n_runway, _ = _compute_for_group(runway_groups, mask, "runway")
        df["dep_queue_n_runway"] = queue_n_runway

    return df


# ---------------------------------------------------------------------------
# E. arr_mvt_cnt_airport_{10,15}m: actual ARR landings at same airport before ref
# ---------------------------------------------------------------------------

def add_arr_mvt_cnt(
    df: pd.DataFrame,
    ref_col: str = "ref_dt",
    airport_col: str = "ADEP_mvt",
    mvt_col: str = "MVT_TIME_UTC_mvt",
    phase_col: str = "PHASE_mvt",
    windows_min: tuple[int, ...] = (10, 15),
) -> pd.DataFrame:
    """Feature E: actual ARR landings at the same airport before ref (inbound pressure).

    Input df contains both DEP and ARR (controlled by data_loader); only ARR rows are counted.
    """
    df = df.copy()
    print(f"  [E] arr_mvt_cnt_airport w={windows_min} (PHASE=ARR)...")

    is_arr = df[phase_col].astype(str).values == "ARR"
    # MVT_TIME for ARR = actual landing time (inbound completion)
    # DEP also uses the same field (MVT_TIME = actual takeoff time); but we only look at MVT of PHASE=ARR
    # Key: ADEP_mvt differs for DEP/ARR rows (for ARR rows ADEP_mvt is the original departure airport)
    # Here we want "same airport": from the DEP perspective, only ARR's ADES_mvt equal to DEP's ADEP_mvt means the same airport
    # Use the ARR row's ADES_mvt (arrival airport) = DEP's ADEP_mvt
    arr_airport = df.get("ADES_mvt", df.get("ADEP_mvt")).astype(str).values
    arr_mvt = df[mvt_col].values

    # Build an arr-only subset
    out = {f"{w}m": np.full(len(df), np.nan, dtype="float32") for w in windows_min}

    # Only compute for DEP rows (since this is a DEP feature), set ARR rows to NaN
    dep_mask = df[phase_col].astype(str).values == "DEP"
    if dep_mask.sum() == 0:
        for w in windows_min:
            df[f"arr_mvt_cnt_airport_{w}m"] = out[f"{w}m"]
        return df

    # Group by airport, use ARR rows' MVT time
    INT64_MIN = np.iinfo(np.int64).min
    arr_mvt_int = pd.to_datetime(arr_mvt, utc=True, errors="coerce").values.astype("datetime64[ns]").view("i8")
    arr_valid = arr_mvt_int != INT64_MIN
    safe_arr_mvt_int = np.where(arr_valid, arr_mvt_int, np.iinfo(np.int64).max)

    # Group by airport (for DEP it is ADEP_mvt, for ARR it is ADES_mvt)
    dep_airport = df[airport_col].astype(str).values
    airport_all = np.where(is_arr, arr_airport, dep_airport)
    _, groups = np.unique(airport_all, return_inverse=True)
    groups = groups.astype(np.int64)

    # Only count ARR rows (DEP rows do not enter cumsum)
    is_arr_indicator = is_arr.astype("float64")
    safe_arr_mvt_int_for_sort = np.where(is_arr, safe_arr_mvt_int, np.iinfo(np.int64).max)
    order = np.lexsort((safe_arr_mvt_int_for_sort, groups))
    sorted_groups = groups[order]
    sorted_arr_valid = is_arr[order]  # True if is ARR

    cum_arr = np.concatenate([[0.0], np.cumsum(sorted_arr_valid.astype("float64"))])

    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    gstarts = np.concatenate([[0], boundaries])
    gends = np.concatenate([boundaries, [len(df)]])

    ref_times = pd.to_datetime(df[ref_col], utc=True, errors="coerce").values.astype("datetime64[ns]")
    ref_int = np.where(ref_times.view("i8") != INT64_MIN, ref_times.view("i8"), np.iinfo(np.int64).max)

    for w in windows_min:
        window_ns = np.timedelta64(w, "m")
        for start, end in zip(gstarts, gends):
            m = end - start
            if m == 0:
                continue
            ref_g = ref_times[order[start:end]]
            ref_valid_g = ref_times[order[start:end]].view("i8") != INT64_MIN
            ref_int_g = np.where(ref_valid_g, ref_g.view("i8"), np.iinfo(np.int64).max)
            arr_mvt_local = arr_mvt_int[order[start:end]]
            arr_valid_local = arr_valid[order[start:end]]
            cum_local = np.concatenate([[0.0], np.cumsum(arr_valid_local.astype("float64"))])

            t_lo_safe = np.where(ref_valid_g, ref_g - window_ns, np.datetime64("NaT"))
            # Interval (ref-w, ref] half-open half-closed: lo = first > ref-w, hi = first > ref
            lo_idx = np.searchsorted(
                np.where(arr_valid_local, arr_mvt_local, np.iinfo(np.int64).max).view("datetime64[ns]"),
                t_lo_safe, side="right",
            )
            t_hi_safe = np.where(ref_valid_g, ref_g, np.datetime64("NaT"))
            hi_idx = np.searchsorted(
                np.where(arr_valid_local, arr_mvt_local, np.iinfo(np.int64).max).view("datetime64[ns]"),
                t_hi_safe, side="right",
            )
            cnt = cum_local[hi_idx] - cum_local[lo_idx]
            cnt = np.where(ref_valid_g, cnt, np.nan).astype("float32")
            local_idx = order[start:end]
            out[f"{w}m"][local_idx] = cnt

    for w in windows_min:
        out[f"{w}m"] = np.where(dep_mask, out[f"{w}m"], np.nan).astype("float32")
        df[f"arr_mvt_cnt_airport_{w}m"] = out[f"{w}m"]
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_queue_features(
    df: pd.DataFrame,
    include_a: bool = True,
    include_b: bool = True,
    include_c: bool = True,
    include_d: bool = False,
    include_e: bool = False,
    windows_dep_mvt: tuple[int, ...] = (5, 10, 15),
    windows_arr_mvt: tuple[int, ...] = (10, 15),
) -> pd.DataFrame:
    """Main entry point: add all Stage 1 queue features.

    WARNING dependencies:
    - df must already be filtered to PHASE=DEP (if include_e=False), or contain both DEP+ARR (if include_e=True)
    - Columns: MVT_TIME_UTC_mvt, AOBT_3_flt, EOBT_1_flt, ADEP_mvt, RUNWAY_mvt, WK_TBL_CAT_flt, MVT_ID_mvt
    """
    df = df.copy()
    df = add_ref_time(df)

    if include_a:
        df = add_dep_mvt_cnt(
            df, ref_col="ref_dt", windows_min=windows_dep_mvt,
            include_runway=include_d,
        )
    if include_b or include_c or include_d:
        df = add_dep_queue_n(
            df, ref_col="ref_dt",
            include_runway=include_d,
        )
    if include_e:
        df = add_arr_mvt_cnt(
            df, ref_col="ref_dt", windows_min=windows_arr_mvt,
        )

    return df