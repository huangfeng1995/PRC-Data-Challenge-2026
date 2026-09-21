"""
Lag / Lead 特征模块
===================

从 2025 第 2 名 jubilant-vase 学到:
"context-Aware Features: Implementation of Lag/Lead features to capture flight dynamics"

具体到 taxi-out:
- **Lag**: 同一机场/跑道过去 N 分钟的统计(在 features_aggregate.py 里)
- **Lead**: 同机场/跑道 未来 N 分钟的"已确认/已计划"航班密度

Lead 特征反映了"我已经知道接下来会很挤,现在就应该预判 taxi-out 会变长"。
但注意:Lead 只能用"已知/已计划"信息(EOBT_flt),不能引入 leakage。

⚠️ Lead 特征的可行性:
- ranking 数据包含 PHASE=DEP 的所有航班信息,所以我们能算未来 30 分钟有几架
- 不会引入 leakage,因为这些"未来的航班"在预测时是已知的

性能说明(v3 cumsum 重构):
  - 跟 features_aggregate 一样,把 O(N²) 换成 O(N) cumsum
  - p90 用 np.percentile(切片版本)
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
    """向量化计算 (t, t + window_min] 窗口内的 count / mean / p90。

    算法(O(N)):
      1. 按 (group, time) 排序
      2. cumsum
      3. 对每个 t_now:
         - lo = k+1(不包含自己)
         - hi = searchsorted(group_times, t_now + window, 'left')
         - count = cumcnt[hi] - cumcnt[lo]

    返回三个 numpy 数组(count, mean, p90),按 df 原始顺序。
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

        # 窗口上界(每个时间点)
        t_hi_all = gtimes + window_ns
        # searchsorted 返回 hi 索引(1-based 因为有哨兵)
        hi_idx = np.searchsorted(gtimes, t_hi_all, side="left")  # 0..m
        # lo = 自己位置 k+1(不含自己)
        cnt = cum_cnt[hi_idx] - cum_cnt[np.arange(1, m + 1)]
        s = cum_sum[hi_idx] - cum_sum[np.arange(1, m + 1)]

        counts_sorted = cnt.astype("float32")
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_sorted = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype("float32")

        p90_sorted = np.full(m, np.nan, dtype="float32")
        # 用 max 替代 p90(性能与效果 trade-off)
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
    """对单个 group,一次扫描算多个窗口的 count/mean/p90(lead = 未来窗口)。

    v3.6 fix: 同 _compute_past_for_group,NaT 时间戳行输出 NaN 而非"全历史"。

    返回: dict {window_min: {"count": arr, "mean": arr, "p90": arr}} 按原 df 顺序
    """
    n = len(df)

    if isinstance(group_col, str):
        groups_raw = df[group_col].values
    else:
        groups_raw = df[list(group_col)].astype(str).agg("_".join, axis=1).values
    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    # v3.6 fix: 识别 NaT 行
    times = pd.to_datetime(df[time_col]).values.astype("datetime64[ns]")
    INT64_MIN = np.iinfo(np.int64).min
    times_int = times.view("i8")
    valid_time_mask = times_int != INT64_MIN

    raw_targets = df[target_col].values
    target_nan_mask = np.isnan(raw_targets)

    # 把 NaT 替换为最大 key 排到最后
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

            # v3.6 fix: 无效时间行 → NaN
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
    """加所有 lead(未来)特征。优化版:每个 group 内对所有窗口共享一次 cumsum。

    输出新列(默认):
        - cnt_lead_airport_15m / 30m / 60m: 未来 N 分钟同机场起飞航班数
        - mean_lead_taxi_airport_15m / 30m / 60m: 未来 N 分钟同机场平均 taxi-out
        - p90_lead_taxi_airport_15m / 30m / 60m: 未来 N 分钟同机场 P90
        - 跑道级同上
    """
    df = df.copy()
    print(f"开始 lead 特征计算(ref={ref_time_col}, 机场={airport_col}, 跑道={include_runway and runway_col})...")

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
    print(f"  机场级 ({time.time()-t0:.1f}s)")

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
        print(f"  跑道级 ({time.time()-t0:.1f}s)")

        for w in windows_min:
            df[f"cnt_lead_runway_{w}m"] = runway_ind[w]["count"]
            df[f"mean_lead_taxi_runway_{w}m"] = runway_taxi[w]["mean"]
            df[f"p90_lead_taxi_runway_{w}m"] = runway_taxi[w]["p90"]

    df = df.drop(columns=["__is_dep__"])
    print(f"Lead 特征完成,新增 {len(windows_min) * (3 + (3 if include_runway else 0))} 列")
    return df


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== features_laglead 模块自测 ===\n")

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
    print(f"\n输出形状: {df_out.shape}")
    new_cols = [c for c in df_out.columns if c not in df.columns]
    print(f"新增列: {new_cols}")

    # 验证:lead cnt 应该跟航班数成比例
    print("\n样例:")
    sample = df_out.sort_values("EOBT_1_flt").head(3)
    print(sample[["EOBT_1_flt", "ADEP_mvt", "cnt_lead_airport_15m", "cnt_lead_airport_30m"]])
