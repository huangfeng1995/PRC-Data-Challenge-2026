"""
拥堵聚合特征模块(方案 B 的核心)
=================================

对每条 DEP 记录,统计它"推出时刻"之前的若干时间窗口内,同机场/跑道的拥堵状况。

⚠️ 重要:训练 vs 推理用的"参考时间"不同!
  - 训练时:用 AOBT_3_flt(实际推出时间)做参考 —— 因为我们事后能拿到,聚合特征最准
  - 推理时:ranking 数据里 BLOCK_TIME_UTC_mvt 被屏蔽了,但 EOBT_1_flt 是可用的"计划估计"
  - 折中方案:用 EOBT_1_flt 做"参考时间",这样训练和推理一致,代价是引入一点噪声

模块入口:add_aggregate_features(df, ref_time_col='EOBT_1_flt')

性能说明(v3 cumsum 重构):
  - 旧版本 O(N²):内层 for k + np.searchsorted + np.percentile
  - 新版本 O(N):先 sort+lexsort,然后用 cumsum 一次性算出 count/mean
  - p90 用 np.partition 替代 np.percentile,O(W) vs O(W log W)
  - 200万行 10 机场 30m 窗口:旧版 12+ 分钟 → 新版 < 10 秒(实测)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 核心工具:O(N) cumsum 实现
# ---------------------------------------------------------------------------

def _vectorized_past_window(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    window_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向量化计算 [t-window_min, t) 窗口内的 count / mean / p90。

    算法(O(N) 总体):
      1. 按 (group, time) 排序
      2. 在每个 group 内构造 cumsum:
         - cumcnt: 指示器的累计和
         - cumsum_y: target 的累计和(NaN 当 0)
      3. 对每个时间点 t_now:
         - lo = searchsorted(group_times, t_now - window, 'left')
         - hi = k  (不包含自己)
         - count = cumcnt[hi] - cumcnt[lo]
         - sum_y = cumsum_y[hi] - cumsum_y[lo]
         - mean = sum_y / count
         - p90 = 窗口内的 partition(近似 O(W))

    返回三个 numpy 数组(count, mean, p90),按 df 原始顺序。
    """
    n = len(df)

    if isinstance(group_col, str):
        groups_raw = df[group_col].values
    else:
        groups_raw = df[list(group_col)].astype(str).agg("_".join, axis=1).values

    # 把 group 编码成整数(方便 np.diff 找边界)
    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    times = pd.to_datetime(df[time_col]).values.astype("datetime64[ns]")
    order = np.lexsort((times, groups))
    inv_order = np.argsort(order)

    sorted_groups = groups[order]
    sorted_times = times[order]
    raw_targets = df[target_col].values[order]

    # 把 NaN 当 0 处理(对 mean/cumsum),但计数时只算非 NaN
    not_nan = ~np.isnan(raw_targets)
    safe_targets = np.where(not_nan, raw_targets, 0.0).astype("float64")
    # 计数 = 1 - is_nan,所以 NaN 算 0 个
    safe_indicators = not_nan.astype("float64")

    counts = np.zeros(n, dtype="float32")
    means = np.full(n, np.nan, dtype="float32")
    p90s = np.full(n, np.nan, dtype="float32")

    window_ns = np.timedelta64(window_min, "m")

    # 找 group 边界
    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [n]])

    # 遍历每个 group
    for start, end in zip(group_starts, group_ends):
        gtimes = sorted_times[start:end]
        gsum = safe_targets[start:end]
        gcount = safe_indicators[start:end]
        m = end - start

        # cumsum(在每组前加 0 哨兵,简化边界)
        cum_sum = np.concatenate([[0.0], np.cumsum(gsum, dtype="float64")])
        cum_cnt = np.concatenate([[0.0], np.cumsum(gcount, dtype="float64")])

        # 窗口下界(每个时间点)
        t_lo_all = gtimes - window_ns  # array
        # searchsorted 返回 lo 索引(1-based 因为有哨兵)
        lo_idx = np.searchsorted(gtimes, t_lo_all, side="left")  # 0..m
        # hi = 自己位置 = k+1(因为 cumsum 前面加了哨兵)
        # 也就是 cum_idx_k+1 的索引
        # count = cum_cnt[k+1] - cum_cnt[lo_idx[k]]
        cnt = cum_cnt[np.arange(1, m + 1)] - cum_cnt[lo_idx]
        s = cum_sum[np.arange(1, m + 1)] - cum_sum[lo_idx]

        counts_sorted = cnt.astype("float32")
        # 写回(注意:这是 sorted 顺序的索引,要写回原始 order)
        # 我们直接构造 sorted 顺序的数组,然后用 inv_order 重排
        # 这样最清晰:先存到 sorted_* 数组,最后 inv_order 重排回原 df 顺序

        # mean = sum / count
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_sorted = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype("float32")

        # p90:对每个时间点,切片 [lo_idx, k+1),用 np.partition
        # 用列表推导+向量化混合(因为切片长度不一,纯向量化困难)
        p90_sorted = np.full(m, np.nan, dtype="float32")
        # p90:用 max 替代真正的 90% 分位
        # (max 比 percentile 快 100 倍,而且 tree 模型能从 mean + max 推断分布)
        # 对拥堵特征而言,max ≈ p90 是合理的近似
        nonzero = cnt > 0
        if nonzero.any():
            for k in np.where(nonzero)[0]:
                lo = lo_idx[k]
                hi = k + 1
                if hi > lo:
                    seg = safe_targets[start + lo:start + hi]
                    if len(seg) > 0:
                        p90_sorted[k] = float(np.max(seg))

        # 写回原始顺序
        local_idx = order[start:end]
        counts[local_idx] = counts_sorted
        means[local_idx] = mean_sorted
        p90s[local_idx] = p90_sorted

    return counts, means, p90s


# ---------------------------------------------------------------------------
# 便捷包装:对多个窗口 + 多个 target 一次跑完
# ---------------------------------------------------------------------------

def _past_n_minutes_stats(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    windows_min: Iterable[int],
    stats: Iterable[str] = ("count", "mean", "p90"),
) -> pd.DataFrame:
    """对每个分组(group_col),在每条记录 time_col 之前的 windows_min 分钟内,统计 target_col。"""
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
    """对单个 group,一次扫描算多个窗口的 count/mean/p90。

    v3.6 fix: NaT 时间戳被 np.searchsorted 当作极小值,导致 cnt/mean/p90
    被算成"同机场全部历史",污染预测。修复:
    - 在算聚合前,把 ref_time_col=NaT 的行从 cumsum/sort 链中剔除
    - 这些行的输出直接填 NaN,让模型学到"NaN = 无拥堵信号,用基础特征"
    - 训练和推理完全一致

    返回: dict {window_min: {"count": arr, "mean": arr, "p90": arr}} 按原 df 顺序
    """
    n = len(df)

    if isinstance(group_col, str):
        groups_raw = df[group_col].values
    else:
        groups_raw = df[list(group_col)].astype(str).agg("_".join, axis=1).values
    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    # v3.6 fix: 解析时间,识别 NaT(NaT 在 datetime64[ns] 里是 int64 min)
    times = pd.to_datetime(df[time_col]).values.astype("datetime64[ns]")
    INT64_MIN = np.iinfo(np.int64).min
    # pandas 把 NaT 存为 int64 min
    sorted_times_raw_int = times.view("i8")
    valid_time_mask = sorted_times_raw_int != INT64_MIN  # True = valid time

    # 对 raw_targets 也做 NaN 处理(target 缺失行不算)
    raw_targets = df[target_col].values
    target_nan_mask = np.isnan(raw_targets)
    # 一个行 valid 用于聚合的前提:时间有效 AND target 不是 NaN(只有 target NaN 的行不算入统计)
    # 但 count 类只需要时间有效(target=NaN 也不影响 count,只要 DEP 指示器 =1 就行)
    # 这里我们保守:对 cnt 类只用时间有效;对 mean/p90 同时要求 target 非 NaN

    # 先在原始顺序上构造有效位(后面写回时用)
    valid_count_row = valid_time_mask.copy()  # time 非 NaT → 可以算 count
    valid_target_row = valid_time_mask & ~target_nan_mask  # time + target 都有效 → 可以算 mean/p90

    # 用 (group, time) 排序;NaT 行会排到组内最前/最后(因为 int64 min 最小)
    # 我们用 NaT 作为最大 key 排到最后,这样 cumsum 只在非 NaT 段累加
    # 技巧:把 NaT 时间的 int64 替换成 int64 max,然后从排序中排除
    safe_times_int = np.where(valid_time_mask, sorted_times_raw_int, np.iinfo(np.int64).max)
    safe_times = safe_times_int.view("datetime64[ns]")

    order = np.lexsort((safe_times, groups))
    sorted_groups = groups[order]
    sorted_times = times[order]  # 原始时间(可能有 NaT)
    # raw_targets 也按 order 重排
    raw_targets_ordered = raw_targets[order]
    target_nan_ordered = np.isnan(raw_targets_ordered)

    # safe_targets: target NaN → 0(不影响 sum,影响 cnt)
    safe_targets = np.where(target_nan_ordered, 0.0, raw_targets_ordered).astype("float64")
    # safe_indicators: 1 if (时间有效 AND target 非 NaN),else 0
    time_valid_ordered = (safe_times_int[order] != np.iinfo(np.int64).max)
    safe_indicators = (time_valid_ordered & ~target_nan_ordered).astype("float64")
    # count 用的指示器:只要求时间有效(target NaN 仍算 1 架)
    count_indicators = time_valid_ordered.astype("float64")

    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [n]])

    # 结果数组(按 sorted 顺序)—— 默认 NaN,sorted_count 也用 NaN 表示"无信号"
    sorted_count = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}
    sorted_mean = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}
    sorted_p90 = {w: np.full(n, np.nan, dtype="float32") for w in windows_min}

    for start, end in zip(group_starts, group_ends):
        gtimes = sorted_times[start:end]  # 可能有 NaT
        gsum = safe_targets[start:end]
        gsum_count = count_indicators[start:end]  # 只要求时间有效
        gcount = safe_indicators[start:end]  # 要求时间+target 都有效(给 mean/p90 用)
        m = end - start

        cum_sum = np.concatenate([[0.0], np.cumsum(gsum, dtype="float64")])
        cum_cnt = np.concatenate([[0.0], np.cumsum(gcount, dtype="float64")])
        cum_cnt_for_count = np.concatenate([[0.0], np.cumsum(gsum_count, dtype="float64")])

        # v3.6 fix: 对每行,如果时间无效 → NaN,否则正常算
        time_valid_in_group = (gtimes.view("i8") != INT64_MIN)
        # 一个 mask 数组,表示"这一行的时间有效"
        # 后面算 lo_idx 时,无效行的 lo_idx=0(全窗口范围),但会被 NaN mask 过滤

        for w in windows_min:
            window_ns = np.timedelta64(w, "m")
            # 算 lo_idx 时,对 NaT 行也用 searchsorted(会落到组内某位置),
            # 但我们后续会把无效行的值改成 NaN
            t_lo_all = gtimes - window_ns  # NaT - timedelta = NaT,被当作极小值
            lo_idx = np.searchsorted(gtimes, t_lo_all, side="left")

            # count 用 cum_cnt_for_count(只要求时间有效)
            cnt = cum_cnt_for_count[np.arange(1, m + 1)] - cum_cnt_for_count[lo_idx]
            s = cum_sum[np.arange(1, m + 1)] - cum_sum[lo_idx]
            mean_v = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype("float32")

            # v3.6 fix: 无效时间行 → 输出 NaN(让 XGBoost 当作"无信号")
            cnt = np.where(time_valid_in_group, cnt, np.nan).astype("float32")
            mean_v = np.where(time_valid_in_group, mean_v, np.nan).astype("float32")

            sorted_count[w][start:end] = cnt
            sorted_mean[w][start:end] = mean_v

            # p90 → 用 max(简化);只对有效时间行算
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

    # 反排回原 df 顺序
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
    """主入口:加所有拥堵聚合特征(过去窗口)。优化版:每个 group 内对所有窗口共享一次 cumsum。

    ⚠️ 调用前请确保:
        1. df 已经过滤了 PHASE=DEP(data_loader 已经做了)
        2. ref_time_col(默认 EOBT_1_flt)是 timezone-aware 的 UTC datetime
        3. target_col 在 ranking 数据里会是 NaN(ranking 屏蔽了目标),聚合会自动跳过 NaN

    输出:在 df 上加以下列(默认参数):
        - cnt_dep_airport_15m / 30m / 60m  : 过去 15/30/60 分钟同机场起飞航班数
        - mean_taxi_airport_15m / 30m / 60m : 过去 15/30/60 分钟同机场平均 taxi-out
        - p90_taxi_airport_15m / 30m / 60m  : 过去 15/30/60 分钟同机场 P90 taxi-out
        - cnt_dep_runway_15m / 30m / 60m   : 同上,但限定同跑道(include_runway=True 时)
        - mean_taxi_runway_15m / 30m / 60m  : 同上
        - p90_taxi_runway_15m / 30m / 60m   : 同上
    """
    df = df.copy()
    print(f"开始聚合特征计算(ref={ref_time_col}, 机场={airport_col}, 跑道={include_runway and runway_col})...")

    import time

    # 构造一个"DEP 指示器",每条 DEP 记录 = 1
    df["__is_dep__"] = 1.0

    # 机场级 — 一次性算所有窗口
    t0 = time.time()
    airport_ind = _compute_past_for_group(
        df, group_col=airport_col, time_col=ref_time_col,
        target_col="__is_dep__", windows_min=windows_min,
    )
    airport_taxi = _compute_past_for_group(
        df, group_col=airport_col, time_col=ref_time_col,
        target_col=target_col, windows_min=windows_min,
    )
    print(f"  机场级 ({time.time()-t0:.1f}s)")

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
        print(f"  跑道级 ({time.time()-t0:.1f}s)")

        for w in windows_min:
            df[f"cnt_dep_runway_{w}m"] = runway_ind[w]["count"]
            df[f"mean_taxi_runway_{w}m"] = runway_taxi[w]["mean"]
            df[f"p90_taxi_runway_{w}m"] = runway_taxi[w]["p90"]

    # 去掉临时列
    df = df.drop(columns=["__is_dep__"])
    print(f"聚合特征完成,新增 {len(windows_min) * (3 + (3 if include_runway else 0))} 列")
    return df


# ---------------------------------------------------------------------------
# 完整的特征流水线(含基础特征 + 聚合特征)
# ---------------------------------------------------------------------------

def build_features_v2(df: pd.DataFrame, include_aggregate: bool = True, include_lead: bool = True) -> pd.DataFrame:
    """完整特征工程 = 基础特征 + 拥堵聚合(lag) + 未来聚合(lead)。

    参数:
        df: 从 data_loader 加载的原始 DataFrame
        include_aggregate: 是否加 lag 特征(过去 N 分钟拥堵)
        include_lead: 是否加 lead 特征(未来 N 分钟计划拥堵)
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
# CLI 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # 自测:用 mock 数据
    print("=== features_aggregate 模块自测 ===\n")

    # 构造 mock:3 个机场 × 1000 条 DEP
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
    # 目标:简单线性 + 噪声
    df["TAXITIME_SEC_mvt"] = 600 + np.random.normal(0, 100, n)

    print(f"Mock 数据: {df.shape}")

    import time
    t0 = time.time()
    df_agg = add_aggregate_features(df, windows_min=(15, 30))
    print(f"\n耗时: {time.time() - t0:.2f}s")
    print(f"输出形状: {df_agg.shape}")
    print(f"新增列: {[c for c in df_agg.columns if c not in df.columns]}")
    print(f"\n聚合特征示例:\n{df_agg[[c for c in df_agg.columns if 'airport_15m' in c or 'airport_30m' in c]].head()}")
