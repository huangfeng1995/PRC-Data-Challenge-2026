"""
阶段1 真实队列特征模块
=====================

定义(全部 leak-free,past-only,排除自身 i):
- 参考时点 ref_i = AOBT_3_flt 优先,缺失时回退 EOBT_1_flt
- 新增 ref_was_estimated 指示列(回退=1)
- 只处理 DEP 离港行(由 data_loader 过滤)

特征集:
A. dep_mvt_cnt_airport_{5,10,15}m: (ref_i − w, ref_i] 内,同机场其他 DEP 航班 MVT_j 落入窗口的数量
   = 真实离港吞吐(已经起飞的架次,反映跑道吞吐/队列消化速度)
B. dep_queue_n_airport: ref_i 时刻同机场"已 AOBT 但尚未 MVT"的其他 DEP 航班数
   = 真实离港队列长度
C. dep_queue_heavy_n_airport: B 中 WK_TBL_CAT ∈ {H, J} 的数量(尾流构成,重型机滑行慢)
D(对照,非主): 跑道级 dep_mvt_cnt_runway_* + dep_queue_n_runway
   - RUNWAY_mvt 可能是实际使用跑道(事后),存在轻微事后信息风险,默认不进入主特征
E(可选): arr_mvt_cnt_airport_{10,15}m: ref 前同机场 ARR 实际落地数(进港压力)

红线(全部满足):
- 自身 MVT/BLOCK/TAXITIME 永不可用
- 只统计其他航班(MVT_ID ≠ i)
- 窗口严格 < ref_i (dep_mvt_cnt 用 (ref_i - w, ref_i),即 MVT_j ∈ (ref_i - w, ref_i])
- 用其他航班 j 的 MVT_j / AOBT_j 重建 ref_i 时刻世界状态(合法历史重建)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 参考时点构造
# ---------------------------------------------------------------------------

def add_ref_time(df: pd.DataFrame) -> pd.DataFrame:
    """构造参考时点 ref_dt:
    - AOBT_3_flt 非缺失 → ref_dt = AOBT_3_flt(实际推出,优先级最高)
    - 否则 → ref_dt = EOBT_1_flt(计划估计,FPL),并 ref_was_estimated=1
    """
    df = df.copy()
    aobt = pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce")
    eobt = pd.to_datetime(df["EOBT_1_flt"], utc=True, errors="coerce")
    df["ref_was_estimated"] = aobt.isna().astype("int8")
    df["ref_dt"] = aobt.fillna(eobt)
    return df


# ---------------------------------------------------------------------------
# A. dep_mvt_cnt_airport_{w}m: 过去 w 分钟同机场 DEP 实际起飞数
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
    """对每个 ref_i,统计 (ref_i - w, ref_i] 窗口内同机场其他 DEP 航班的 MVT_j 数。

    排除:
    - 自身(MVT_ID ≠ i,若 exclude_self_col 提供)
    - NaT 时间戳行

    返回 DataFrame,index=df.index,每窗口一列。
    """
    n = len(df)
    airports = df[airport_col].astype(str).values
    mvt_times = pd.to_datetime(df[mvt_col], utc=True, errors="coerce").values.astype("datetime64[ns]")
    ref_times = pd.to_datetime(df[ref_col], utc=True, errors="coerce").values.astype("datetime64[ns]")
    INT64_MIN = np.iinfo(np.int64).min
    INT64_MAX = np.iinfo(np.int64).max
    mvt_valid = mvt_times.view("i8") != INT64_MIN
    ref_valid = ref_times.view("i8") != INT64_MIN

    # 排除自身(若有 MVT_ID)
    if exclude_self_col is not None and exclude_self_col in df.columns:
        own_mvt_id = df[exclude_self_col].astype(str).values
    else:
        own_mvt_id = None

    # 跑道分组(可选)
    if runway_col is not None and runway_col in df.columns:
        runways = df[runway_col].astype(str).values
        groups_raw = np.char.add(np.char.add(airports, "_"), runways)
    else:
        groups_raw = airports

    _, groups = np.unique(groups_raw, return_inverse=True)
    groups = groups.astype(np.int64)

    # 安全排序键(NaT → INT64_MAX 排到最后,不参与 cumsum)
    safe_mvt_int = np.where(mvt_valid, mvt_times.view("i8"), INT64_MAX)

    order = np.lexsort((safe_mvt_int, groups))
    sorted_groups = groups[order]
    sorted_mvt_valid = mvt_valid[order]
    sorted_own_mvt_id = own_mvt_id[order] if own_mvt_id is not None else None

    # cumsum 计数(每个排序位置上的"已 MVT 航班数")
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

        # 这个组里, sorted 后第 k 个位置对应的原始行 = order[start + k]
        # searchsorted 在 [start, end) 内做
        local_mvt_valid = sorted_mvt_valid[start:end]
        cum_local = np.concatenate([[0.0], np.cumsum(local_mvt_valid.astype("float64"))])  # 长度 m+1

        for w in windows_min:
            window_ns = np.timedelta64(w, "m")
            t_lo = ref_in_group - window_ns  # 窗口下界(> 这个)
            # 区间 (ref_i - w, ref_i] 半开半闭: lo = first > ref_i - w, hi = first > ref_i
            t_lo_safe = np.where(ref_valid_in_group, t_lo, np.datetime64("NaT"))
            lo_idx = np.searchsorted(
                mvt_times[order[start:end]],
                t_lo_safe,
                side="right",  # 严格 > ref_i - w
            )
            # hi = 第一个 MVT > ref_i,即 searchsorted(ref, side='right') - 1? 不,我们用 ≤ ref_i 的最后一个
            # 实际上:dep_mvt_cnt = cum_local[hi] - cum_local[lo],其中 hi 是第一个 MVT > ref_i 的索引
            t_hi_safe = np.where(ref_valid_in_group, ref_in_group, np.datetime64("NaT"))
            hi_idx = np.searchsorted(
                mvt_times[order[start:end]],
                t_hi_safe,
                side="right",  # 严格 > ref_i,故 hi 是第一个 > ref 的位置(不含 ref_i 自身)
            )
            cnt = cum_local[hi_idx] - cum_local[lo_idx]

            # 排除自身:若 own_in_group[k] == own_in_group[order[k]] ...
            # 简化:对每个 k,在 [lo_idx[k], hi_idx[k]) 范围内,如果 own_mvt_id[j] == own_in_group[k], 则 -1
            if own_in_group is not None:
                own_sub = own_in_group  # length m,own_mvt_id of each row in this group (sorted by mvt_time)
                cnt_adj = cnt.copy()
                for k in np.where(ref_valid_in_group & (cnt > 0))[0]:
                    lo, hi = int(lo_idx[k]), int(hi_idx[k])
                    if hi > lo:
                        # 检查 own_sub[lo:hi] 是否等于 own_in_group[k]
                        in_window = own_sub[lo:hi]
                        # 自己出现在窗口里 → -1(只有 MVT_i 真的在窗口中才会减,但用户红线禁止用 MVT_i)
                        # 由于我们用 MVT_ID 排除(不是 MVT_i),且 MVT_i 通常不在 (ref_i-w, ref_i] 窗口内(ref_i 就是自己起飞时刻前 w 分钟),
                        # MVT_i 大概率 > ref_i(还未起飞)或 << ref_i-w(早就起飞了)。少数情况 MVT_i 恰好落入 → 减 1。
                        if own_in_group[k] in in_window:
                            cnt_adj[k] -= 1.0
                cnt = cnt_adj

            # 无效 ref → NaN
            cnt = np.where(ref_valid_in_group, cnt, np.nan).astype("float32")
            # 写回原顺序
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
    """A 特征:dep_mvt_cnt_airport_{5,10,15}m(可选加跑道级)。"""
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
# B. dep_queue_n_airport: ref 时刻同机场"已 AOBT 但尚未 MVT"的其他 DEP 数
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
    """B/C 特征:ref 时刻同机场"已推出未起飞"队列长度。

    队列定义: AOBT_j ≤ ref_i AND (MVT_j NaT OR MVT_j > ref_i) AND MVT_ID_j ≠ i
    = #{j: AOBT_j ≤ ref_i} - #{j: AOBT_j ≤ ref_i AND MVT_j 有效 AND MVT_j ≤ ref_i} - self_excluded

    实现:对每个 group(机场/跑道),把 ref 也排序,逐 ref 扫描 + 二分定位。
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

    aobt_int_safe = np.where(aobt_valid, aobt_times.view("i8"), INT64_MAX)  # NaT → 最大,排在最后
    mvt_int_safe = np.where(mvt_valid, mvt_times.view("i8"), INT64_MAX)

    def _compute_for_group(group_arr, mask, key_label):
        """对每个 group,计算 queue_n, queue_heavy_n(全 numpy 向量化 sweep line)。
        """
        import time
        t_total0 = time.time()
        n_total = len(df)
        out_queue_n = np.full(n_total, np.nan, dtype="float32")
        out_queue_heavy = np.full(n_total, np.nan, dtype="float32")

        ref_safe = np.where(ref_valid, ref_times.view("i8"), INT64_MAX)

        # 把 group_arr 编码成 int(快速 group by)
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

            # ===== 全向量化 sweep line =====
            # events: 3 类 (aobt enter, mvt leave, query ref)
            # 用 np.argsort 一次性排序, 然后 cumsum
            n_aobt = int(aobt_v_g.sum())
            n_mvt = int(mvt_v_g.sum())
            n_query = int(ref_v_g.sum())

            times = np.concatenate([
                aobt_g[aobt_v_g],       # aobt 时刻
                mvt_g[mvt_v_g],          # mvt 时刻
                ref_g[ref_v_g],          # ref 时刻
            ])
            # kind: +1 (aobt), -1 (mvt), 0 (query)
            # 排序键: (time, -kind) 让 +1 先于 0 先于 -1
            kinds = np.concatenate([
                np.ones(n_aobt, dtype="int8"),     # +1
                -np.ones(n_mvt, dtype="int8"),     # -1
                np.zeros(n_query, dtype="int8"),   # 0
            ])
            # sort by (time, -kind) → asc
            sort_key = times.astype("int64") * 2 + (-kinds.astype("int64") + 1) * 1
            # 简化:用 lexsort
            # numpy 排序 aobt > query > mvt 的优先级通过 sort_key (kind 越正越先)
            # sort by time asc, kind desc: +1, 0, -1
            # 对每个事件,记 idx_in_g_idx (在 g_idx 内的行号)
            idx_in_g = np.concatenate([
                np.where(aobt_v_g)[0],
                np.where(mvt_v_g)[0],
                np.where(ref_v_g)[0],
            ])
            # 用 lexsort:primary = times, secondary = -kinds
            # np.lexsort([secondary, primary])  primary 是最后一个,先按 secondary 排
            # 不,np.lexsort 接受 [last_key, ..., first_key], primary 是 last_key(最后一列)
            # 我们要 primary = times, secondary = -kinds
            order = np.lexsort((-kinds, times))
            sorted_times = times[order]
            sorted_kinds = kinds[order]
            sorted_idx = idx_in_g[order]

            # cumsum (kind 已经是 +1/-1/0,直接累加)
            cum = np.cumsum(sorted_kinds.astype("int64"))
            cum_heavy = np.cumsum((sorted_kinds == 1) & heavy_g[sorted_idx]).astype("int64")

            # query 在 sorted 序列中的位置:
            # query kind=0 对应 sorted_kinds==0
            # query_idx_in_sorted = np.where(sorted_kinds == 0)[0]
            query_mask = sorted_kinds == 0
            query_positions = np.where(query_mask)[0]

            # 对每个 query, 队列 = cum[query_pos]
            # 但 cum[query_pos] = "query 时累计" = 到 query 时刻的 +1 累计 - -1 累计
            # = #{j: aobt_j ≤ ref AND mvt_j ≤ ref} + #{j: aobt_j ≤ ref AND mvt NaT} - #{j: mvt_j ≤ ref AND aobt NaT}
            # 等等,cum 的语义是:
            #   +1 事件: 在 aobt 时刻发生(到达队列)
            #   -1 事件: 在 mvt 时刻发生(离开队列)
            #   0 事件: query (不改变 cum)
            # 在 query 时刻,cum = (此时所有 +1 累计) - (此时所有 -1 累计)
            # +1 累计 = #{j: aobt_j ≤ ref}
            # -1 累计 = #{j: mvt_j ≤ ref}  (mvt_j 有效)
            # cum = #{aobt_j ≤ ref} - #{mvt_j ≤ ref}
            # = #{j: aobt_j ≤ ref AND (mvt NaT OR mvt_j > ref)} + #{j: aobt NaT AND mvt_j ≤ ref} (NaT 永远存在)
            # 我们要的是 #{j: aobt_j ≤ ref AND (mvt NaT OR mvt > ref)}
            # 所以要排除 "aobt NaT AND mvt ≤ ref" 的 j
            # 这些 j 的 aobt NaT → +1 事件不存在(我们只对 aobt_v_g 加事件)
            # 但 mvt 有效 + aobt NaT → -1 事件存在(因为 mvt_v_g)
            # 而 aobt NaT 不应该让 j 进入队列 — 这部分由 aobt_v_g mask 处理
            # 还要排除 aobt_j > ref 但 mvt_j ≤ ref(永远不会发生在 sort by ref 后 — wait)
            # 排序时按 time, +1 先于 query 先于 -1。所以 cum[query_pos] 包含 query 之前的所有 +1 和 -1
            # 即: cum[query] = #{j: aobt_j ≤ ref} - #{j: mvt_j ≤ ref}
            # 但若 mvt_j ≤ ref 但 aobt_j > ref(尚未推出,已经飞走)— 这种情况 cum[mvt] = -1,aobt = +0,被扣一个
            # 我们应该排除: aobt NaT 时 -1 仍然有效(因为我们 mvt_v_g mask 还在),但 aobt 没"进入"队列 → 这个 j 不该在队列里
            # 这对应 cum 减 1(因为 mvt -1 减了一个但 aobt 没加)
            # 用 self_in_queue 类似:对每个 j,如果 aobt NaT AND mvt ≤ ref → cum 多减了 1
            # 但 aobt NaT AND mvt ≤ ref 的 j 在队列定义下: aobt NaT → 不算"已 AOBT", 不在队列 ✓
            # 所以 cum 多减 1,要把这 1 加回来
            # 但这与 self_in_queue 不同 — self_in_queue 是排除"自己"在队列中
            # 总结:cum 是 #{aobt ≤ ref} - #{mvt ≤ ref}
            # 真正的队列 = #{aobt_v AND aobt ≤ ref AND (mvt NaT OR mvt > ref)} - self
            # = #{aobt ≤ ref AND mvt NaT} + #{aobt ≤ ref AND mvt > ref} - self
            # = #{aobt ≤ ref} - #{aobt ≤ ref AND mvt_v AND mvt ≤ ref} - self
            # 而 #{aobt ≤ ref AND mvt_v AND mvt ≤ ref} ≠ #{mvt ≤ ref}(后者含 aobt NaT 的 mvt)
            # 差 = #{aobt NaT AND mvt_v AND mvt ≤ ref}
            # 所以 队列 = cum + #{aobt NaT AND mvt_v AND mvt ≤ ref} - self
            # = cum + cnt_aobt_nat_mvt_le_ref - self

            # 现在算出 cnt_aobt_nat_mvt_le_ref:
            # 在 sorted 序列里,aobt NaT 不生成 +1 事件(已 mask),但 mvt NaT 不生成 -1 事件
            # 所以 aobt NaT 的 j 的 mvt -1 还在 — 这就是上面 "cum 多减 1" 的原因
            # 用 cum_aobt_nat_mvt_le = #{k: aobt NaT AND mvt_v AND mvt ≤ ref_k}
            # = 对每个 query,在 sorted 中 [0, query_pos) 范围内,有 mvt 事件(idx where kind=-1) 且 aobt 是 NaT
            # 向量化: 在 sorted 序列, query 之前的 -1 事件: kind=-1 的位置
            # 对每个 query_pos, 找 [0, query_pos) 内 -1 的 AND aobt_v_g[idx]=False
            # 用 cum count
            # 简单:对 query 之前的 -1 事件,查 idx_in_g 的 aobt_v_g == False 的个数
            # 用 prefix sums

            # 更简单:用 sweep 内的 (aobt_v AND kind=+1) vs (mvt_v AND kind=-1) 的对比
            # 但 aobt NaT 时 aobt_v=False,kind 不增加事件 — 这部分已被 mask
            # 所以对每个 j:
            #   aobt_v=T, mvt_v=T: +1 then -1,净 0,贡献 0 到 cum 但"aobt ≤ ref AND mvt ≤ ref"应该被减 — 这部分不准确
            #   aobt_v=T, mvt_v=F: 只 +1, 净 +1 ✓
            #   aobt_v=F, mvt_v=T: 只 -1, 净 -1 ✗ (错误扣除)
            #   aobt_v=F, mvt_v=F: 0 事件, 净 0 ✓

            # 对情况 (aobt_v=F, mvt_v=T): 这种 j 不该在队列里(没推出),但 cum 减了 1
            # 需要把这些 "phantom" -1 加回去
            # 等价于:队列 = cum + #{j: aobt_v=F AND mvt_v=T AND mvt ≤ ref} - self

            # 计算 phantom_mvt_le_ref:
            # 在 sorted 序列中,query 之前的 -1 事件,若 idx 的 aobt_v=False → phantom
            phantom_mask_in_sorted = (sorted_kinds == -1) & (~aobt_v_g[sorted_idx])
            # prefix sum: cum_phantom[query_pos] = #phantom up to (and including) this position
            cum_phantom = np.cumsum(phantom_mask_in_sorted.astype("int64"))
            # query 时, 队列 = cum[query] + cum_phantom[query] - self
            query_cum = cum[query_positions] + cum_phantom[query_positions]
            query_heavy_v = cum_heavy[query_positions]

            # 写回:但 sorted 是按 (time, -kind) 排,query 在 sorted 里的位置 query_positions
            # 但 query_idx_in_g 是 ref 在 g_idx 内的行号
            query_idx_in_g = sorted_idx[query_positions]

            # self_in_queue: 自己 (aobt_v AND aobt ≤ ref AND (mvt_v=F OR mvt > ref))
            # 用 aobt_g / mvt_g / ref_g 的向量比较
            self_in_queue = np.zeros(m, dtype=bool)
            self_heavy = np.zeros(m, dtype=bool)
            # 只对 ref_v 的行算
            valid_ref_pos = np.where(ref_v_g)[0]
            self_aobt_ok = aobt_v_g[valid_ref_pos] & (aobt_g[valid_ref_pos] <= ref_g[valid_ref_pos])
            self_mvt_after = (~mvt_v_g[valid_ref_pos]) | (mvt_g[valid_ref_pos] > ref_g[valid_ref_pos])
            self_in_queue[valid_ref_pos] = self_aobt_ok & self_mvt_after
            self_heavy[valid_ref_pos] = self_in_queue[valid_ref_pos] & heavy_g[valid_ref_pos]

            # 写回(全向量化):query_idx_in_g 是 ref 在 g_idx 内的行号
            # query_cum / query_heavy_v 与 query_idx_in_g 一一对应
            result_in_g = np.full(m, np.nan, dtype="float32")
            heavy_in_g = np.full(m, np.nan, dtype="float32")
            if len(query_idx_in_g) > 0:
                base = query_cum.astype("float32")
                base_h = query_heavy_v.astype("float32")
                # 排除自己(self_in_queue at query_idx_in_g)
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

    # 机场级
    mask = np.ones(len(df), dtype=bool)
    queue_n_airport, queue_heavy_n_airport = _compute_for_group(airports, mask, "airport")
    df["dep_queue_n_airport"] = queue_n_airport
    df["dep_queue_heavy_n_airport"] = queue_heavy_n_airport

    # 跑道级
    if include_runway:
        print(f"  [D] dep_queue_n_runway (对照)...", flush=True)
        runway_str = df[runway_col].astype(str).values
        runway_groups = np.char.add(np.char.add(airports, "_"), runway_str)
        queue_n_runway, _ = _compute_for_group(runway_groups, mask, "runway")
        df["dep_queue_n_runway"] = queue_n_runway

    return df


# ---------------------------------------------------------------------------
# E. arr_mvt_cnt_airport_{10,15}m: ref 前同机场 ARR 实际落地数
# ---------------------------------------------------------------------------

def add_arr_mvt_cnt(
    df: pd.DataFrame,
    ref_col: str = "ref_dt",
    airport_col: str = "ADEP_mvt",
    mvt_col: str = "MVT_TIME_UTC_mvt",
    phase_col: str = "PHASE_mvt",
    windows_min: tuple[int, ...] = (10, 15),
) -> pd.DataFrame:
    """E 特征:ref 前同机场 ARR 实际落地数(进港压力)。

    输入 df 既含 DEP 也含 ARR(由 data_loader 控制);只统计 ARR 行。
    """
    df = df.copy()
    print(f"  [E] arr_mvt_cnt_airport w={windows_min} (PHASE=ARR)...")

    is_arr = df[phase_col].astype(str).values == "ARR"
    # ARR 的 MVT_TIME = 实际落地时间(进港完成)
    # DEP 也用同一字段(MVT_TIME = 实际起飞时间);但我们只看 PHASE=ARR 的 MVT
    # 关键:DEP/ARR 行的 ADEP_mvt 不一样(ARR 行 ADEP_mvt 是原起飞机场)
    # 这里我们想看"同机场":从 DEP 视角, ARR 的 ADES_mvt 等于 DEP 的 ADEP_mvt 才是同一机场
    # 用 ARR 行的 ADES_mvt(进港机场) = DEP 的 ADEP_mvt
    arr_airport = df.get("ADES_mvt", df.get("ADEP_mvt")).astype(str).values
    arr_mvt = df[mvt_col].values

    # 构造一个 arr-only 子集
    out = {f"{w}m": np.full(len(df), np.nan, dtype="float32") for w in windows_min}

    # 只对 DEP 行计算(因为这是 DEP 的特征),ARR 行设为 NaN
    dep_mask = df[phase_col].astype(str).values == "DEP"
    if dep_mask.sum() == 0:
        for w in windows_min:
            df[f"arr_mvt_cnt_airport_{w}m"] = out[f"{w}m"]
        return df

    # 按机场分组,用 ARR 行的 MVT 时间
    INT64_MIN = np.iinfo(np.int64).min
    arr_mvt_int = pd.to_datetime(arr_mvt, utc=True, errors="coerce").values.astype("datetime64[ns]").view("i8")
    arr_valid = arr_mvt_int != INT64_MIN
    safe_arr_mvt_int = np.where(arr_valid, arr_mvt_int, np.iinfo(np.int64).max)

    # 按机场(对 DEP 是 ADEP_mvt,对 ARR 是 ADES_mvt)
    dep_airport = df[airport_col].astype(str).values
    airport_all = np.where(is_arr, arr_airport, dep_airport)
    _, groups = np.unique(airport_all, return_inverse=True)
    groups = groups.astype(np.int64)

    # 只算 ARR 行(DEP 行不进 cumsum)
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
            # 区间 (ref-w, ref] 半开半闭: lo = first > ref-w, hi = first > ref
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
# 主入口
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
    """主入口:加阶段1全部队列特征。

    ⚠️ 依赖:
    - df 必须 PHASE=DEP 已过滤(若 include_e=False),或同时含 DEP+ARR(若 include_e=True)
    - 列: MVT_TIME_UTC_mvt, AOBT_3_flt, EOBT_1_flt, ADEP_mvt, RUNWAY_mvt, WK_TBL_CAT_flt, MVT_ID_mvt
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
