"""
mvtseq_feats.py — MVT-only 实时序列特征构造 (3 个, 训练/推理两侧都可算)

背景
----
ranking.parquet 的 TAXITIME_SEC_mvt 344,841 行全空 (答案字段被屏蔽),
因此任何依赖"前序航班真实滑行时间"的特征在推理时算不出来。
本模块只构造**仅依赖 MVT** 的特征, 训练折和 ranking 折口径完全一致。

3 个特征
--------
  seq_rwy_gap_mean_prev3    同 (机场,跑道,日) 内, 按 MVT 排序, 前 3 架起飞间隔均值 (秒)
  seq_rwy_gap_mean_prev10   同上, 前 10 架
  seq_ap_dep_cnt_15min     同 (机场,日) 内, [MVT-900s, MVT) 实际起飞架次

关键口径 (与用户确认)
--------------------
  * 排序键: (ADEP_mvt, RUNWAY_mvt, date, MVT_TIME) 四元组, 严格按天切, 不跨天
  * 窗口不足: prev<2 → NaN (LGB 原生支持缺失; 不填 0, 不用 1 架均值)
  * **同秒并列的处理 (关键)**: MVT 只精确到秒, 同一秒常有多架
    (实测 EDDF 单日 15496 架仅 1871 个不同秒值, 平均 8.3 架/秒)。
    因此"前序"= 排序位置严格在前的行, **不排除同秒的其它架**;
    排序时以原行号 idx 作最终 tie-break, 保证确定性且不含自身。
    若改成"仅取 MVT 严格更小", 会把同秒的 8 架全部丢掉, 窗口系统性偏空。
  * 只读: 不改 label, 不 clip 真值
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NS = 1_000_000_000
SEQ_FEATS = ["seq_rwy_gap_mean_prev3", "seq_rwy_gap_mean_prev10", "seq_ap_dep_cnt_15min"]
MIN_PREV = 2          # prev < 2 → NaN
GAP_WINDOWS = (3, 10)
AP_WINDOW_S = 900     # 15 分钟


def _to_ns(s: pd.Series) -> pd.Series:
    """datetime -> int64 纳秒 (显式 ns, 避免 us 精度陷阱)。NaT → iNaT。"""
    v = pd.to_datetime(s, utc=True, errors="coerce")
    v = v.dt.tz_convert("UTC").dt.tz_localize(None)
    return v.astype("datetime64[ns]")


def build_seq_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """输入原始 DEP 帧 (需含 MVT_TIME_UTC_mvt / ADEP_mvt / RUNWAY_mvt),
    返回 index 与 df_raw 对齐的 3 列特征。"""
    n = len(df_raw)
    out = {c: np.full(n, np.nan, dtype=np.float32) for c in SEQ_FEATS}

    mvt_ns = _to_ns(df_raw["MVT_TIME_UTC_mvt"]).to_numpy(np.int64)
    iNaT = np.iinfo(np.int64).min
    valid = mvt_ns != iNaT
    if not valid.any():
        return pd.DataFrame(out, index=df_raw.index)

    ap = df_raw["ADEP_mvt"].astype(str).to_numpy()
    rw = df_raw["RUNWAY_mvt"].astype(str).to_numpy()
    # 日期 key: 用 ns 直接整除日秒数, 避免 tz/date 边界问题
    day = np.where(valid, mvt_ns // (86400 * NS), 0)
    # np.lexsort 不能可靠比较 object 类型的字符串数组 (会退化为指针比较,
    # 实测把跑道 18 的行排到 07C 之后, 导致负间隔)。这里先做整数编码。
    ap_c, ap_u = pd.factorize(ap, sort=True)
    rw_c, rw_u = pd.factorize(rw, sort=True)

    # ---------- A) 同跑道前 N 架起飞间隔 ----------
    # 目标排序: 先按 (日, 机场, 跑道) 分块, 块内按 MVT 升序。
    # np.lexsort 的 key 顺序是**从低优先到高优先** (最后一个 key 是主键),
    # 因此写作 (tie, MVT, 跑道, 机场, 日): MVT 先比, 再跑道、机场、日。
    # 附加 idx 作最终 tie-break, 保证同时刻 MVT 顺序确定且不含自身。
    # 机场/跑道必须用整数编码 (见上方 factorize 注释)。
    idx_rw = np.where(valid)[0]
    order_rw = np.lexsort((idx_rw, mvt_ns[idx_rw], rw_c[idx_rw], ap_c[idx_rw], day[idx_rw]))
    srt = idx_rw[order_rw]
    t_sorted = mvt_ns[srt]
    d_sorted = day[srt]
    # 分组边界必须用**与 lexsort 相同的整数编码**判断, 否则字符串比较与
    # 编码顺序不一致 (实测 rw '18' vs '07C': 编码 3 != 18 但字符串比较也会 !=,
    # 真正的问题是排序按编码、而 new_grp 按字符串, 两者边界可能落在不同位置)。
    rw_sorted_c = rw_c[srt]
    ap_sorted_c = ap_c[srt]
    new_grp = np.ones(len(srt), bool)
    if len(srt) > 1:
        new_grp[1:] = ((d_sorted[1:] != d_sorted[:-1]) |
                       (ap_sorted_c[1:] != ap_sorted_c[:-1]) |
                       (rw_sorted_c[1:] != rw_sorted_c[:-1]))
    grp_id = np.cumsum(new_grp) - 1
    n_grp = int(grp_id[-1]) + 1

    # 自检: 排序必须是 (日,机场,跑道) 分块 + 块内 MVT 非降。
    # 注意: 不是**全局** MVT 非降 —— 不同天的 MVT 当然可以回退。
    if len(srt) > 1:
        within = ~new_grp[1:]                      # 与前一行同组的位置
        assert np.all(np.diff(t_sorted)[within] >= 0), "组内 MVT 非升序: lexsort 主键顺序有误"
        # 相同 (日,机场,跑道) 的行不能被切成两块
        same_key = ((ap_sorted_c[1:] == ap_sorted_c[:-1]) &
                    (rw_sorted_c[1:] == rw_sorted_c[:-1]) &
                    (d_sorted[1:] == d_sorted[:-1]))
        assert not np.any(new_grp[1:] & same_key), "相同 (日,机场,跑道) 被错误切断"

    # 每组内 rolling mean of gap (纯 numpy, 不用 pandas groupby-roll:
    # pandas 的 shift/rolling 会跨组边界串值, 组首元素拿到上一组末尾的 gap)。
    #
    # 定义: 组内排序后位置 p。gap[p] = t[p] - t[p-1] (p>=1), gap[0] 无定义。
    # "前 K 架的起飞间隔均值" = 最近 K 个连续间隔的均值 = mean(gap[p-K+1 .. p])
    #                        = (t[p] - t[p-K]) / K, 需要 p-K >= 0。
    # prev<MIN_PREV → NaN。
    ts_f = t_sorted.astype(np.float64) / NS
    p = np.arange(len(ts_f))
    # 组内间隔必须非负 (排序正确性的硬校验: 负值 = 分组被打散)
    if len(srt) > 1:
        dts = np.diff(ts_f)
        assert np.all(dts[~new_grp[1:]] >= 0), "出现负间隔: lexsort 分组被破坏"
    # 组内位置 (块首 = 0)。np.maximum.accumulate 把每个位置映射到"最近块首的位置",
    # 再与当前位置相减即得组内偏移。不要用 np.where(new_grp, p, 0) —— 那样
    # 非块首位置会保留原值, 算出的是绝对位置而非组内偏移。
    grp_start = np.maximum.accumulate(np.where(new_grp, p, 0))
    grp_pos = p - grp_start
    # csum_grp[j] = new_grp[:j].sum(); 用于判断区间 [back, p) 内是否跨越了组边界
    csum_grp = np.concatenate([[0], np.cumsum(new_grp.astype(np.int64))])
    for k in GAP_WINDOWS:
        back = p - k                       # 回溯 K 个间隔
        b = np.clip(back, 0, None)
        # 窗口完整的充要条件:
        #   (1) back >= 0            —— 回溯目标存在
        #   (2) 区间 [b, p) 内无新组 —— 窗口不跨 (日,机场,跑道) 边界
        #   (3) grp_pos >= MIN_PREV  —— 前序架数 >= MIN_PREV
        # grp_pos = 组内序号(含自身), 故前序架数 = grp_pos, 首个间隔需要 grp_pos>=1,
        # 再叠加 MIN_PREV>=2 的要求 → grp_pos >= MIN_PREV。
        # 跨界判定: 窗口覆盖位置 [back, p-1] 的 K 个间隔, 要求这些位置同组。
        # 即 new_grp[b+1 .. p] 全为 False (new_grp[p] 是 p 自身的组首标记,
        # p 已在当前组内故不参与判断)。
        crossed = (csum_grp[p + 1] - csum_grp[b + 1]) > 0
        ok = (back >= 0) & (~crossed) & (grp_pos >= MIN_PREV)
        vals = np.where(ok, (ts_f - ts_f[b]) / k, np.nan)
        assert np.all(np.isnan(vals) | (vals >= 0)), "出现负间隔: 窗口跨越了组边界"
        out[f"seq_rwy_gap_mean_prev{k}"][srt] = vals

    # ---------- B) 同机场过去 15min 实际起飞架次 ----------
    # 同 A 段: lexsort 最后一列为主键 → (tie, MVT, 机场, 日), 机场用整数编码
    idx_ap = np.where(valid)[0]
    order_ap = np.lexsort((idx_ap, mvt_ns[idx_ap], ap_c[idx_ap], day[idx_ap]))
    sa = idx_ap[order_ap]
    t_a = mvt_ns[sa]
    d_a = day[sa]
    ap_a_c = ap_c[sa]                      # 与 lexsort 一致的整数编码
    newg = np.ones(len(sa), bool)
    if len(sa) > 1:
        newg[1:] = (d_a[1:] != d_a[:-1]) | (ap_a_c[1:] != ap_a_c[:-1])
        assert np.all(np.diff(t_a)[~newg[1:]] >= 0), "同机场窗口内 MVT 非升序"
    gid = np.cumsum(newg) - 1
    # 滑窗计数: 同 (机场,日) 内, 与当前 MVT 相差 <= AP_WINDOW_S 的**前序**架数。
    # 窗口左开右闭 (t-W, t): newg[i] 为组首时 start=i, 天然不跨组。
    cnt = np.zeros(len(sa), dtype=np.float64)
    start = 0
    for i in range(len(sa)):
        if newg[i]:
            start = i
        lo = t_a[i] - AP_WINDOW_S * NS
        while start < i and t_a[start] <= lo:   # 保留差值 > W 的, 即窗口 (t-W, t)
            start += 1
        cnt[i] = i - start                        # 不含自身 (位置 < i)
    out["seq_ap_dep_cnt_15min"][sa] = cnt

    return pd.DataFrame(out, index=df_raw.index)
