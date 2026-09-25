"""
批次 1: 队列特征 (feature-factory 批次 1)
==========================================

口径约束:
  - 参考时点 ref_i = AOBT_3_flt if notna else EOBT_1_flt  (与 features_queue.py 一致, 不锚 BLOCK_TIME)
  - 窗口严格向过去: (ref_i - w, ref_i] 左开右闭, 不包含未来
  - 排除自身: 队列/计数都跳过 MVT_ID_mvt 相同行
  - 机场隔离: DEP 用 ADEP_mvt; ARR 用 ADES_mvt
  - df 需含 MVT_ID_mvt / AOBT_3_flt / EOBT_1_flt / MVT_TIME_UTC_mvt / ADEP_mvt / RUNWAY_mvt
  - datetime 列以 UTC 解析

产出列 (全部 float32, NaN 表示"无历史数据 / 该时刻没有同组航班"):
  mu_rwy_{10,20,30,60}          — ref_i 时刻, 同跑道 DEP 队列的 TAXITIME_SEC_mvt 均值
  n_queue_rwy_{15,30,60}        — 过去 w 分钟, 同跑道 ARR 活跃航班数 (ARR 在跑道上占用)
  n_dep_queue_{15,30,60}        — 过去 w 分钟, 同机场已 AOBT 未 MVT 的 DEP 数 (窗口累积)
  rwy_arr_active                — ref_i 时刻, 同跑道 ARR 活跃航班数 (刚落地 / 待离场, 单列 G4 风险)

业务假设 & 预期方向 (待 Mac 全量 add-one 验证):
  mu_rwy_*         → "跑道排队长 / 重机多 → DEP taxi-out 慢", 预期正 (TAXITIME_SEC_mvt ↑)
  n_queue_rwy_*    → "跑道 ARR 频繁 → 跑道被占用, DEP 被迫等待", 预期正
  n_dep_queue_*    → "同机场 DEP 积压 → 滑行排队长", 预期正
  rwy_arr_active   → "G4 风险: 跑道 ARR 占用高 → DEP 被迫绕滑 / 等待脱离", 预期正
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_ref_time(df: pd.DataFrame) -> pd.Series:
    """ref_i = AOBT_3_flt if notna else EOBT_1_flt, UTC datetime64[ns]."""
    aobt = pd.to_datetime(df.get("AOBT_3_flt"), utc=True, errors="coerce")
    eobt = pd.to_datetime(df.get("EOBT_1_flt"), utc=True, errors="coerce")
    out = aobt.where(aobt.notna(), eobt)
    return out


def _safe_sortby_group(df: pd.DataFrame, group_col: str, time_col: str) -> pd.DataFrame:
    """按 group_col 分组, 组内按 time_col 升序排序, 返回索引."""
    tmp = df[[group_col, time_col]].copy()
    tmp["_gid"] = tmp[group_col].astype("category").cat.codes
    tmp = tmp.sort_values(["_gid", time_col], na_position="last")
    return tmp.index


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def add_batch1_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """对 df 追加批次 1 的 13 个队列特征列, 返回副本."""
    out = df.copy()

    # 准备 ref_i
    out["_ref_i"] = _parse_ref_time(out)
    out["_MVT"] = pd.to_datetime(out["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")

    # TAXITIME_SEC_mvt (有的数据没有, 用 BLOCK - MVT 兜底)
    if "TAXITIME_SEC_mvt" not in out.columns:
        block = pd.to_datetime(out.get("BLOCK_TIME_UTC_mvt"), utc=True, errors="coerce")
        mvt = out["_MVT"]
        out["_taxi_sec"] = (mvt - block).dt.total_seconds().clip(lower=0).astype("float32")
    else:
        out["_taxi_sec"] = out["TAXITIME_SEC_mvt"].astype("float32")

    # --- DEP 侧特征 ---
    # 我们只对有 DEP 的航班行算. df_sample 里 10 机场 DEP 全有, arr 是另一批
    dep_mask = out["MVT_TIME_UTC_mvt"].notna() & out["ADEP_mvt"].notna()

    # 1. mu_rwy_{10,20,30,60}: ref_i 时刻, 同跑道 DEP 队列的 taxi 均值 (瞬时, 窗口表示"ref_i 前多少分钟内排队的")
    out = _add_mu_rwy(out, dep_mask, [10, 20, 30, 60])

    # 2. n_dep_queue_{15,30,60}: 过去窗口内同机场 DEP 队列累积
    out = _add_n_dep_queue(out, dep_mask, [15, 30, 60])

    # --- ARR 侧特征 (跑道占用) ---
    arr_ok = (
        "ARVT_3_flt" in out.columns and "ADES_mvt" in out.columns
        and out["ARVT_3_flt"].notna().any() and out["ADES_mvt"].notna().any()
    )
    if arr_ok:
        arr_mask = out["ARVT_3_flt"].notna() & out["ADES_mvt"].notna()
        out = _add_n_queue_rwy_arr(out, arr_mask, [15, 30, 60])
        out = _add_rwy_arr_active(out, arr_mask)
    else:
        for w in [15, 30, 60]:
            out[f"n_queue_rwy_{w}"] = 0.0
        out["rwy_arr_active"] = 0.0

    # 清理临时列
    out.drop(columns=["_ref_i", "_MVT", "_taxi_sec"], inplace=True, errors="ignore")
    return out


# ---------------------------------------------------------------------------
# 1. mu_rwy_{10,20,30,60} —— ref_i 时刻同跑道 DEP 队列 taxi 均值
# ---------------------------------------------------------------------------
def _add_mu_rwy(df, dep_mask, windows_min):
    """对每条 DEP 行 i, ref_i 时刻同 runway 上 '已 AOBT 但尚未 MVT, 且 AOBT_j > ref_i - w' 的其他 DEP 的 _taxi_sec 均值.

    注意: 瞬时队列 (ref_i 时刻), 窗口 w 表示"纳入队列的其他航班 j 的 AOBT_j 不能早于 ref_i - w".
          避免 w 过大把早就推走的航班也拉进来.
    """
    df = df.copy()
    idx_all = df.index[dep_mask]
    df_dep = df.loc[idx_all].copy()

    runways = df_dep["RUNWAY_mvt"].astype(str).values
    ref_i = df_dep["_ref_i"].values  # datetime64[ns]
    aobt_j = pd.to_datetime(df_dep["AOBT_3_flt"], utc=True, errors="coerce").values
    mvt_j = df_dep["_MVT"].values
    taxi_j = df_dep["_taxi_sec"].values
    mvt_id = df_dep["MVT_ID_mvt"].astype(str).values

    for w in windows_min:
        w_ns = np.timedelta64(w * 60, "s")
        out_col = f"mu_rwy_{w}"
        result = np.full(len(df), np.nan, dtype="float32")

        for rwy in np.unique(runways):
            grp_mask = runways == rwy
            gi = np.where(grp_mask)[0]  # df_dep 内位置
            if len(gi) == 0:
                continue
            # sort by ref_i ascending
            order = np.argsort(ref_i[gi], kind="stable")
            gi_sorted = gi[order]

            # for each i in group: find all j in group where
            #   AOBT_j <= ref_i (已推出)
            #   MVT_j > ref_i 或 MVT_j NaT (尚未起飞)
            #   AOBT_j > ref_i - w
            #   MVT_ID_j != MVT_ID_i
            for k in range(len(gi_sorted)):
                i_pos = gi_sorted[k]
                ref_this = ref_i[i_pos]
                if pd.isna(ref_this):
                    continue
                t_lo = ref_this - w_ns
                ids = []
                for j_pos in gi_sorted:
                    if j_pos == i_pos:
                        continue
                    if pd.isna(aobt_j[j_pos]) or aobt_j[j_pos] > ref_this:
                        continue  # 未推出 / 推得比 ref 晚
                    if not pd.isna(mvt_j[j_pos]) and mvt_j[j_pos] <= ref_this:
                        continue  # 已起飞 (已离开跑道)
                    if aobt_j[j_pos] <= t_lo:
                        continue  # 推得太早, 不在窗口内
                    if mvt_id[j_pos] == mvt_id[i_pos]:
                        continue
                    ids.append(j_pos)
                if ids:
                    result[idx_all[i_pos]] = float(np.nanmean(taxi_j[ids]))

        df[out_col] = result
        print(f"  mu_rwy_{w}: nn={np.sum(~np.isnan(result))}/{len(result)}, 均值={np.nanmean(result):.1f}", file=sys.stderr if "sys" in dir() else None)

    return df


# ---------------------------------------------------------------------------
# 2. n_dep_queue_{15,30,60} —— 过去 w 分钟同机场 DEP 已 AOBT 未 MVT 累积
# ---------------------------------------------------------------------------
def _add_n_dep_queue(df, dep_mask, windows_min):
    """每条 DEP i, 过去 w 分钟内 (ref_i - w, ref_i] 同机场已 AOBT 未 MVT 的 DEP 数量, 排除自身."""
    df = df.copy()
    idx_all = df.index[dep_mask]
    df_dep = df.loc[idx_all].copy()

    airports = df_dep["ADEP_mvt"].astype(str).values
    ref_i = df_dep["_ref_i"].values
    aobt_j = pd.to_datetime(df_dep["AOBT_3_flt"], utc=True, errors="coerce").values
    mvt_j = df_dep["_MVT"].values
    mvt_id = df_dep["MVT_ID_mvt"].astype(str).values

    for w in windows_min:
        w_ns = np.timedelta64(w * 60, "s")
        out_col = f"n_dep_queue_{w}"
        result = np.zeros(len(df), dtype="float32")
        result[~dep_mask] = np.nan

        for ap in np.unique(airports):
            gi = np.where(airports == ap)[0]
            if len(gi) == 0:
                continue
            order = np.argsort(ref_i[gi], kind="stable")
            gi_sorted = gi[order]

            # sweep line: 所有 AOBT_j 在 (ref_i - w, ref_i] 内, 且 MVT_j 还没发生 (MVT_j > ref_i 或 NaT)
            for k in range(len(gi_sorted)):
                i_pos = gi_sorted[k]
                ref_this = ref_i[i_pos]
                if pd.isna(ref_this):
                    result[idx_all[i_pos]] = np.nan
                    continue
                t_lo = ref_this - w_ns
                cnt = 0
                for j_pos in gi_sorted:
                    if j_pos == i_pos:
                        continue
                    a = aobt_j[j_pos]
                    if pd.isna(a) or a <= t_lo or a > ref_this:
                        continue
                    m = mvt_j[j_pos]
                    if not pd.isna(m) and m <= ref_this:
                        continue
                    if mvt_id[j_pos] == mvt_id[i_pos]:
                        continue
                    cnt += 1
                result[idx_all[i_pos]] = cnt

        df[out_col] = result
        print(f"  n_dep_queue_{w}: nn={np.sum(~np.isnan(result))}/{len(result)}, mean={np.nanmean(result):.2f}")

    return df


# ---------------------------------------------------------------------------
# 3. n_queue_rwy_{15,30,60} —— 过去 w 分钟同跑道 ARR 活跃航班数
# ---------------------------------------------------------------------------
def _add_n_queue_rwy_arr(df, arr_mask, windows_min):
    """每条 ARR i, 过去 w 分钟内 (ref_i - w, ref_i] 同跑道 ARVT 落在窗口内的 ARR 数 (ARR 在跑道上).

    ARR 活跃定义: ARVT_j ∈ (ref_i - w, ref_i], 即"过去 w 分钟刚落地的 ARR".
    对 DEP 行 (dep_mask=True), 取同跑道、过去 w 分钟 ARR 活跃数作为该 DEP 起飞前的跑道压力.
    """
    df = df.copy()

    # ARR 基线: ARR 行的 ARVT_3_flt, 按 (ADES_mvt / RUNWAY_mvt) 分组
    idx_arr = df.index[arr_mask]
    df_arr = df.loc[idx_arr].copy()
    arr_runways = df_arr["RUNWAY_mvt"].astype(str).values
    arr_arvt = pd.to_datetime(df_arr["ARVT_3_flt"], utc=True, errors="coerce").values
    arr_id = df_arr["MVT_ID_mvt"].astype(str).values

    for w in windows_min:
        w_ns = np.timedelta64(w * 60, "s")
        out_col = f"n_queue_rwy_{w}"

        # 对整个 df (所有行), 每条行 i 看自己的跑道 RUNWAY_mvt_i + ref_i, 统计窗口内 ARR 活跃数
        all_runways = df["RUNWAY_mvt"].astype(str).values
        all_ref = df["_ref_i"].values
        all_id = df["MVT_ID_mvt"].astype(str).values

        result = np.zeros(len(df), dtype="float32")
        result[df["RUNWAY_mvt"].isna()] = np.nan

        # 按跑道分组
        unique_rwys = np.unique(all_runways)
        for rwy in unique_rwys:
            if rwy == "nan":
                continue
            # 该跑道上所有 ARR 落地时间
            arr_here_mask = arr_runways == rwy
            if not np.any(arr_here_mask):
                continue
            arr_arvt_here = arr_arvt[arr_here_mask]
            arr_id_here = arr_id[arr_here_mask]
            arr_valid_mask = ~pd.isna(arr_arvt_here)
            arr_arvt_valid = arr_arvt_here[arr_valid_mask]
            arr_id_valid = arr_id_here[arr_valid_mask]

            # 该跑道上所有行 (DEP + ARR), 用 searchsorted 统计
            row_idx_here = np.where(all_runways == rwy)[0]
            ref_here = all_ref[row_idx_here]
            order = np.argsort(ref_here, kind="stable")
            for k in order:
                global_i = row_idx_here[k]
                ref_this = ref_here[k]
                if pd.isna(ref_this):
                    result[global_i] = np.nan
                    continue
                t_lo = ref_this - w_ns
                # 二分: 找 arr_arvt_valid 在 (t_lo, ref_this] 的数
                left = np.searchsorted(arr_arvt_valid, t_lo, side="right")
                right = np.searchsorted(arr_arvt_valid, ref_this, side="right")
                cnt = int(right - left)
                # 排除自己 (如果这一行是 ARR, 且自己的 ARVT 落在窗口)
                my_id = all_id[global_i]
                # arr_id_valid[left:right] 是候选; 逐个减 1 (最坏 O(cnt), cnt 通常很小)
                for arr_j in range(left, right):
                    if arr_id_valid[arr_j] == my_id:
                        cnt -= 1
                        break
                result[global_i] = cnt

        df[out_col] = result
        print(f"  n_queue_rwy_{w}: nn={np.sum(~np.isnan(result))}/{len(result)}, mean={np.nanmean(result):.2f}")

    return df


# ---------------------------------------------------------------------------
# 4. rwy_arr_active (单列 G4 风险) —— ref_i 时刻同跑道 ARR 活跃数
# ---------------------------------------------------------------------------
def _add_rwy_arr_active(df, arr_mask):
    """ref_i 时刻, 同跑道上 '已落地 ARVT ≤ ref_i 但还没脱离跑道' 的 ARR 数量.

    活跃定义: 刚落地 ≤ 2 min (ref_i - 120s, ref_i], 或已落地但 LOBT - ARVT ≥ 120s 仍在跑道.
    简化版: 过去 180s 内 ARVT 的 ARR 数 (跑道占用释放通常 < 3 min).
    """
    df = df.copy()
    idx_arr = df.index[arr_mask]
    df_arr = df.loc[idx_arr].copy()
    arr_runways = df_arr["RUNWAY_mvt"].astype(str).values
    arr_arvt = pd.to_datetime(df_arr["ARVT_3_flt"], utc=True, errors="coerce").values
    arr_id = df_arr["MVT_ID_mvt"].astype(str).values

    all_runways = df["RUNWAY_mvt"].astype(str).values
    all_ref = df["_ref_i"].values
    all_id = df["MVT_ID_mvt"].astype(str).values

    W = np.timedelta64(180, "s")  # 180s 跑道占用窗口
    result = np.zeros(len(df), dtype="float32")
    result[df["RUNWAY_mvt"].isna()] = np.nan

    for rwy in np.unique(all_runways):
        if rwy == "nan":
            continue
        arr_here_mask = arr_runways == rwy
        if not np.any(arr_here_mask):
            continue
        arr_arvt_here = arr_arvt[arr_here_mask]
        arr_id_here = arr_id[arr_here_mask]
        arr_valid = ~pd.isna(arr_arvt_here)
        arr_arvt_valid = arr_arvt_here[arr_valid]
        arr_id_valid = arr_id_here[arr_valid]

        row_idx = np.where(all_runways == rwy)[0]
        ref_here = all_ref[row_idx]
        order = np.argsort(ref_here, kind="stable")
        for k in order:
            gi = row_idx[k]
            ref_this = ref_here[k]
            if pd.isna(ref_this):
                result[gi] = np.nan
                continue
            t_lo = ref_this - W
            left = np.searchsorted(arr_arvt_valid, t_lo, side="right")
            right = np.searchsorted(arr_arvt_valid, ref_this, side="right")
            cnt = int(right - left)
            my_id = all_id[gi]
            for j in range(left, right):
                if arr_id_valid[j] == my_id:
                    cnt -= 1
                    break
            result[gi] = cnt

    df["rwy_arr_active"] = result
    print(f"  rwy_arr_active: nn={np.sum(~np.isnan(result))}/{len(result)}, mean={np.nanmean(result):.2f}")
    return df
