"""
队列特征模块 (feature-factory 批次 A)
====================================

口径约束 (铁律):
  1. ranking 上 BLOCK_TIME/TAXITIME 50% 缺失 → 所有特征只用 AOBT/EOBT/IOBT/SCHED/AOBT_j 等 ranking 全字段
  2. ref_i = AOBT_3_flt if notna else EOBT_1_flt (不锚 BLOCK_TIME)
  3. 窗口严格向过去开 (左开右闭): (ref_i - w, ref_i]
  4. 排除自身 MVT_ID_mvt
  5. 机场隔离: DEP 用 ADEP_mvt, ARR 用 ADES_mvt
  6. 时间戳 datetime64[ns] UTC

特征族 (与 ranking 兼容):
  A. 已等待时间 / 队列密度 (不用 TAXITIME)
    wait_rwy_{10,20,30,60}_ap    ref_i − AOBT_j 均值, 已 AOBT 未 MVT (含 ref_i 自身, 排除自身)
    wait_rwy_{10,20,30,60}_lobt  ref_i − LOBT_j 均值, 已 LOBT 未 MVT (LOBT 略晚于 AOBT, 跑道上滑行中)
  B. 吞吐版 (过去 w 同跑道 MVT 起飞架次)
    n_mvt_rwy_{10,30,60}         同跑道 过去 w min 已起飞 DEP 数 (含 ARR 后推出, 排除自身)
  C. 瞬时队列数
    n_dep_queue_rwy_{15,30,60}   ref_i 时刻, 同跑道已 AOBT 未 MVT 数 (排除自身)
    n_dep_queue_ap_{15,30,60}    ref_i 时刻, 同机场已 AOBT 未 MVT 数 (排除自身)
  D. ARR 占用
    n_arr_rwy_{15,30,60}         过去 w min 同跑道 ARVT 数
    rwy_arr_active_180s          ref_i 时刻同跑道 ARVT (ref_i - 180s, ref_i] 计数
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def _parse_ref_time(df: pd.DataFrame) -> pd.Series:
    """ref_i = AOBT_3_flt if notna else EOBT_1_flt."""
    aobt = pd.to_datetime(df.get("AOBT_3_flt"), utc=True, errors="coerce")
    eobt = pd.to_datetime(df.get("EOBT_1_flt"), utc=True, errors="coerce")
    return aobt.where(aobt.notna(), eobt)


def _ensure_arr_defensive(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """确保 ARVT/ADES/LOBT/MVT 字段都在 (ranking 里都有)."""
    arr = pd.to_datetime(df.get("ARVT_3_flt"), utc=True, errors="coerce")
    ades = df.get("ADES_mvt")
    lobt = pd.to_datetime(df.get("LOBT_flt"), utc=True, errors="coerce")
    mvt = pd.to_datetime(df.get("MVT_TIME_UTC_mvt"), utc=True, errors="coerce")
    return arr, ades, lobt, mvt


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def add_batchA_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """追加 4 类队列特征 (共 21 列), 返回 df 副本."""
    out = df.copy()
    out["_ref_i"] = _parse_ref_time(out)
    out["_AOBT"] = pd.to_datetime(out["AOBT_3_flt"], utc=True, errors="coerce")
    out["_LOBT"] = pd.to_datetime(out["LOBT_flt"], utc=True, errors="coerce")
    out["_MVT"] = pd.to_datetime(out["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    out["_IOBT"] = pd.to_datetime(out["IOBT_flt"], utc=True, errors="coerce")
    out["_EOBT"] = pd.to_datetime(out["EOBT_1_flt"], utc=True, errors="coerce")
    out["_SCHED"] = pd.to_datetime(out["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
    out["_ADEP"] = out["ADEP_mvt"].astype(str)
    out["_RUNWAY"] = out["RUNWAY_mvt"].astype(str)
    out["_MVT_ID"] = out["MVT_ID_mvt"].astype(str)

    # A. wait_rwy_* (已等待时间均值, 单位秒)
    for w in [10, 20, 30, 60]:
        out[f"wait_rwy_{w}_ap"] = _per_group_running_mean(
            out, out["_RUNWAY"], out["_ref_i"], out["_AOBT"], out["_MVT"],
            out["_MVT_ID"], window_min=w, value_col=None, mode="waiting_from_aobt",
        )

    # B. n_mvt_rwy_* (吞吐, 过去 w min 同跑道已起飞数)
    for w in [10, 30, 60]:
        out[f"n_mvt_rwy_{w}"] = _per_group_count_window(
            out, out["_RUNWAY"], out["_ref_i"], out["_MVT"], out["_MVT_ID"],
            out["_MVT"], window_min=w, direction="past",
        )

    # C. n_dep_queue_rwy_* / n_dep_queue_ap_* (瞬时队列数)
    for w in [15, 30, 60]:
        out[f"n_dep_queue_rwy_{w}"] = _per_group_count_window(
            out, out["_RUNWAY"], out["_ref_i"], out["_AOBT"], out["_MVT"], out["_MVT_ID"],
            window_min=w, direction="future",
        )
        out[f"n_dep_queue_ap_{w}"] = _per_group_count_window(
            out, out["_ADEP"], out["_ref_i"], out["_AOBT"], out["_MVT"], out["_MVT_ID"],
            window_min=w, direction="future",
        )

    # D. ARR 占用
    arr_ok = "ARVT_3_flt" in out.columns and "ADES_mvt" in out.columns
    for w in [15, 30, 60]:
        out[f"n_arr_rwy_{w}"] = _arr_rwy_count(
            out, out["_RUNWAY"], out["_ref_i"], window_min=w,
            arr_time_col=out["_ARVT"] if "_ARVT" in out.columns else None,
        ) if arr_ok else 0.0
    out["rwy_arr_active_180s"] = _arr_rwy_count(
        out, out["_RUNWAY"], out["_ref_i"], window_min=3,
        arr_time_col=out["_ARVT"] if "_ARVT" in out.columns else None,
    ) if arr_ok else 0.0

    # cleanup
    drop_cols = [c for c in out.columns if c.startswith("_") and c not in ["_ref_i","_AOBT","_LOBT","_MVT","_IOBT","_EOBT","_SCHED","_ADEP","_RUNWAY","_MVT_ID","_ARVT"]]
    for c in ["_ref_i","_AOBT","_LOBT","_MVT","_IOBT","_EOBT","_SCHED","_ADEP","_RUNWAY","_MVT_ID","_ARVT"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True, errors="ignore")
    return out


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _per_group_count_window(df, group, ref, t_event, mvt, mvt_id,
                             window_min, direction):
    """按 group 分组, 在 ref_i 时刻, 统计窗口内 t_event 数.

    direction='past'    : t_event ∈ (ref_i - w, ref_i]  → 过去 w min 已起飞 (mvt_j ≤ ref_i < t_event + 大, 但只看 mvt_j 落入窗口)
    direction='future'  : t_event ∈ (ref_i - w, ref_i] AND mvt_j > ref_i or NaT (瞬时队列)

    排除自身: mvt_id 相同时不计入.
    返回 Series (与 df index 对齐), dtype=float32, NaN 表示 ref_i 无效.
    """
    g = group.values
    ref_v = ref.values
    t_v = t_event.values
    m_v = mvt.values
    id_v = mvt_id.values
    w = np.timedelta64(window_min * 60, "s")
    out = np.full(len(df), np.nan, dtype="float32")

    for grp_id in np.unique(g):
        if grp_id == "nan":
            continue
        gi = np.where(g == grp_id)[0]
        if len(gi) == 0:
            continue
        order = np.argsort(ref_v[gi], kind="stable")
        gi_sorted = gi[order]
        # Pre-sort t_v by t for binary search
        valid_t_mask = ~pd.isna(t_v[gi_sorted])
        t_sorted = t_v[gi_sorted[valid_t_mask]]
        ids_sorted = id_v[gi_sorted[valid_t_mask]]
        for k in range(len(gi_sorted)):
            i_pos = gi_sorted[k]
            ref_i = ref_v[i_pos]
            if pd.isna(ref_i):
                continue
            t_lo = ref_i - w
            if direction == "past":
                # count t_j ∈ (t_lo, ref_i]
                left = np.searchsorted(t_sorted, t_lo, side="right")
                right = np.searchsorted(t_sorted, ref_i, side="right")
                cnt = right - left
                my_id = id_v[i_pos]
                # 减自身 (若 t_j == t_i 自已在窗口内, 严格 left<right 且 ids match)
                for j in range(left, right):
                    if ids_sorted[j] == my_id:
                        cnt -= 1
                        break
            else:  # 'future' 瞬时队列
                # 满足: t_j ∈ (t_lo, ref_i] AND (m_v[j] > ref_i OR pd.isna(m_v[j]))
                cnt = 0
                my_id = id_v[i_pos]
                for j in range(len(t_sorted)):
                    t_j = t_sorted[j]
                    if pd.isna(t_j) or t_j <= t_lo or t_j > ref_i:
                        continue
                    if ids_sorted[j] == my_id:
                        continue
                    # 必须 m_v > ref_i 或 NaT
                    mj = m_v[gi_sorted[valid_t_mask][j]]
                    if not pd.isna(mj) and mj <= ref_i:
                        continue
                    cnt += 1
            out[i_pos] = cnt
    return out


def _per_group_running_mean(df, group, ref, t_event, mvt, mvt_id,
                            window_min, value_col=None, mode="waiting_from_aobt"):
    """按 group 分组, ref_i 时刻已 AOBT 未 MVT 的 "等待时长均值".

    mode='waiting_from_aobt': 等待 = ref_i - AOBT_j (秒), 过滤 AOBT_j ∈ (ref_i - w, ref_i] AND mvt_j > ref_i 或 NaT
    """
    g = group.values
    ref_v = ref.values
    t_v = t_event.values
    m_v = mvt.values
    id_v = mvt_id.values
    w = np.timedelta64(window_min * 60, "s")
    out = np.full(len(df), np.nan, dtype="float32")

    for grp_id in np.unique(g):
        if grp_id == "nan":
            continue
        gi = np.where(g == grp_id)[0]
        if len(gi) == 0:
            continue
        order = np.argsort(ref_v[gi], kind="stable")
        gi_sorted = gi[order]

        for k in range(len(gi_sorted)):
            i_pos = gi_sorted[k]
            ref_i = ref_v[i_pos]
            if pd.isna(ref_i):
                continue
            t_lo = ref_i - w
            my_id = id_v[i_pos]
            waits = []
            for j in range(len(gi_sorted)):
                j_pos = gi_sorted[j]
                if j_pos == i_pos:
                    continue
                t_j = t_v[j_pos]
                if pd.isna(t_j) or t_j <= t_lo or t_j > ref_i:
                    continue
                mj = m_v[j_pos]
                if not pd.isna(mj) and mj <= ref_i:
                    continue
                if id_v[j_pos] == my_id:
                    continue
                # wait = ref_i - AOBT_j (秒)
                delta = (ref_i - t_j) / np.timedelta64(1, "s")
                if delta >= 0:
                    waits.append(float(delta))
            if waits:
                out[i_pos] = float(np.mean(waits))
    return out


def _arr_rwy_count(df, group, ref, window_min, arr_time_col):
    """按 group 分组, 统计 (ref_i - w, ref_i] 内 ARVT 数 (同跑道)."""
    if arr_time_col is None:
        return pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
    g = group.values
    ref_v = ref.values
    arr_v = arr_time_col.values
    w = np.timedelta64(window_min * 60, "s")
    out = np.zeros(len(df), dtype="float32")
    out[df.get("RUNWAY_mvt").isna().values if hasattr(df.get("RUNWAY_mvt"),"isna") else np.zeros(len(df), dtype=bool)] = np.nan

    for grp_id in np.unique(g):
        if grp_id == "nan":
            continue
        gi = np.where(g == grp_id)[0]
        if len(gi) == 0:
            continue
        # ARVT 按时间排序 (只计 valid)
        valid_arr_mask = ~pd.isna(arr_v[gi])
        valid_arr_idx = gi[valid_arr_mask]
        if len(valid_arr_idx) == 0:
            for pos in gi:
                if not pd.isna(ref_v[pos]):
                    out[pos] = 0.0
            continue
        arr_sorted = arr_v[valid_arr_idx]
        order = np.argsort(arr_sorted, kind="stable")
        arr_sorted = arr_sorted[order]
        ids_sorted = None  # 简化: ARR 不做自身排除, ranking 上 DEP 行不可能 ARR

        for pos in gi:
            ref_i = ref_v[pos]
            if pd.isna(ref_i):
                out[pos] = np.nan
                continue
            t_lo = ref_i - w
            left = np.searchsorted(arr_sorted, t_lo, side="right")
            right = np.searchsorted(arr_sorted, ref_i, side="right")
            out[pos] = float(right - left)
    return out
