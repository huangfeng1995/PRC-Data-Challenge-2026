"""
官方框架新机制特征 (feature-factory 批次 E)
============================================

铁律: 不锚 BLOCK_TIME, ranking 全字段

特征族 (4 列):

E.1 流量调控代理 (机场×时段)
    flow_proxy_aobt_sched_min    float32  (AOBT_hour - SCHED_hour)*60 + (AOBT_min - SCHED_min)
                                                  真实推出 vs 计划时段的整体性后移 (分钟)
                                                  > 15min 占比 → flow_proxy_late_15p
    flow_proxy_late_15p         Int64    同 (ADEP, AOBT_date) 内, |AOBT - SCHED| > 15min 的航班占比

E.2 跑道穿越 / 构型 (坐标计算)
    stand_bearing_to_rwy_deg    float32  机位→所用跑道 方位角 (0~360°), haversine 推算
    active_rwys_in_window       Int64    ref_i 时刻 ref_i-15min 同机场活跃跑道数 (MVT/ARVT 都算)

业务假设:
  flow_proxy_aobt_sched_min → AOBT 系统性后移 (机场流量调控) → taxi 长 → 正
  flow_proxy_late_15p       → 1 = 高占比后移 → 机场拥堵 → 正
  stand_bearing_to_rwy_deg  → 长距 = 长 taxi → 正
  active_rwys_in_window     → 跑道少 = 拥堵 → 正

依赖:
  data/external/xplane/stands_geo.csv
  data/external/xplane/runways.csv
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XP_STANDS = REPO / "data" / "external" / "xplane" / "stands_geo.csv"
XP_RUNWAYS = REPO / "data" / "external" / "xplane" / "runways.csv"


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """机位 lat1,lon1 → 跑道端点 lat2,lon2 方位角, 0~360° (北为 0)."""
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def add_batchE_official_features(df: pd.DataFrame) -> pd.DataFrame:
    """追加 4 列批次 E 特征."""
    out = df.copy()
    aobt = pd.to_datetime(out.get("AOBT_3_flt"), utc=True, errors="coerce")
    eobt = pd.to_datetime(out.get("EOBT_1_flt"), utc=True, errors="coerce")
    iobt = pd.to_datetime(out.get("IOBT_flt"), utc=True, errors="coerce")
    sched = pd.to_datetime(out.get("SCHED_TIME_UTC_mvt"), utc=True, errors="coerce")
    mvt = pd.to_datetime(out.get("MVT_TIME_UTC_mvt"), utc=True, errors="coerce")
    arvt = pd.to_datetime(out.get("ARVT_3_flt"), utc=True, errors="coerce")

    # E.1 flow_proxy: AOBT 相对 SCHED 的分钟后移 (按行)
    out["flow_proxy_aobt_sched_min"] = (
        (aobt - sched).dt.total_seconds() / 60
    ).astype("float32")

    # E.1b flow_proxy_late_15p: 同 (ADEP, AOBT_date) 内 > 15min 的占比
    if aobt.notna().any() and sched.notna().any():
        aobt_date = aobt.dt.date
        late_mask = (aobt - sched).dt.total_seconds().abs() > 900
        df_tmp = pd.DataFrame({"ADEP": out["ADEP_mvt"].values, "date": aobt_date.values, "late": late_mask.fillna(False).values})
        grp = df_tmp.groupby(["ADEP", "date"])["late"]
        total = grp.transform("size")
        late_cnt = grp.transform("sum")
        out["flow_proxy_late_15p"] = (late_cnt / total.replace(0, np.nan)).astype("float32")
    else:
        out["flow_proxy_late_15p"] = np.nan

    # E.2 stand_bearing_to_rwy_deg: 机位→跑道方位角
    # 需要 xplane 坐标
    out["stand_bearing_to_rwy_deg"] = _compute_stand_rwy_bearing(out)

    # E.2b active_rwys_in_window: ref_i 时刻同机场过去 15min 活跃跑道数 (DEP/ARR 合并)
    out["active_rwys_in_window"] = _compute_active_runways(out)

    return out


def _compute_stand_rwy_bearing(df) -> np.ndarray:
    """机位→对应跑道端点的方位角."""
    # 加载 xplane 坐标
    stand_coords: dict[str, dict[str, tuple[float, float]]] = {}  # icao -> name -> (lat, lon)
    rwy_coords: dict[str, dict[str, tuple[float, float]]] = {}  # icao -> rwy_name -> (lat, lon)
    with open(XP_STANDS) as f:
        for r in csv.DictReader(f):
            stand_coords.setdefault(r["icao"], {})[r["name"].strip()] = (float(r["lat"]), float(r["lon"]))
    with open(XP_RUNWAYS) as f:
        for r in csv.DictReader(f):
            icao = r["icao"]
            rwy_coords.setdefault(icao, {})[r["rwy1"].strip()] = (float(r["lat1"]), float(r["lon1"]))
            rwy_coords.setdefault(icao, {})[r["rwy2"].strip()] = (float(r["lat2"]), float(r["lon2"]))

    icaos = df["ADEP_mvt"].astype(str).values
    stands = df.get("STAND_mvt", pd.Series([""]*len(df))).astype(str).values
    rwys = df.get("RUNWAY_mvt", pd.Series([""]*len(df))).astype(str).values

    out = np.full(len(df), np.nan, dtype="float32")
    for i in range(len(df)):
        icao = icaos[i]
        s = stands[i].strip().upper()
        r = rwys[i].strip().upper()
        if icao == "nan" or icao not in stand_coords:
            continue
        s_coord = stand_coords[icao].get(s) or stand_coords[icao].get(s.rstrip("LR"))
        r_coord = rwy_coords.get(icao, {}).get(r) or rwy_coords.get(icao, {}).get(r.rstrip("LR"))
        if s_coord and r_coord:
            out[i] = _bearing_deg(s_coord[0], s_coord[1], r_coord[0], r_coord[1])
    return out


def _compute_active_runways(df) -> np.ndarray:
    """ref_i 时刻同机场过去 15min MVT/ARVT 涉及的活跃跑道数."""
    mvt = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    arvt = pd.to_datetime(df.get("ARVT_3_flt"), utc=True, errors="coerce") if "ARVT_3_flt" in df.columns else pd.Series([pd.NaT]*len(df))
    ref_i = pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce")
    adep = df["ADEP_mvt"].astype(str).values
    rwy_dep = df["RUNWAY_mvt"].astype(str).values
    # ARR 跑道不在我们数据里 (ranking 上 ADES_mvt 不等于 ADEP), 但我们合成的 df 里没有 ARR 的 RUNWAY_mvt
    # 简化: 只用 DEP MVT 涉及的 RUNWAY_mvt 计数活跃跑道
    W = np.timedelta64(15 * 60, "s")
    out = np.zeros(len(df), dtype="int64")
    for ap in np.unique(adep):
        if ap == "nan":
            continue
        gi = np.where(adep == ap)[0]
        # 收集同机场的所有 DEP MVT (RUNWAY_mvt 在 ref_i-15min 内的)
        for pos in gi:
            ref = ref_i.iloc[pos]
            if pd.isna(ref):
                continue
            t_lo = ref - W
            # 在 gi 里找 MVT ∈ (t_lo, ref_i] 的 RUNWAY 唯一值
            active = set()
            for j in gi:
                m = mvt.iloc[j]
                if pd.isna(m) or m <= t_lo or m > ref:
                    continue
                r = rwy_dep[j]
                if r and r != "nan":
                    active.add(r)
            out[pos] = len(active)
    return out
