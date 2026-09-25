"""
计划/时间特征 (feature-factory 批次 D)
====================================

口径约束:
  1. ranking TAXITIME_SEC_mvt 50% 缺失, BLOCK_TIME 50% 缺失 → 不用 TAXITIME/BLOCK
  2. 全部基于 ranking 全字段: AOBT_3_flt / EOBT_1_flt / IOBT_flt / LOBT_flt / SCHED_TIME_UTC_mvt / ADEP_mvt
  3. 时间戳 datetime64[ns] UTC, float32 输出

特征族 (8 列):
  plan_aobt_eobt_min      float32  (AOBT - EOBT) / 60  (分钟, 反映 EOBT 估计误差)
  plan_lobt_iobt_min      float32  (LOBT - IOBT) / 60
  plan_aobt_sched_min     float32  (AOBT - SCHED) / 60
  plan_eobt_iobt_min      float32  (EOBT - IOBT) / 60   (内部计划对齐)
  plan_iobt_eobt_consist  float32  (IOBT-EOBT 一致性 = |IOBT-EOBT|< 60s → 1, else 0)
  plan_lobt_eobt_min      float32  (LOBT - EOBT) / 60
  ap_hour_coord_peak      Int64    机场×小时 (ADEP×AOBT_hour) 协调高峰 1=高于 P80, 0=否
  ap_hour_density_rank    Int64    1~5 密度分位 (基于 (ADEP, AOBT_hour) 航班数, 5=最忙)

业务假设:
  plan_*_min  → 反映"计划/实际偏差", taxi-out 时间常因实际偏差增加 → 预期正相关
  plan_iobt_eobt_consist → IOBT/EOBT 一致 = 内部协调好 → 预期负相关 (好→taxi 短)
  ap_hour_coord_peak → 高峰期 → 拥堵 → taxi 长 → 预期正
  ap_hour_density_rank → 越忙 → taxi 长 → 预期正
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _parse(df, col):
    return pd.to_datetime(df.get(col), utc=True, errors="coerce")


def add_batchD_plan_features(df: pd.DataFrame) -> pd.DataFrame:
    """追加 8 列 plan/time, 返回 df 副本."""
    out = df.copy()
    aobt = _parse(out, "AOBT_3_flt")
    eobt = _parse(out, "EOBT_1_flt")
    iobt = _parse(out, "IOBT_flt")
    lobt = _parse(out, "LOBT_flt")
    sched = _parse(out, "SCHED_TIME_UTC_mvt")

    # 1-5. 分钟级差值 (AOBT 优先, EOBT 兜底)
    out["plan_aobt_eobt_min"] = ((aobt - eobt).dt.total_seconds() / 60).astype("float32")
    out["plan_lobt_iobt_min"] = ((lobt - iobt).dt.total_seconds() / 60).astype("float32")
    out["plan_aobt_sched_min"] = ((aobt - sched).dt.total_seconds() / 60).astype("float32")
    out["plan_eobt_iobt_min"] = ((eobt - iobt).dt.total_seconds() / 60).astype("float32")
    out["plan_lobt_eobt_min"] = ((lobt - eobt).dt.total_seconds() / 60).astype("float32")

    # 6. IOBT/EOBT 一致性 (内部协调)
    out["plan_iobt_eobt_consist"] = (
        (iobt.notna() & eobt.notna() & ((iobt - eobt).dt.total_seconds().abs() < 60))
        .fillna(False).astype("int8")
    ).astype("Int64")

    # 7. ap_hour_coord_peak: (ADEP, AOBT_hour) 航班数 > P80 (基于训练集统计)
    if aobt.notna().any():
        out["_aobt_hour"] = aobt.dt.hour
        # 密度 (count) per (adep, hour)
        density = out.groupby(["ADEP_mvt", "_aobt_hour"]).size()
        # P80 cutoff
        p80 = density.quantile(0.8)
        is_peak = density[density > p80].index
        peak_mask = pd.MultiIndex.from_frame(out[["ADEP_mvt","_aobt_hour"]]).isin(is_peak)
        out["ap_hour_coord_peak"] = pd.Series(peak_mask).astype("Int64")
        # 8. 密度分位 rank (1~5)
        # qcut 在 density 上: 把 (adep, hour) 的密度 5 等分
        ranks = pd.qcut(density, 5, labels=False, duplicates="drop")
        # ranks 现在是 0~N-1, 映射回 out
        rank_map = ranks.to_dict()
        out["ap_hour_density_rank"] = (
            out.apply(lambda r: rank_map.get((r["ADEP_mvt"], r["_aobt_hour"]), -1) + 1, axis=1)
            .astype("Int64")
        )
        out.drop(columns=["_aobt_hour"], inplace=True)
    else:
        out["ap_hour_coord_peak"] = pd.Series([0]*len(out), dtype="Int64")
        out["ap_hour_density_rank"] = pd.Series([0]*len(out), dtype="Int64")

    return out
