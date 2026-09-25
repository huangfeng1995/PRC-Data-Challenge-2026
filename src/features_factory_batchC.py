"""
Outlier 指纹特征 (feature-factory 批次 C, ranking 兼容)
======================================================

铁律: ranking 上 BLOCK_TIME / TAXITIME 50% 缺失 → 改用 AOBT/SCHED/AOBT 日期

口径约束:
  1. 只用 ranking 全字段: AOBT_3_flt / EOBT_1_flt / SCHED_TIME_UTC_mvt / STAND_mvt / RUNWAY_mvt / ADEP_mvt
  2. 机场隔离: stand_match / rwy_match 按 (ADEP_mvt, field) 查找
  3. 跑道号/机位名归一化: 先做口径统一, 减少假阳性

特征族 (6 列):
  outlier_stand_match         Int64   STAND_mvt 在 xplane/stands_geo 精确匹配 (含归一化)
  outlier_rwy_match           Int64   RUNWAY_mvt 在 xplane/runways 精确匹配 (含归一化)
  outlier_no_aobt             Int64   AOBT_3_flt NaT
  outlier_aobt_plan_bad       Int64   |AOBT - EOBT| > 20min (推出严重偏离计划)
  outlier_aobt_sched_clash    Int64   |AOBT - SCHED| < 5min (几乎无 delay → 数据可疑, 替代 BLOCK clash)
  outlier_crossday            Int64   AOBT 日期 ≠ SCHED 日期 (跨日, 替代 MVT-BLOCK)
  outlier_risk_score          float32 加权和 (按子项重要性重定)
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XP_STANDS = REPO / "data" / "external" / "xplane" / "stands_geo.csv"
XP_RUNWAYS = REPO / "data" / "external" / "xplane" / "runways.csv"

# 跑道号口径归一化: xplane 用 "07C/25C" 字母数字+方向, sample 用 "07"/"28" 纯数字 / 双向
# 实际归一化规则:
#   - 大写, 去 L/R 后缀 (样本里 "07L"/"07R" 都应匹配到 xplane 的 "07L/07R", 但有些样本不带 L/R)
#   - 去 "0" 前缀: "07" → "7"
# 我们在 _norm_rwy 做双向归一化 (样本 → xplane 字典 / xplane → 字典)
_RWY_LR_SUFFIX_RE = re.compile(r"[LR]$")
_RWY_LEADING_ZERO_RE = re.compile(r"^0")


def _load_xplane_sets() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """返回 (icao -> stand set, icao -> runway set). 归一化同时去除 LR 后缀."""
    stands: dict[str, set[str]] = {}
    runways: dict[str, set[str]] = {}

    _stand_suffix_re = re.compile(r"-\([A-Z]\)$")
    _stand_paren_re = re.compile(r"\([A-Z]+\)")

    with open(XP_STANDS) as f:
        for r in csv.DictReader(f):
            icao = r["icao"].strip()
            name = r["name"].strip().upper()
            name = _stand_suffix_re.sub("", name)
            name = _stand_paren_re.sub("", name).strip()
            if name:
                stands.setdefault(icao, set()).add(name)
                # 也加 "去 LR 后缀" 版本
                name2 = _RWY_LR_SUFFIX_RE.sub("", name)
                if name2 != name:
                    stands[icao].add(name2)

    with open(XP_RUNWAYS) as f:
        for r in csv.DictReader(f):
            icao = r["icao"].strip()
            for key in ("rwy1", "rwy2"):
                name = r[key].strip().upper()
                if name:
                    runways.setdefault(icao, set()).add(name)
                    # 加归一化版本 (去 LR 后缀, 去 0 前缀)
                    name2 = _RWY_LR_SUFFIX_RE.sub("", name)
                    name3 = _RWY_LEADING_ZERO_RE.sub("", name2)
                    if name2 != name:
                        runways[icao].add(name2)
                    if name3 != name and name3:
                        runways[icao].add(name3)

    return stands, runways


def _norm_stand(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


def _norm_rwy(s) -> str:
    if pd.isna(s):
        return ""
    n = str(s).strip().upper()
    # 双向归一化: 大写后去掉 LR 后缀 (sample "07L" → "07", xplane "07L" 原样保留)
    n2 = _RWY_LR_SUFFIX_RE.sub("", n)
    return n2


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def add_batchC_outlier_features(df: pd.DataFrame) -> pd.DataFrame:
    """追加 7 列 outlier 指纹, 返回 df 副本."""
    stands, runways = _load_xplane_sets()
    out = df.copy()

    # 1/2. STAND / RUNWAY 匹配
    out_std = pd.Series(np.zeros(len(out), dtype="int8")).astype("Int64")
    out_rwy = pd.Series(np.zeros(len(out), dtype="int8")).astype("Int64")

    icaos = out["ADEP_mvt"].astype(str).values
    stands_v = out.get("STAND_mvt", pd.Series([""]*len(out))).astype(str).values
    rwys_v = out.get("RUNWAY_mvt", pd.Series([""]*len(out))).astype(str).values

    for i in range(len(out)):
        icao = icaos[i]
        if icao == "nan" or icao not in stands:
            out_std.iloc[i] = 1
            out_rwy.iloc[i] = 1
            continue
        s_norm = _norm_stand(stands_v[i])
        r_norm = _norm_rwy(rwys_v[i])
        if s_norm and s_norm not in stands[icao]:
            out_std.iloc[i] = 1
        if r_norm and r_norm not in runways[icao]:
            out_rwy.iloc[i] = 1

    out["outlier_stand_match"] = out_std
    out["outlier_rwy_match"] = out_rwy

    # 3. no_aobt
    aobt = pd.to_datetime(out.get("AOBT_3_flt"), utc=True, errors="coerce")
    out["outlier_no_aobt"] = aobt.isna().astype("Int64")

    # 4. aobt_plan_bad: |AOBT - EOBT| > 20min
    eobt = pd.to_datetime(out.get("EOBT_1_flt"), utc=True, errors="coerce")
    out["outlier_aobt_plan_bad"] = (
        aobt.notna() & eobt.notna() & ((aobt - eobt).dt.total_seconds().abs() > 1200)
    ).astype("Int64")

    # 5. aobt_sched_clash: |AOBT - SCHED| < 5min (替代 BLOCK clash)
    sched = pd.to_datetime(out.get("SCHED_TIME_UTC_mvt"), utc=True, errors="coerce")
    out["outlier_aobt_sched_clash"] = (
        aobt.notna() & sched.notna() & ((aobt - sched).dt.total_seconds().abs() < 300)
    ).astype("Int64")

    # 6. crossday: AOBT.date ≠ SCHED.date
    out["outlier_crossday"] = (
        aobt.notna() & sched.notna() & (aobt.dt.date != sched.dt.date)
    ).astype("Int64")

    # 7. risk_score (子项已用 AOBT 系列, BLOCK 替代物已就绪)
    stand_v = out["outlier_stand_match"].fillna(0).astype("float32")
    rwy_v = out["outlier_rwy_match"].fillna(0).astype("float32")
    bad_v = out["outlier_aobt_plan_bad"].fillna(0).astype("float32")
    clash_v = out["outlier_aobt_sched_clash"].fillna(0).astype("float32")
    cross_v = out["outlier_crossday"].fillna(0).astype("float32")
    no_aobt_v = out["outlier_no_aobt"].fillna(0).astype("float32")
    # 权重: crossday > stand/rwy (数据错误) > plan_bad > no_aobt > clash
    score = (
        0.5 * stand_v + 0.8 * rwy_v + 1.5 * bad_v + 0.5 * clash_v
        + 2.0 * cross_v + 1.0 * no_aobt_v
    ).astype("float32")
    out["outlier_risk_score"] = score

    return out
