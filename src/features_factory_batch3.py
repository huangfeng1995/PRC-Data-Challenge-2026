"""
批次 3: outlier 指纹特征 (feature-factory 批次 3)
==================================================

口径约束:
  - 全部基于已有业务字段, 不锚 BLOCK_TIME 为锚 (仅用 BLOCK_TIME 作为被比较字段)
  - 机场隔离: STAND / RUNWAY 匹配用 (ADEP_mvt + STAND_mvt) / (ADEP_mvt + RUNWAY_mvt)
  - 独立二值特征 → 合成 outlier_risk_score 加权求和 (float32)
  - 模糊匹配候选只用于报告, 不参与特征构造

依赖: data/external/xplane/{stands_geo.csv, runways.csv} (10 机场全覆盖)

产出列 (全部 Int8 二值 except risk_score 为 float32):
  outlier_stand_match        Int8  0=匹配成功 / 1=未匹配到 X-Plane
  outlier_rwy_match          Int8  0=匹配成功 / 1=未匹配到 X-Plane
  outlier_block_sched_clash  Int8  1=|BLOCK-SCHED| < 5min (几乎没 delay 可疑)
  outlier_crossday_slip      Int8  1=MVT - BLOCK > 86400s (24h, 数据错误)
  outlier_no_aobt            Int8  1=AOBT_3_flt NaT (无实际推出时刻)
  outlier_aobt_plan_bad      Int8  1=|AOBT - EOBT| > 20min (推出严重偏离计划)
  outlier_risk_score         float32 加权和 = 0.5*stand_match + 1.0*rwy_match
                                          + 1.5*block_sched + 2.0*crossday
                                          + 1.0*no_aobt + 1.5*aobt_plan_bad

业务假设 & 预期方向:
  outlier_* 本身不直接进模型 (特征是 risk_score), 但 risk_score 预期正 (数据脏 → taxi-out 不可信 / 需兜底)
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


def _load_xplane_sets() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """返回 (icao -> stand name set, icao -> runway name set).

    stand 名归一化: 去 -(C)/(E)/(F)/(H) 等机型后缀 (LTFM 等机场 apt.dat 有这种后缀),
    以及括号里的机型标识. runway 名不需要归一化.
    """
    stands: dict[str, set[str]] = {}
    runways: dict[str, set[str]] = {}

    _stand_suffix_re = re.compile(r"-\([A-Z]\)$")
    _stand_paren_re = re.compile(r"\([A-Z]+\)")

    # stands
    with open(XP_STANDS) as f:
        for r in csv.DictReader(f):
            icao = r["icao"].strip()
            name = r["name"].strip().upper()
            # 归一化: 去机型后缀/括号
            name = _stand_suffix_re.sub("", name)
            name = _stand_paren_re.sub("", name).strip()
            if name:
                stands.setdefault(icao, set()).add(name)

    # runways
    with open(XP_RUNWAYS) as f:
        for r in csv.DictReader(f):
            icao = r["icao"].strip()
            for key in ("rwy1", "rwy2"):
                name = r[key].strip().upper()
                if name:
                    runways.setdefault(icao, set()).add(name)

    return stands, runways


def _norm_stand(s: str) -> str:
    """机位号归一化: 去空格、全大写、去尾部冗余后缀 (不做模糊)。"""
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


def _norm_rwy(s: str) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def add_batch3_outlier_features(df: pd.DataFrame) -> pd.DataFrame:
    """对 df 追加批次 3 的 7 个 outlier 指纹列, 返回副本."""
    stands, runways = _load_xplane_sets()
    out = df.copy()

    # --- 1/2. STAND / RUNWAY 精确匹配 ---
    outlier_stand = np.zeros(len(out), dtype="int8")
    outlier_rwy = np.zeros(len(out), dtype="int8")

    icaos = out["ADEP_mvt"].astype(str).values
    stands_v = out.get("STAND_mvt", pd.Series([None] * len(out))).astype(str).values
    rwys_v = out.get("RUNWAY_mvt", pd.Series([None] * len(out))).astype(str).values

    for i in range(len(out)):
        icao = icaos[i]
        if icao == "nan" or icao not in stands:
            # 机场不在 xplane 数据里 → 标记为未匹配
            outlier_stand[i] = 1
            outlier_rwy[i] = 1
            continue
        s_norm = _norm_stand(stands_v[i])
        r_norm = _norm_rwy(rwys_v[i])
        if s_norm and s_norm not in stands[icao]:
            outlier_stand[i] = 1
        if r_norm and r_norm not in runways[icao]:
            outlier_rwy[i] = 1

    out["outlier_stand_match"] = pd.Series(outlier_stand).astype("Int64")
    out["outlier_rwy_match"] = pd.Series(outlier_rwy).astype("Int64")

    # --- 3. BLOCK vs SCHED clash (几乎没 delay → 数据可疑) ---
    block = pd.to_datetime(out.get("BLOCK_TIME_UTC_mvt"), utc=True, errors="coerce")
    sched = pd.to_datetime(out.get("SCHED_TIME_UTC_mvt"), utc=True, errors="coerce")
    outlier_block_clash = (
        block.notna() & sched.notna() & ((block - sched).dt.total_seconds().abs() < 300)
    ).astype("Int8")
    out["outlier_block_sched_clash"] = pd.Series(outlier_block_clash).astype("Int64")

    # --- 4. 跨日滑行 slip (>24h) ---
    mvt = pd.to_datetime(out.get("MVT_TIME_UTC_mvt"), utc=True, errors="coerce")
    crossday = (
        block.notna() & mvt.notna() & ((mvt - block).dt.total_seconds() > 86400)
    ).astype("Int8")
    out["outlier_crossday_slip"] = pd.Series(crossday).astype("Int64")

    # --- 5. 无 AOBT ---
    aobt = pd.to_datetime(out.get("AOBT_3_flt"), utc=True, errors="coerce")
    no_aobt = aobt.isna().astype("Int8")
    out["outlier_no_aobt"] = pd.Series(no_aobt).astype("Int64")

    # --- 6. AOBT 严重偏离计划 (|AOBT - EOBT| > 20min) ---
    eobt = pd.to_datetime(out.get("EOBT_1_flt"), utc=True, errors="coerce")
    aobt_bad = (
        aobt.notna() & eobt.notna() & ((aobt - eobt).dt.total_seconds().abs() > 1200)
    ).astype("Int8")
    out["outlier_aobt_plan_bad"] = pd.Series(aobt_bad).astype("Int64")

    # --- 7. outlier_risk_score (加权和) ---
    # 用 df 里已赋值的列算 risk_score, 避免 numpy/Series 混用
    stand_v = out["outlier_stand_match"].fillna(0).astype("float32")
    rwy_v = out["outlier_rwy_match"].fillna(0).astype("float32")
    block_v = out["outlier_block_sched_clash"].fillna(0).astype("float32")
    cross_v = out["outlier_crossday_slip"].fillna(0).astype("float32")
    aobtno_v = out["outlier_no_aobt"].fillna(0).astype("float32")
    aobtbad_v = out["outlier_aobt_plan_bad"].fillna(0).astype("float32")
    score = (0.5 * stand_v + 1.0 * rwy_v + 1.5 * block_v + 2.0 * cross_v
             + 1.0 * aobtno_v + 1.5 * aobtbad_v).astype("float32")
    # 全 NaT / 无数据的行 → risk_score 用 NaN 表示"无法判断"
    all_nan = (block.isna() & sched.isna() & mvt.isna() & aobt.isna() & eobt.isna())
    score = score.where(~all_nan, np.nan)
    out["outlier_risk_score"] = score

    return out


# ---------------------------------------------------------------------------
# 模糊匹配报告 (只报告不并入特征)
# ---------------------------------------------------------------------------
def report_fuzzy_candidates(df: pd.DataFrame, top_n_per_icao: int = 5) -> pd.DataFrame:
    """对 outlier_stand_match / outlier_rwy_match 的行, 尝试模糊匹配候选, 返回清单.

    模糊规则 (只用于报告, 不自动合并):
      - 去常见后缀: /(C|E|F|L|R|B|A|N|X)$/ (机型后缀)
      - 去分隔符: - / _ / 空格
      - 全大写 + strip

    返回 DataFrame: (icao, field_type, raw, fuzzy_match_candidates, n_candidates)
    """
    stands, runways = _load_xplane_sets()
    records = []
    icaos = df["ADEP_mvt"].astype(str).values
    stands_v = df.get("STAND_mvt", pd.Series([""] * len(df))).astype(str).values
    rwys_v = df.get("RUNWAY_mvt", pd.Series([""] * len(df))).astype(str).values

    for i in range(len(df)):
        icao = icaos[i]
        if icao == "nan" or icao not in stands:
            continue
        for field_type, raw, candidate_set in [
            ("STAND", stands_v[i], stands[icao]),
            ("RUNWAY", rwys_v[i], runways[icao]),
        ]:
            norm = raw.strip().upper()
            if norm in candidate_set or not norm:
                continue
            # 生成候选
            cand = set()
            cand.add(norm)
            cand.add(norm.rstrip("CEFLRBANX"))  # 去机型后缀
            cand.add(norm.replace("-", "").replace("_", "").replace(" ", ""))
            cand.add(norm.rstrip("CEFLRBANX").replace("-", "").replace(" ", ""))
            found = []
            for c in cand:
                if c in candidate_set:
                    found.append(c)
                else:
                    # 再试 substring / 去前缀 'Z' (AOA 常见 Z prefix)
                    for real in candidate_set:
                        if real.startswith(c) or c.startswith(real):
                            found.append(real)
                            break
            records.append({
                "icao": icao,
                "field": field_type,
                "raw": raw,
                "fuzzy_candidates": sorted(set(found))[:3],
                "n_candidates": len(set(found)),
            })

    rep = pd.DataFrame(records)
    if len(rep) == 0:
        return rep
    # 每个 icao×field 只留 top_n
    rep = rep.groupby(["icao", "field"]).head(top_n_per_icao).reset_index(drop=True)
    return rep
