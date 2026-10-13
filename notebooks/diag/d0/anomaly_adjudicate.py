"""
anomaly_adjudicate.py — 裁定 171 行 BLOCK==SCHED 的性质
核心问题: 这些行是"真实超长滑行", 还是"BLOCK 被填成计划值导致的标签损坏"?
判据: 若 AOBT 真实且 MVT-AOBT 正常(与正常样本同量级), 则 TAXITIME 是被 BLOCK 污染的假长尾。
只读
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "anomaly_adjudicate.json"
NS, BIG, TOL = 1_000_000_000, 10800, 60

def log(*a): print(" ".join(str(x) for x in a), flush=True)

COLS = ["MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt",
        "IOBT_flt", "EOBT_1_flt", "ADEP_mvt", "ADEP_flt", "RUNWAY_mvt",
        "STAND_mvt", "FLIGHT_mvt"]
dep = pd.concat([pd.read_parquet(p, columns=COLS)
                 for p in sorted(glob.glob(str(ROOT / "data/raw/training_2025*.parquet")))],
                ignore_index=True)
dep = dep[dep["PHASE_mvt"] == "DEP"].reset_index(drop=True)
TC = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
      "AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt"]
for c in TC:
    dep[c] = pd.to_datetime(dep[c], utc=True, errors="coerce")
    dep[c + "_s"] = (dep[c].dt.tz_convert("UTC").dt.tz_localize(None)
                     .astype("datetime64[ns]").astype("int64") // NS)

anom = dep[dep["TAXITIME_SEC_mvt"] > BIG].copy().reset_index(drop=True)
norm = dep[dep["TAXITIME_SEC_mvt"] <= BIG].copy().reset_index(drop=True)
d_bs = anom["BLOCK_TIME_UTC_mvt_s"] - anom["SCHED_TIME_UTC_mvt_s"]
grp = anom[d_bs.abs() <= TOL].copy()          # BLOCK==SCHED 组
oth = anom[d_bs.abs() > TOL].copy()           # 其他异常
log(f"异常 {len(anom)} 行 = BLOCK==SCHED {len(grp)} + 其他 {len(oth)}；正常样本 {len(norm):,} 行")

def dist(name, s):
    s = pd.Series(s).dropna()
    if not len(s):
        return f"  {name:26s} (空)"
    return (f"  {name:26s} n={len(s):>9,} med={s.median():>8.0f} p25={s.quantile(.25):>8.0f} "
            f"p75={s.quantile(.75):>8.0f} p99={s.quantile(.99):>9.0f} max={s.max():>9.0f}")

log("")
log("=" * 92)
log("(A) 三组的关键差值分布 (单位 s)")
log("=" * 92)
for nm, g in [("BLOCK==SCHED", grp), ("其他异常", oth), ("正常样本", norm)]:
    log(f"\n【{nm}】 n={len(g):,}")
    log(dist("TAXI(=MVT-BLOCK)", g["TAXITIME_SEC_mvt"]))
    log(dist("MVT-AOBT (AOBT当推出)", g["MVT_TIME_UTC_mvt_s"] - g["AOBT_3_flt_s"]))
    log(dist("AOBT-SCHED (推出延误)", g["AOBT_3_flt_s"] - g["SCHED_TIME_UTC_mvt_s"]))
    log(dist("BLOCK-AOBT", g["BLOCK_TIME_UTC_mvt_s"] - g["AOBT_3_flt_s"]))

log("")
log("=" * 92)
log("(B) 裁定: 用 AOBT 当推出时刻, 真实滑行时间是否正常?")
log("=" * 92)
def frac_norm(g, thr=3600):
    """MVT-AOBT 落在正常样本 p99 范围内的比例"""
    v = (g["MVT_TIME_UTC_mvt_s"] - g["AOBT_3_flt_s"]).dropna()
    lo, hi = norm["TAXITIME_SEC_mvt"].quantile(.01), norm["TAXITIME_SEC_mvt"].quantile(.99)
    inr = ((v >= lo) & (v <= hi)).mean() * 100
    in3600 = (v <= 3600).mean() * 100
    return dict(n=int(len(v)), norm_p01=float(lo), norm_p99=float(hi),
                pct_in_norm_range=round(float(inr), 2),
                pct_le_3600s=round(float(in3600), 2))

r_grp, r_oth = frac_norm(grp), frac_norm(oth)
log(f"  正常样本 TAXITIME 1%–99% 区间: [{r_grp['norm_p01']:.0f}, {r_grp['norm_p99']:.0f}] s")
log(f"  BLOCK==SCHED 组: MVT-AOBT 落在该区间 {r_grp['pct_in_norm_range']:.1f}%，≤3600s 占 {r_grp['pct_le_3600s']:.1f}%")
log(f"  其他异常组    : MVT-AOBT 落在该区间 {r_oth['pct_in_norm_range']:.1f}%，≤3600s 占 {r_oth['pct_le_3600s']:.1f}%")

log("")
log("=" * 92)
log("(C) 若用 AOBT 重建标签, 全样本 RMSE 会怎样 (仅诊断, 不改任何文件)")
log("=" * 92)
# 用训练集自身分布做对照: 现有 TAGITIME vs AOBT 重建
d_taxi = norm["TAXITIME_SEC_mvt"]
d_rebuilt = (norm["MVT_TIME_UTC_mvt_s"] - norm["AOBT_3_flt_s"]).dropna()
log(f"  正常样本 TAXI       med={d_taxi.median():.0f}  mean={d_taxi.mean():.0f}")
log(f"  正常样本 MVT-AOBT   med={d_rebuilt.median():.0f}  mean={d_rebuilt.mean():.0f}")
log(f"  两者差 (AOBT-真BLOCK 平均滞后): {(d_rebuilt - d_taxi).median():.0f}s")
n_out = int(((grp["MVT_TIME_UTC_mvt_s"] - grp["AOBT_3_flt_s"]) > BIG).sum())
log(f"  171 行中, 用 AOBT 重建后仍 >10800s 的: {n_out} 行 ({n_out/len(grp)*100:.1f}%)")

RES = {
    "n_anomaly": int(len(anom)), "n_block_eq_sched": int(len(grp)), "n_other": int(len(oth)),
    "n_normal": int(len(norm)),
    "adjudicate_group_bsched": r_grp, "adjudicate_other": r_oth,
    "taxi_median_group_bsched": float(grp["TAXITIME_SEC_mvt"].median()),
    "mvt_minus_aobt_median_group_bsched":
        float((grp["MVT_TIME_UTC_mvt_s"] - grp["AOBT_3_flt_s"]).median()),
    "normal_taxi_p01": float(d_taxi.quantile(.01)),
    "normal_taxi_p99": float(d_taxi.quantile(.99)),
    "n_still_over_10800_after_rebuild": n_out,
    "pct_still_over": round(n_out / len(grp) * 100, 2),
    "verdict": ("BLOCK 被填成 SCHED 导致标签损坏" if r_grp["pct_in_norm_range"] > 50
                else "真实超长滑行"),
}
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
log("")
log(f"裁定: {RES['verdict']}")
log(f"→ {OUT}")
