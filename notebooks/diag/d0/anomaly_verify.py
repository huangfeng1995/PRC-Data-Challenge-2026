"""
anomaly_verify.py — 复核 anomaly_rule_match 的两个自身缺陷 + 深化归因
只读, 不改任何既有文件
  (1) AOBT 等列是否为 9999-12-31 哨兵值 (污染检查4的差值)
  (2) 检查3 是否把"本行自己"误判为"他机" (pool 含自身)
  (3) 171 行 BLOCK==SCHED 的真值性质: 与正常样本对比
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "anomaly_verify.json"
NS = 1_000_000_000
BIG = 10800

def log(*a):
    print(" ".join(str(x) for x in a), flush=True)

COLS = ["MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt",
        "IOBT_flt", "EOBT_1_flt", "ADEP_mvt", "ADEP_flt", "RUNWAY_mvt",
        "STAND_mvt", "FLIGHT_mvt", "MVT_flt" if False else "FLIGHT_ID_mvt"]
parts = []
for p in sorted(glob.glob(str(ROOT / "data/raw/training_2025*.parquet"))):
    parts.append(pd.read_parquet(p, columns=COLS))
dep = pd.concat(parts, ignore_index=True)
dep = dep[dep["PHASE_mvt"] == "DEP"].reset_index(drop=True)

TC = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
      "AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt"]
for c in TC:
    dep[c] = pd.to_datetime(dep[c], utc=True, errors="coerce")
    dep[c + "_s"] = (dep[c].dt.tz_convert("UTC").dt.tz_localize(None)
                     .astype("datetime64[ns]").astype("int64") // NS)

# ---- (1) 哨兵值检测 ----
log("=" * 88)
log("(1) 哨兵值检测 (9999-12-31 = 253402214400s 量级)")
log("-" * 88)
SENT = 253402214400
sent_stats = {}
for c in TC:
    v = dep[c + "_s"]
    n_sent = int((v >= SENT - 86400 * 400).sum())
    frac = n_sent / max(len(v), 1)
    sent_stats[c] = {"n_sentinel_like": n_sent, "frac": round(frac, 5),
                     "max_s": int(np.nanmax(v.values)) if len(v) else None}
    log(f"  {c:22s} 哨兵样行 {n_sent:>9,} ({frac*100:6.3f}%)  max={np.nanmax(v.to_numpy()) if len(v) else 0}")

# 异常行里 AOBT 是否也是哨兵
anom = dep[dep["TAXITIME_SEC_mvt"] > BIG].copy().reset_index(drop=True)
log("")
log(f"  异常行 {len(anom)} 行的 AOBT 哨兵率: "
    f"{int((anom['AOBT_3_flt_s'] >= SENT-86400*400).sum())}/{len(anom)}")

# ---- (2) 检查3 排除自身 ----
log("")
log("=" * 88)
log("(2) 检查3 排除自身后重算 (原实现 pool 含本行, 会把自身误判为他机)")
log("-" * 88)
dep["ADEP_use"] = dep["ADEP_mvt"].fillna(dep["ADEP_flt"])
anom["ADEP_use"] = anom["ADEP_mvt"].fillna(anom["ADEP_flt"])
# 用 (ADEP, MVT_ID) 作为身份, 排除同 MVT_ID 的自身行
CAND = [("AOBT", "AOBT_3_flt_s"), ("LOBT", "LOBT_flt_s"), ("SCHED", "SCHED_TIME_UTC_mvt_s"),
        ("EOBT", "EOBT_1_flt_s"), ("MVT", "MVT_TIME_UTC_mvt_s")]
pool = {}
for cn, cc in CAND:
    ok = dep[cc].notna() & (dep[cc] < SENT - 86400 * 400)
    t = dep.loc[ok, "ADEP_use"]; v = dep.loc[ok, cc].values; ids = dep.loc[ok, "MVT_ID_mvt"].values
    pool[cn] = pd.DataFrame({"ap": t.values, "t": v, "id": ids})

n_self = 0; n_other = 0; other_detail = {}
for _, r in anom.iterrows():
    b = r["BLOCK_TIME_UTC_mvt_s"]; ap = r["ADEP_use"]
    if not np.isfinite(b) or not isinstance(ap, str):
        continue
    hit_self = False; hit_other = None
    for cn, dfp in pool.items():
        if ap not in set(dfp["ap"]):
            continue
        s = dfp[dfp["ap"] == ap]
        d = np.abs(s["t"].values - b)
        j = int(np.argmin(d))
        if d[j] <= 60:
            if s["id"].values[j] == r["MVT_ID_mvt"]:
                hit_self = True
            else:
                hit_other = (cn, int(s["id"].values[j]), int(d[j]))
                break
    if hit_self:
        n_self += 1
    if hit_other:
        n_other += 1
        other_detail.setdefault(hit_other[0], 0)
        other_detail[hit_other[0]] += 1
log(f"  自身 AOBT 命中(假'他机'): {n_self} 行")
log(f"  真·他机命中(排除自身):  {n_other} 行  {other_detail}")

# ---- (3) 171 行 BLOCK==SCHED 的性质 ----
log("")
log("=" * 88)
log("(3) BLOCK==SCHED 组的性质 vs 正常样本")
log("-" * 88)
norm = dep[dep["TAXITIME_SEC_mvt"] <= BIG]
d_bs = anom["BLOCK_TIME_UTC_mvt_s"] - anom["SCHED_TIME_UTC_mvt_s"]
mask_bs = d_bs.abs() <= 60
grp_bs = anom[mask_bs]; grp_oth = anom[~mask_bs]
log(f"  BLOCK==SCHED 组: {len(grp_bs)} 行 | 其他异常: {len(grp_oth)} 行 | 正常样本: {len(norm):,}")
log("")
for nm, g in [("BLOCK==SCHED", grp_bs), ("其他异常", grp_oth), ("正常样本", norm)]:
    if not len(g):
        continue
    mv = g["MVT_TIME_UTC_mvt_s"]; bl = g["BLOCK_TIME_UTC_mvt_s"]; sc = g["SCHED_TIME_UTC_mvt_s"]
    ab = g["AOBT_3_flt_s"]
    ab_ok = ab[ab < SENT - 86400 * 400]
    _t = g["TAXITIME_SEC_mvt"]
    log(f"  {nm:12s} n={len(g):>9,}  TAXI med={_t.median():>8.0f} "
        f"max={_t.max():>7.0f} | MVT-SCHED med={(mv-sc).median():>7.0f} "
        f"| AOBT非哨兵率={len(ab_ok)/max(len(g),1)*100:5.1f}%")

# MVT-SCHED 是否异常大 (真实起飞比计划晚 3 小时+)
log("")
log("  MVT-SCHED 分布对比 (真实起飞比计划晚多久):")
for nm, g in [("BLOCK==SCHED", grp_bs), ("其他异常", grp_oth), ("正常样本", norm)]:
    d = (g["MVT_TIME_UTC_mvt_s"] - g["SCHED_TIME_UTC_mvt_s"])
    d = d[np.isfinite(d)]
    log(f"    {nm:12s} med={d.median():>8.0f}s  p90={d.quantile(.9):>9.0f}s  "
        f"p99={d.quantile(.99):>9.0f}s  max={d.max():>8.0f}s  | >10800s 占比={float((d>10800).mean())*100:5.2f}%")

# 覆盖度: 是否有 >=20 行的明确单一规律
log("")
log("=" * 88)
log("(4) 规则覆盖度 (门槛 >=20 行)")
log("-" * 88)
cov = {"BLOCK==SCHED(±60s)": int(mask_bs.sum()),
       "排除自身后真·他机AOBT==BLOCK": other_detail.get("AOBT", 0)}
for k, v in cov.items():
    log(f"  {k:34s} {v:>4d} 行  {'✓ 达标' if v >= 20 else '✗ 不达标'}")
log(f"  {'占异常行比例':34s} {mask_bs.sum()/len(anom)*100:>4.1f}%")

RES = {"sentinel_stats": sent_stats, "c3_self_matches": n_self,
       "c3_other_matches": n_other, "c3_other_detail": other_detail,
       "block_eq_sched": int(mask_bs.sum()),
       "block_eq_sched_pct": round(float(mask_bs.sum())/len(anom)*100, 2),
       "coverage_ge20": {k: v >= 20 for k, v in cov.items()}}
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"\n→ {OUT}")
