"""
ranking_key_grid.py — ranking 侧钥匙/阈值网格搜索 (只读)

holdout 上 LIRF 满足 v21 规则的只有 21 行, ranking 有 26 行 → 集合不同。
说明 ranking 存在 2025 未见的模式, 需在 ranking 侧直接扫钥匙空间。
输出用于挑"上传后预期收益最大"的规则。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "ranking_key_grid.json"
NS = 1_000_000_000
SENT = -9223372037

def log(*a): print(" ".join(str(x) for x in a), flush=True)

rk = pd.read_parquet(RAW / "ranking.parquet")
log(f"ranking {len(rk):,} 行, {rk.shape[1]} 列")
for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]:
    v = pd.to_datetime(rk[c], utc=True, errors="coerce")
    rk[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    rk[c + "_s"] = rk[c].astype("int64") // NS
rk["d"] = rk["MVT_TIME_UTC_mvt_s"] - rk["SCHED_TIME_UTC_mvt_s"]
dep = rk[rk["PHASE_mvt"] == "DEP"].reset_index(drop=True)
log(f"DEP {len(dep):,}")

def nul(s):
    return s.isna() | s.astype(str).str.strip().isin(["", "None", "nan"])

# ---- 候选钥匙 (全部 ranking 可见) ----
KEYS = {
    "ADEP_flt空": nul(dep["ADEP_flt"]),
    "AIRCRAFT_TYPE_mvt空": nul(dep["AIRCRAFT_TYPE_mvt"]),
    "AOBT哨兵": (dep["AOBT_3_flt_s"].to_numpy(np.float64) == SENT),
    "STAND空": nul(dep["STAND_mvt"]),
    "RUNWAY空": nul(dep["RUNWAY_mvt"]),
    "CALLSIGN空": nul(dep["CALLSIGN_flt"]) if "CALLSIGN_flt" in dep.columns else pd.Series(False, index=dep.index),
    "FLIGHT_ID空": nul(dep["FLIGHT_ID_mvt"]) if "FLIGHT_ID_mvt" in dep.columns else pd.Series(False, index=dep.index),
}
for k in ["ADEP_flt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_flt",
          "MARKET_SEGMENT_flt", "ADES_FILED_flt", "WORKLOAD_flt"]:
    if k not in dep.columns:
        continue
    KEYS[f"{k}空"] = nul(dep[k])

log("\n" + "=" * 96)
log("ranking 各钥匙缺失率 (DEP)")
log("=" * 96)
for k, v in KEYS.items():
    log(f"  {k:<28} {int(v.sum()):>7,}  ({v.sum()/len(dep)*100:>6.3f}%)")

# 等价类合并
groups = {}
for k, v in KEYS.items():
    groups.setdefault(tuple(np.asarray(v, dtype=bool)), []).append(k)
log(f"\n不同缺失模式数: {len(groups)}")
for i, (kk, names) in enumerate(sorted(groups.items(), key=lambda x: -sum(x[0]))):
    log(f"  模式{i}: n={sum(kk):>7,}  = {' | '.join(names)}")

# ---- 网格: 钥匙 × 阈值 × 机场范围 ----
THR = [0, 3600, 7200, 10800, 14400, 18000, 21600, 28800, 36000]
rows = []
base_keys = {
    "NM(ADEP_flt)": KEYS["ADEP_flt空"],
    "NM(机型)": KEYS["ADEP_flt空"] | KEYS["AIRCRAFT_TYPE_mvt空"],
    "机型空": KEYS["AIRCRAFT_TYPE_mvt空"],
}
for scope in ["LIRF", "ALL10"]:
    for kn, km in base_keys.items():
        for t in THR:
            if scope == "LIRF":
                m = km & (dep["ADEP_mvt"].astype(str) == "LIRF") & (dep["d"] > t)
            else:
                m = km & (dep["d"] > t)
            n = int(m.sum())
            if n == 0:
                continue
            sub = dep[m]
            rows.append({
                "scope": scope, "key": kn, "T": t, "n": n,
                "d_med": float(sub["d"].median()),
                "d_min": int(sub["d"].min()), "d_max": int(sub["d"].max()),
                "d_sum": float(sub["d"].sum()),
                "ap": {str(k): int(v) for k, v in sub["ADEP_mvt"].value_counts().head(12).items()},
            })

log("\n" + "=" * 96)
log("网格结果")
log("=" * 96)
log(f"{'范围':>6} {'钥匙':>14} {'T':>7} {'n':>6} {'D中位':>9} {'D和(s)':>13}  主要机场")
for r in sorted(rows, key=lambda x: -(x["n"])):
    aps = ", ".join(f"{k}:{v}" for k, v in list(r["ap"].items())[:4])
    log(f"{r['scope']:>6} {r['key']:>14} {r['T']:>7} {r['n']:>6} {r['d_med']:>9.0f} "
        f"{r['d_sum']:>13,.0f}  {aps}")

json.dump({"n_dep": int(len(dep)), "keys": {k: int(v.sum()) for k, v in KEYS.items()},
           "n_patterns": len(groups), "grid": rows}, open(OUT, "w"),
          ensure_ascii=False, indent=2, default=str)
log(f"\n→ {OUT}")
