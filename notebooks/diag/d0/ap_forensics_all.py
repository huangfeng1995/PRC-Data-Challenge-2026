"""
ap_forensics_all.py — 逐机场极端行(>10800)机制取证 (只读)
LIRF 机制已在 v21 真榜验证(363.51)。本脚本查其余机场是否同机制。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "ap_forensics_all.json"
NS = 1_000_000_000
SENT = -9223372037
TOL = 60
EXTREME = 10800

TCOLS = ["MVT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt",
         "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
         "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
         "TAXITIME_SEC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt",
         "ARVT_1_flt", "ARVT_3_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
         "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "ADEP_flt",
         "ADES_FILED_flt", "CALLSIGN_flt"]
TS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
      "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt",
      "ARVT_3_flt", "ADES_FILED_flt"]

def mf(m: int) -> Path:
    ny, n2 = (2025, m + 1) if m < 12 else (2026, 1)
    return RAW / f"training_2025-{m:02d}-01_{ny}-{n2:02d}-01.parquet"

def load(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=TCOLS)
    for c in TS:
        v = pd.to_datetime(d[c], utc=True, errors="coerce")
        d[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
        d[c + "_s"] = d[c].astype("int64") // NS
    return d

def enrich(d: pd.DataFrame) -> pd.DataFrame:
    aobt = d["AOBT_3_flt_s"].to_numpy(np.float64)
    d["aobt_sentinel"] = aobt == SENT
    d["aobt_null"] = d["AOBT_3_flt_s"].isna().to_numpy()
    d["adep_flt_null"] = (d["ADEP_flt"].isna()
                          | d["ADEP_flt"].astype(str).str.strip().isin(["", "None", "nan"]))
    d["nm_missing"] = d["aobt_sentinel"] | d["aobt_null"] | d["adep_flt_null"]
    d["taxi"] = d["MVT_TIME_UTC_mvt_s"] - d["BLOCK_TIME_UTC_mvt_s"]
    d["d_mvt_sched"] = d["MVT_TIME_UTC_mvt_s"] - d["SCHED_TIME_UTC_mvt_s"]
    d["d_block_sched"] = d["BLOCK_TIME_UTC_mvt_s"] - d["SCHED_TIME_UTC_mvt_s"]
    d["d_block_aobt"] = d["BLOCK_TIME_UTC_mvt_s"] - d["AOBT_3_flt_s"]
    d["hour"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"]).dt.hour
    d["flight_pref"] = d["FLIGHT_mvt"].astype(str).str.slice(0, 2)
    d["extreme"] = d["taxi"] > EXTREME
    d["block_eq_sched"] = d["d_block_sched"].abs() <= TOL
    return d

# ================= holdout 1+7 =================
hd = pd.concat([load(mf(m)) for m in (1, 7)], ignore_index=True)
hd["month"] = np.where(hd["MVT_TIME_UTC_mvt"].dt.month == 1, 1, 7)
hd = enrich(hd)
dep = hd[hd["PHASE_mvt"] == "DEP"].reset_index(drop=True)
ext = dep[dep["extreme"]].reset_index(drop=True)
print("=" * 104)
print(f"holdout(1+7) DEP={len(dep):,}   >{EXTREME} 极端行={len(ext)}")
print("=" * 104)

R = {"n_dep": int(len(dep)), "n_ext": int(len(ext)), "by_airport": {}}

print(f"\n{'机场':>6} {'极端n':>6} {'BLOCK==SCHED':>13} {'AOBT哨兵':>10} {'ADEP_flt空':>11} "
      f"{'NM缺失':>8} {'MVT-SCHED中位':>13} {'TAXI中位':>10} {'裁定':>14}")
groups = {}
for a in sorted(ext["ADEP_mvt"].unique()):
    g = ext[ext["ADEP_mvt"] == a]
    allg = dep[dep["ADEP_mvt"] == a]
    groups[a] = g
    nbs = int(g["block_eq_sched"].sum())
    verdict = "SCHED损坏" if nbs / len(g) > 0.5 else "真实极端"
    print(f"{a:>6} {len(g):>6} {nbs:>3}/{len(g):<3} ({nbs/len(g)*100:5.1f}%) "
          f"{int(g['aobt_sentinel'].sum()):>4}/{len(g):<4} {int(g['adep_flt_null'].sum()):>5}/{len(g):<5} "
          f"{int(g['nm_missing'].sum()):>4}/{len(g):<3} {g['d_mvt_sched'].median():>13.0f} "
          f"{g['taxi'].median():>10.0f} {verdict:>14}")
    R["by_airport"][a] = {
        "n_ext": int(len(g)), "n_dep_ap": int(len(allg)),
        "block_eq_sched": nbs, "block_eq_sched_pct": round(nbs / len(g) * 100, 1),
        "aobt_sentinel": int(g["aobt_sentinel"].sum()),
        "adep_flt_null": int(g["adep_flt_null"].sum()),
        "nm_missing": int(g["nm_missing"].sum()),
        "mvt_sched_median": float(g["d_mvt_sched"].median()),
        "mvt_sched_min": int(g["d_mvt_sched"].min()),
        "mvt_sched_max": int(g["d_mvt_sched"].max()),
        "taxi_median": float(g["taxi"].median()),
        "verdict": verdict,
        "flight_pref": {str(k): int(v) for k, v in g["flight_pref"].value_counts().items()},
        "runway": {str(k): int(v) for k, v in g["RUNWAY_mvt"].value_counts().items()},
        "hour": {str(k): int(v) for k, v in g["hour"].value_counts().sort_index().items()},
        "mids": [int(x) for x in g["MVT_ID_mvt"]],
    }

# ================= LFPG 逐行 =================
print("\n" + "=" * 104)
print("LFPG 逐行全部时间列 (3 行)")
print("=" * 104)
lf = groups.get("LFPG")
if lf is not None:
    show = lf[["month", "MVT_ID_mvt", "FLIGHT_mvt", "flight_pref", "AIRCRAFT_TYPE_mvt",
               "RUNWAY_mvt", "STAND_mvt", "taxi", "d_mvt_sched", "d_block_sched",
               "d_block_aobt", "aobt_sentinel", "adep_flt_null", "nm_missing"]].copy()
    for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
              "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]:
        show[c + "_UTC"] = pd.to_datetime(lf[c]).dt.strftime("%Y-%m-%d %H:%M:%S")
    print(show.to_string(index=False, max_colwidth=20))
    for _, r in lf.iterrows():
        print(f"\n  MVT_ID={int(r['MVT_ID_mvt'])}: TAXI={r['taxi']:.0f}s")
        print(f"    MVT-SCHED={r['d_mvt_sched']:.0f}s  BLOCK-SCHED={r['d_block_sched']:.0f}s  "
              f"BLOCK-AOBT={r['d_block_aobt'] if np.isfinite(r['d_block_aobt']) else 'NaN'}")
        print(f"    AOBT哨兵={r['aobt_sentinel']}  ADEP_flt空={r['adep_flt_null']}  "
              f"NM缺失={r['nm_missing']}  跑道={r['RUNWAY_mvt']}  机型={r['AIRCRAFT_TYPE_mvt']}")
    R["lfpg_detail"] = json.loads(lf[["month", "MVT_ID_mvt", "FLIGHT_mvt", "taxi",
                                      "d_mvt_sched", "d_block_sched", "aobt_sentinel",
                                      "adep_flt_null", "nm_missing", "RUNWAY_mvt",
                                      "AIRCRAFT_TYPE_mvt"]].to_json(orient="records"))

# ================= 识别钥匙 precision/recall =================
print("\n" + "=" * 104)
print("识别钥匙搜索 (逐机场, 全 DEP 为底)")
print("=" * 104)
KEYS = {
    "NM缺失": lambda d: d["nm_missing"],
    "ADEP_flt空": lambda d: d["adep_flt_null"],
    "AOBT哨兵": lambda d: d["aobt_sentinel"],
}
for a in sorted(ext["ADEP_mvt"].unique()):
    allg = dep[dep["ADEP_mvt"] == a].reset_index(drop=True)
    g = groups[a]
    pos = allg["extreme"]
    P = int(pos.sum())
    if P == 0:
        continue
    print(f"\n--- {a} (DEP={len(allg):,}, 极端={P}, 基线率={P/len(allg)*100:.3f}%) ---")
    for kn, kf in KEYS.items():
        m = kf(allg)
        n = int(m.sum()); tp = int((m & pos).sum())
        if n == 0:
            continue
        prec = tp / n; rec = tp / P
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {kn:<12} n={n:>6,} TP={tp:>3} P={prec*100:>6.2f}% R={rec*100:>6.2f}% F1={f1:.3f}")
    # 组合 NM缺失 & MVT-SCHED 阈值
    for thr in [10800, 14400, 21600]:
        m = allg["nm_missing"] & (allg["d_mvt_sched"] > thr)
        n = int(m.sum()); tp = int((m & pos).sum())
        if n == 0:
            print(f"  NM缺失&>{thr:<6} n=0")
            continue
        prec = tp / n; rec = tp / P
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  NM缺失&>{thr:<6} n={n:>6,} TP={tp:>3} P={prec*100:>6.2f}% R={rec*100:>6.2f}% F1={f1:.3f}")
        R["by_airport"][a].setdefault("keys", []).append(
            {"key": f"NM缺失&MVT-SCHED>{thr}", "n": n, "tp": tp,
             "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})

# ================= ranking 命中 =================
print("\n" + "=" * 104)
print("2026 ranking 各机场候选命中 (规则 = NM缺失 且 MVT-SCHED > T)")
print("=" * 104)
rk = enrich(load(RAW / "ranking.parquet"))
rd = rk[rk["PHASE_mvt"] == "DEP"].reset_index(drop=True)
print(f"ranking DEP={len(rd):,}")
print(f"\n{'机场':>6} {'DEP':>8} {'NM缺失':>8} {'NM&>10800':>10} {'NM&>14400':>10} "
      f"{'NM&>21600':>10} {'2025极端':>9}")
R["ranking"] = {}
for a in sorted(groups.keys()):
    r = rd[rd["ADEP_mvt"] == a]
    if not len(r):
        continue
    nm = r["nm_missing"]
    row = {"n_dep": int(len(r)), "n_nm": int(nm.sum())}
    for thr in [10800, 14400, 21600]:
        row[f"nm_gt_{thr}"] = int((nm & (r["d_mvt_sched"] > thr)).sum())
    row["n_ext_2025"] = int(len(groups[a]))
    R["ranking"][a] = row
    print(f"{a:>6} {row['n_dep']:>8,} {row['n_nm']:>8,} {row['nm_gt_10800']:>10,} "
          f"{row['nm_gt_14400']:>10,} {row['nm_gt_21600']:>10,} {row['n_ext_2025']:>9}")

# 汇总: 非 LIRF 机场 ranking 潜在改动量
tot = sum(v["nm_gt_14400"] for k, v in R["ranking"].items() if k != "LIRF")
print(f"\n非 LIRF 机场 ranking 命中合计 (T=14400): {tot} 行")
R["ranking_nonlirf_hits_14400"] = int(tot)

json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
print(f"\n→ {OUT}")
