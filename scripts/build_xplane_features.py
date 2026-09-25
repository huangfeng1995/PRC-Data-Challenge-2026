"""
基于已入库的 X-Plane apt.dat 衍生几何,产出:
  1. STAND_mvt ↔ apt.dat 机位 匹配报告 (per-airport exact + 模糊候选)
  2. stand → runway 距离 (haversine, 只对精确匹配算)

数据约定:
  - data/external/xplane/runways.csv      (入档)
  - data/external/xplane/stands_geo.csv   (入档)
  - data/external/xplane/taxi_network.json (入档, 仅 edges, 节点坐标空缺 → 不做最短路径)

输出:
  - data/external/xplane/_reports/match_report_<month>.csv
  - data/external/xplane/_reports/stand_rwy_distance_<month>.csv
  - data/external/xplane/_reports/match_summary_<month>.json

口径:
  - 模糊候选使用 difflib.get_close_matches(cutoff=0.6, n=3),只报告不擅自合并。
  - 未匹配距离 = NaN,不取均值/中位数兜底。
  - 距离 = min over 跑道两端点(haversine)。
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XP = REPO / "data" / "external" / "xplane"
RPT = XP / "_reports"
RPT.mkdir(parents=True, exist_ok=True)

EARTH_R_M = 6371008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))


def load_geom() -> tuple[dict, dict, dict]:
    """return stands_by_icao{name->row}, runways_by_icao[name->row], runway_names_by_icao(set)."""
    stands = defaultdict(dict)
    with open(XP / "stands_geo.csv") as f:
        for r in csv.DictReader(f):
            stands[r["icao"]][r["name"]] = {
                "name": r["name"], "lat": float(r["lat"]), "lon": float(r["lon"]),
                "heading": float(r["heading"]) if r["heading"] else None,
                "type": r["type"],
            }
    runways = defaultdict(dict)
    runway_names = defaultdict(set)
    with open(XP / "runways.csv") as f:
        for r in csv.DictReader(f):
            row = {
                "rwy1": r["rwy1"], "rwy2": r["rwy2"],
                "lat1": float(r["lat1"]), "lon1": float(r["lon1"]),
                "lat2": float(r["lat2"]), "lon2": float(r["lon2"]),
                "width_m": float(r["width_m"]),
            }
            key = f"{r['rwy1']}/{r['rwy2']}"
            runways[r["icao"]][key] = row
            runway_names[r["icao"]].add(r["rwy1"])
            runway_names[r["icao"]].add(r["rwy2"])
    return stands, runways, runway_names


def match_for_airport(
    stand_names_data: list[str],
    stand_lookup: dict,
    cutoff: float = 0.6,
    n_cands: int = 3,
) -> tuple[int, int, list[tuple[str, list[str]]]]:
    """return (exact, fuzzy_count, miss_examples).
    fuzzy_count: 未精确匹配但至少 1 个候选 ≥ cutoff.
    """
    exact = 0
    fuzzy = 0
    miss_examples = []
    keys = list(stand_lookup.keys())
    seen = set()
    for s in stand_names_data:
        if s in stand_lookup:
            exact += 1
            continue
        # 模糊
        if s in seen:  # 已报告过这个 miss
            cands = []
        else:
            cands = difflib.get_close_matches(s, keys, n=n_cands, cutoff=cutoff)
            seen.add(s)
        if cands:
            fuzzy += 1
        if len(miss_examples) < 5:
            miss_examples.append((s, cands))
    return exact, fuzzy, miss_examples


def per_row_distance(
    adep: str, stand_mvt: str, runway_mvt: str,
    stands: dict, runways: dict, runway_names: dict,
) -> tuple[float | None, str | None, str | None]:
    """return (distance_m, status, matched_stand_name_or_None).
    status ∈ {"exact","miss","runway_miss","airport_miss"}."""
    if adep not in stands:
        return None, "airport_miss", None
    srow = stands[adep].get(stand_mvt)
    if srow is None:
        return None, "miss", None
    if not runway_mvt or runway_mvt not in runway_names.get(adep, set()):
        return None, "runway_miss", stand_mvt
    # find the row containing this runway endpoint
    best = None
    for row in runways[adep].values():
        if runway_mvt in (row["rwy1"], row["rwy2"]):
            d1 = haversine_m(srow["lat"], srow["lon"], row["lat1"], row["lon1"])
            d2 = haversine_m(srow["lat"], srow["lon"], row["lat2"], row["lon2"])
            cand = min(d1, d2)
            if best is None or cand < best:
                best = cand
    return best, "exact", stand_mvt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="训练月 parquet 路径")
    ap.add_argument("--cutoff", type=float, default=0.6)
    ap.add_argument("--n-cands", type=int, default=3)
    args = ap.parse_args()

    stands, runways, runway_names = load_geom()
    df = pd.read_parquet(args.parquet,
                         columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "STAND_mvt", "RUNWAY_mvt"])
    dep = df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    print(f"载入 {args.parquet}: DEP={len(dep)}", file=sys.stderr)

    month_tag = Path(args.parquet).stem.replace("training_", "")

    # ---------- 1. 匹配报告 ----------
    match_rows = []
    summary = {"month": month_tag, "total_dep": int(len(dep)), "per_icao": {}}
    for icao, sub in dep.groupby("ADEP_mvt"):
        pool = stands.get(icao, {})
        names = list(sub["STAND_mvt"])
        exact, fuzzy, miss = match_for_airport(names, pool, args.cutoff, args.n_cands)
        total = len(sub)
        summary["per_icao"][icao] = {
            "dep_total": total,
            "exact_match": exact,
            "fuzzy_candidate_only": fuzzy,
            "exact_rate": round(exact / total, 4) if total else 0.0,
            "miss_examples": miss,
            "runway_in_data": sorted({str(x) for x in sub["RUNWAY_mvt"] if x is not None}),
            "runway_in_apt": sorted(runway_names.get(icao, set())),
            "runway_extra_in_apt": sorted(runway_names.get(icao, set()) - {str(x) for x in sub["RUNWAY_mvt"] if x is not None}),
        }
        for s, cands in miss:
            match_rows.append({
                "icao": icao, "STAND_mvt": s,
                "fuzzy_candidates": " | ".join(cands),
                "status": "fuzzy_or_miss",
            })

    match_csv = RPT / f"match_report_{month_tag}.csv"
    pd.DataFrame(match_rows).to_csv(match_csv, index=False)
    summary_json = RPT / f"match_summary_{month_tag}.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== 匹配汇总 ===")
    for icao, s in summary["per_icao"].items():
        print(f"  {icao}: exact={s['exact_match']}/{s['dep_total']}  +fuzzy={s['fuzzy_candidate_only']}  rate={s['exact_rate']:.1%}")
    print(f"  -> {match_csv}")
    print(f"  -> {summary_json}")

    # ---------- 2. 距离 ----------
    rows = []
    for r in dep.itertuples(index=False):
        d, status, matched = per_row_distance(
            r.ADEP_mvt, r.STAND_mvt, r.RUNWAY_mvt,
            stands, runways, runway_names,
        )
        rows.append({
            "MVT_ID_mvt": r.MVT_ID_mvt,
            "ADEP_mvt": r.ADEP_mvt,
            "STAND_mvt": r.STAND_mvt,
            "RUNWAY_mvt": r.RUNWAY_mvt,
            "matched_stand": matched,
            "stand_to_rwy_m": None if d is None else round(d, 1),
            "status": status,
        })
    dist_df = pd.DataFrame(rows)
    dist_csv = RPT / f"stand_rwy_distance_{month_tag}.csv"
    dist_df.to_csv(dist_csv, index=False)
    print(f"\n=== 距离 ===  rows={len(dist_df)}  non-null={dist_df['stand_to_rwy_m'].notna().sum()}  "
          f"null(runway_miss)={(dist_df['status']=='runway_miss').sum()}  "
          f"null(stand_miss)={(dist_df['status']=='miss').sum()}")
    print(f"  by airport:")
    for icao, sub in dist_df.groupby("ADEP_mvt"):
        nn = sub["stand_to_rwy_m"].notna().sum()
        print(f"    {icao}: {nn}/{len(sub)} non-null, mean={sub['stand_to_rwy_m'].mean():.0f} m, "
              f"median={sub['stand_to_rwy_m'].median():.0f} m")
    print(f"  -> {dist_csv}")


if __name__ == "__main__":
    main()
