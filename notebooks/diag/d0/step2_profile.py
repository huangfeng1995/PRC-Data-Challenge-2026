"""
step2_profile.py — 刻画各类 BLOCK==计划时刻 的匹配行 (只读)

第一��已确认 (训练 DEP 全年 2,085,047):
  only_SCHED=3,397  only_LOBT=0  only_EOBT=12,674  only_IOBT=46  all4=30,298
本步回答: 这些行的 TAXI 是否异常? ADEP_flt 缺失率 / AOBT 哨兵率是否高于基线?
         是否集中在特定机场? 是否对应已知的 LIRF 未覆盖池?

红线: 只读, 不写死机场 (机场只作分组维度观察, 不进任何规则)。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "step2_profile.json"
NS = 1_000_000_000
TARGETS = ["SCHED", "LOBT", "EOBT", "IOBT"]
EXACT_S = 2
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "LOBT_flt", "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt", "TAXITIME_SEC_mvt"]
SRC_COL = {"SCHED": "SCHED_TIME_UTC_mvt", "LOBT": "LOBT_flt",
           "EOBT": "EOBT_1_flt", "IOBT": "IOBT_flt"}
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def to_sec(s):
    v = pd.to_datetime(s, utc=True, errors="coerce")
    v = v.dt.tz_convert("UTC").dt.tz_localize(None)
    out = (v.astype("datetime64[ns]").astype("int64") // NS).astype(np.float64)
    out[(out >= -62135596800) & (out <= -9223372037)] = np.nan
    out[out > 4_102_444_800] = np.nan
    return out


def is_null(s):
    v = s.astype(str).str.strip()
    return s.isna().to_numpy() | v.isin(["", "None", "nan", "NaT"]).to_numpy()


def month_file(m):
    nxt = "2026-01-01" if m == 12 else f"2025-{m + 1:02d}-01"
    p = ROOT / f"data/raw/training_2025-{m:02d}-01_{nxt}.parquet"
    return str(p) if p.exists() else None


def main():
    parts = []
    for m in range(1, 13):
        f = month_file(m)
        if f:
            d = pd.read_parquet(f, columns=COLS)
            d = d[d["PHASE_mvt"] == "DEP"].copy()
            d["month"] = m
            parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    lap(f"训练 DEP: {len(raw):,}")

    blk = to_sec(raw["BLOCK_TIME_UTC_mvt"]); mvt = to_sec(raw["MVT_TIME_UTC_mvt"])
    T = {t: to_sec(raw[SRC_COL[t]]) for t in TARGETS}
    valid = ~np.isnan(blk) & ~np.isnan(mvt)
    diff = {t: np.where(valid & ~np.isnan(T[t]), np.abs(blk - T[t]), np.nan) for t in TARGETS}
    ex = {t: (diff[t] <= EXACT_S) for t in TARGETS}
    TAXI = np.where(valid, mvt - blk, np.nan)
    TAXI_lbl = raw["TAXITIME_SEC_mvt"].to_numpy(float)

    nm = is_null(raw["ADEP_flt"])                        # ADEP_flt 缺失
    aobt = to_sec(raw["AOBT_3_flt"])                     # AOBT 哨兵 (转 datetime 后落在 1970 附近 = 哨兵)
    aobt_sent = np.isnan(aobt)
    ap = raw["ADEP_mvt"].astype(str).to_numpy()
    mo = raw["month"].to_numpy(np.int16)

    # 互斥掩码
    only = {}
    for t in TARGETS:
        oth = np.zeros(len(raw), bool)
        for u in TARGETS:
            if u != t:
                oth |= ex[u]
        only[t] = ex[t] & ~oth
    anym = np.zeros(len(raw), bool)
    for t in TARGETS:
        anym |= ex[t]
    allm = np.ones(len(raw), bool)
    for t in TARGETS:
        allm &= ex[t]
    base = ~anym                                        # 正常行基线

    def prof(m, tag):
        n = int(m.sum())
        if n == 0:
            return {"n": 0}
        tv = TAXI[m]
        return {"n": n,
                "TAXI_p50": round(float(np.nanpercentile(tv, 50)), 1),
                "TAXI_p90": round(float(np.nanpercentile(tv, 90)), 1),
                "TAXI_p99": round(float(np.nanpercentile(tv, 99)), 1),
                "TAXI_max": round(float(np.nanmax(tv)), 1),
                "ADEP_flt_missing_rate": round(float(nm[m].mean()), 4),
                "AOBT_sentinel_rate": round(float(aobt_sent[m].mean()), 4),
                "TAXI_gt10800": int((tv > 10800).sum()),
                "n_airports": int(len(np.unique(ap[m]))),
                "top_airports": {str(a): int(c) for a, c in
                                 pd.Series(ap[m]).value_counts().head(6).items()}}

    res = {"n_dep": int(len(raw)),
           "baseline_normal": prof(base, "base"),
           "exclusive": {f"only_{t}": prof(only[t], t) for t in TARGETS},
           "all4": prof(allm, "all4"),
           "any_match": prof(anym, "any")}

    print("\n" + "=" * 108)
    print("各类匹配的画像 (TAXI = MVT - BLOCK)")
    print("=" * 108)
    hdr = f"{'类别':<14} {'n':>9} {'TAXI p50':>10} {'p90':>9} {'p99':>10} {'max':>11} {'ADEP缺':>8} {'AOBT哨':>8} {'>10800':>7} {'机场数':>7}"
    print(hdr)
    rows = [("正常行基线", res["baseline_normal"]),
            ("only_SCHED", res["exclusive"]["only_SCHED"]),
            ("only_LOBT", res["exclusive"]["only_LOBT"]),
            ("only_EOBT", res["exclusive"]["only_EOBT"]),
            ("only_IOBT", res["exclusive"]["only_IOBT"]),
            ("all4", res["all4"]),
            ("any_match", res["any_match"])]
    for lab, e in rows:
        if e["n"] == 0:
            print(f"{lab:<14} {0:>9,}")
            continue
        print(f"{lab:<14} {e['n']:>9,} {e['TAXI_p50']:>10.1f} {e['TAXI_p90']:>9.1f} "
              f"{e['TAXI_p99']:>10.1f} {e['TAXI_max']:>11.1f} "
              f"{e['ADEP_flt_missing_rate']:>8.2%} {e['AOBT_sentinel_rate']:>8.2%} "
              f"{e['TAXI_gt10800']:>7,} {e['n_airports']:>7,}")

    # ---------- only_EOBT 的机场分布 ----------
    print("\n" + "=" * 108)
    print("only_EOBT 的机场分布 (前 12)")
    print("=" * 108)
    oe = only["EOBT"]
    vc = pd.Series(ap[oe]).value_counts()
    tot = len(oe)
    for a, c in vc.head(12).items():
        sub = oe & (ap == a)
        tv = TAXI[sub]
        print(f"  {a:<6} n={c:>7,} ({100*c/tot:>5.1f}%)  TAXI p50={np.nanmedian(tv):>8.1f} "
              f"p99={np.nanpercentile(tv,99):>9.1f} max={np.nanmax(tv):>10.1f}  "
              f"ADEP缺={nm[sub].mean():>6.1%}  AOBT哨={aobt_sent[sub].mean():>6.1%}")
    res["only_EOBT_by_airport"] = {str(a): {"n": int(c),
                                            "pct_of_only_EOBT": round(100 * c / tot, 2),
                                            "TAXI_p50": round(float(np.nanmedian(TAXI[oe & (ap == a)])), 1),
                                            "TAXI_max": round(float(np.nanmax(TAXI[oe & (ap == a)])), 1),
                                            "ADEP_flt_missing_rate": round(float(nm[oe & (ap == a)].mean()), 4)}
                                   for a, c in vc.head(30).items()}

    # ---------- 与已知 LIRF 未覆盖池的关系 ----------
    print("\n" + "=" * 108)
    print("与已知 LIRF 池的关系")
    print("=" * 108)
    # 已知的 LIRF 损坏行判据: ADEP_flt 缺 + TAXI>10800 (真标签损坏)
    corrupt = nm & (TAXI > 10800)
    print(f"  训练集 ADEP_flt缺 & TAXI>10800 (已知损坏批): {int(corrupt.sum()):,}")
    for lab, m in [("only_SCHED", only["SCHED"]), ("only_EOBT", only["EOBT"]),
                   ("all4", allm), ("any_match", anym)]:
        inter = int((m & corrupt).sum())
        print(f"  {lab:<12} ∩ 损坏批 = {inter:>7,}"
              + (f"  (占该类 {100*inter/max(1,int(m.sum())):.1f}%)" if m.sum() else ""))
    # 已知 340 行池: ADEP_flt 缺 + TAXI 在 (10800, 14400] (未覆盖 T=14400 的部分)
    pool = nm & (TAXI > 10800) & (TAXI <= 14400)
    print(f"\n  ADEP缺 & 10800<TAXI<=14400 (T=14400 未覆盖池): {int(pool.sum()):,}")
    for lab, m in [("only_SCHED", only["SCHED"]), ("only_EOBT", only["EOBT"])]:
        print(f"  {lab:<12} ∩ 未覆盖池 = {int((m & pool).sum()):,}")

    # ---------- 离散聚类检查 (only_EOBT 的 |BLOCK-EOBT| 分布) ----------
    print("\n" + "=" * 108)
    print("only_EOBT 的 |BLOCK-EOBT| 分布 (检查是否离散聚类)")
    print("=" * 108)
    de = diff["EOBT"][oe]
    de = de[np.isfinite(de)]
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]
    print("  分位: " + "  ".join(f"p{q}={np.percentile(de,q):.0f}s" for q in qs))
    for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 60), (60, 300), (300, 1e9)]:
        c = int(((de >= lo) & (de < hi)).sum())
        print(f"    [{lo:>4}, {hi if hi < 1e9 else 'inf'}): {c:>7,}")

    res["diff_EOBT_quantiles"] = {str(q): round(float(np.percentile(de, q)), 1) for q in qs}
    res["overlap_corrupt"] = {lab: int((m & corrupt).sum())
                              for lab, m in [("only_SCHED", only["SCHED"]),
                                             ("only_EOBT", only["EOBT"]),
                                             ("all4", allm), ("any_match", anym)]}
    res["pool_14400"] = int(pool.sum())
    res["overlap_pool"] = {lab: int((m & pool).sum())
                           for lab, m in [("only_SCHED", only["SCHED"]),
                                          ("only_EOBT", only["EOBT"])]}
    # 落盘掩码供第三步用
    np.save(D0 / "_step2_masks.npy", np.array([only[t] for t in TARGETS] + [allm, anym, base, nm, aobt_sent, corrupt, pool]))
    np.save(D0 / "_step2_ids.npy", raw["MVT_ID_mvt"].to_numpy(float))
    np.save(D0 / "_step2_month.npy", mo)
    np.save(D0 / "_step2_ap.npy", ap)
    np.save(D0 / "_step2_taxi.npy", TAXI)
    np.save(D0 / "_step2_mvt.npy", mvt)
    for t in TARGETS:
        np.save(D0 / f"_step2_T_{t}.npy", T[t])
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    lap(f"→ {OUT}")


if __name__ == "__main__":
    main()
