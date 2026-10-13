"""
anomaly_rule_match.py — 训练集 TAXITIME > 10800s 异常行穷尽式错误规则匹配
只读: 不训练、不推理、不碰 ranking、不改任何既有文件
"""
from __future__ import annotations
import sys, json, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
WORK = Path("/Users/openclaw/.trae-cn/work/6ab542cc7e92a50f562d7d7b")
OUT = D0 / "anomaly_rule_match.json"
LOG = WORK / "anomaly_rule_match.log"

TOL = 60          # 容差 ±60s
BIG = 10800       # 异常阈值

def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(m + "\n")

# ---------------- 加载 12 个月 DEP 行 ----------------
FILES = sorted(glob.glob(str(ROOT / "data/raw/training_2025*.parquet")))
COLS = ["MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
        "ADEP_mvt", "ADEP_flt", "RUNWAY_mvt", "STAND_mvt",
        "FLIGHT_mvt", "FLIGHT_ID_mvt", "AIRCRAFT_TYPE_mvt",
        "AIRCRAFT_OPERATOR_flt", "CALLSIGN_flt", "ADES_mvt"]

def load_one(p):
    d = pd.read_parquet(p, columns=COLS)
    d = d[d["PHASE_mvt"] == "DEP"]
    return d

parts = []
for i, p in enumerate(FILES, 1):
    d = load_one(p)
    parts.append(d)
    log(f"  [{i:2d}/12] {Path(p).name}  DEP {len(d):>9,}")
dep = pd.concat(parts, ignore_index=True)
log(f"训练集 DEP 合计 {len(dep):,} 行")

# ---------------- 时间戳统一: 先 ns 再 int64 秒 (R5) ----------------
# 原始是微秒(us), 直接 astype int64 会错 1000 倍
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt"]
NS = 1_000_000_000
for c in TIME_COLS:
    # 原始 us + tz-aware; 必须先转 UTC 再去时区, 才能安全 astype('datetime64[ns]')
    dep[c] = pd.to_datetime(dep[c], utc=True, errors="coerce")
    dep[c + "_s"] = (dep[c].dt.tz_convert("UTC").dt.tz_localize(None)
                     .astype("datetime64[ns]").astype("int64") // NS)

anom = dep[dep["TAXITIME_SEC_mvt"] > BIG].copy().reset_index(drop=True)
log(f"\nTAXITIME > {BIG}s 的 DEP 行: {len(anom)}")
anom["month"] = pd.to_datetime(anom["MVT_TIME_UTC_mvt"], utc=True).dt.month
log("  月份分布: " + "  ".join(f"M{m}={c}" for m, c in
                              anom["month"].value_counts().sort_index().items()))
log(f"  TAXITIME 范围: {anom['TAXITIME_SEC_mvt'].min():.0f} ~ "
    f"{anom['TAXITIME_SEC_mvt'].max():.0f} s  ({(anom['TAXITIME_SEC_mvt'].max()/3600):.1f} 小时)")

# 校验恒等式 TAXITIME = MVT - BLOCK
idt = (anom["MVT_TIME_UTC_mvt_s"] - anom["BLOCK_TIME_UTC_mvt_s"]) - anom["TAXITIME_SEC_mvt"]
log(f"  恒等式 MVT-BLOCK-TAXITIME: max|err| = {idt.abs().max():.6f} s")

# ---------------- 差值 (检查4) ----------------
DIFFS = [("BLOCK-AOBT", "BLOCK_TIME_UTC_mvt_s", "AOBT_3_flt_s"),
         ("BLOCK-LOBT", "BLOCK_TIME_UTC_mvt_s", "LOBT_flt_s"),
         ("BLOCK-EOBT", "BLOCK_TIME_UTC_mvt_s", "EOBT_1_flt_s"),
         ("BLOCK-MVT",  "BLOCK_TIME_UTC_mvt_s", "MVT_TIME_UTC_mvt_s"),
         ("BLOCK-IOBT", "BLOCK_TIME_UTC_mvt_s", "IOBT_flt_s"),
         ("BLOCK-SCHED","BLOCK_TIME_UTC_mvt_s", "SCHED_TIME_UTC_mvt_s")]
for name, a, b in DIFFS:
    anom[name] = anom[a] - anom[b]

# ---------------- 检查1: 单值匹配 ----------------
CAND = [("AOBT", "AOBT_3_flt_s"), ("LOBT", "LOBT_flt_s"), ("IOBT", "IOBT_flt_s"),
        ("EOBT", "EOBT_1_flt_s"), ("SCHED", "SCHED_TIME_UTC_mvt_s"),
        ("MVT", "MVT_TIME_UTC_mvt_s")]
OFFSETS = [0, 86400, -86400, 3600, -3600, 7200, -7200, 10800, -10800,
           21600, -21600, 43200, -43200]

def match_c1(row):
    hits = []
    b = row["BLOCK_TIME_UTC_mvt_s"]
    if not np.isfinite(b):
        return hits
    for cname, ccol in CAND:
        v = row[ccol]
        if not np.isfinite(v):
            continue
        for off in OFFSETS:
            if abs(v + off - b) <= TOL:
                tag = f"{cname}" if off == 0 else f"{cname}{off:+d}s"
                hits.append({"rule": f"BLOCK=={tag}", "target_col": cname, "offset": off,
                             "resid_s": int(v + off - b)})
                break
    return hits

# ---------------- 检查2: 日期错位 (时分秒相同, 日期不同) ----------------
def match_c2(row):
    hits = []
    b = row["BLOCK_TIME_UTC_mvt_s"]
    if not np.isfinite(b):
        return hits
    bts = pd.Timestamp(b, unit="s", tz="UTC")
    for cname, ccol in CAND:
        v = row[ccol]
        if not np.isfinite(v):
            continue
        vts = pd.Timestamp(v, unit="s", tz="UTC")
        if vts.date() != bts.date():
            # 时分秒一致?
            if (vts.hour, vts.minute, vts.second) == (bts.hour, bts.minute, bts.second):
                dd = (bts.date() - vts.date()).days
                hits.append({"rule": f"BLOCK与{cname}时分秒同/日期差{dd}天",
                             "target_col": cname, "date_shift_days": int(dd)})
            # 或者相差整数天 (±1d) 但不是同一天
            elif abs((b - v) % 86400) <= TOL and abs((b - v) / 86400 - round((b - v) / 86400)) < 1e-9:
                dd = round((b - v) / 86400)
                if dd != 0:
                    hits.append({"rule": f"BLOCK-{cname}={dd}整天", "target_col": cname,
                                 "date_shift_days": int(dd)})
    return hits

anom["c1_hits"] = anom.apply(match_c1, axis=1)
anom["c2_hits"] = anom.apply(match_c2, axis=1)

# ---------------- 检查3: 跨航班匹配 ----------------
# 同机场 ±1 天, 全部航班的任一时间列是否等于本行 BLOCK
dep["ADEP_use"] = dep["ADEP_mvt"].fillna(dep["ADEP_flt"])
ap_pool = {}
log("\n[检查3] 构建同机场时间索引 ...")
for cname, ccol in CAND:
    v = dep[ccol]
    ok = v.notna()
    tmp = pd.DataFrame({"ap": dep.loc[ok, "ADEP_use"], "t": v[ok].values})
    ap_pool[cname] = tmp.groupby("ap")["t"].apply(lambda s: np.sort(s.values))
log(f"  索引完成, {len(ap_pool)} 个时间列")

def match_c3(i, row):
    hits = []
    b = row["BLOCK_TIME_UTC_mvt_s"]
    ap = row["ADEP_use"]
    if not np.isfinite(b) or not isinstance(ap, str):
        return hits
    bt = pd.Timestamp(b, unit="s", tz="UTC")
    day0 = bt.floor("D")
    for cname, arr in ap_pool.items():
        if ap not in arr.index:
            continue
        cand_t = arr[ap]
        # 只看 ±1 天窗口内的候选, 减少比较量
        lo = (day0 - pd.Timedelta(days=1)).value // NS
        hi = (day0 + pd.Timedelta(days=2)).value // NS
        sub = cand_t[(cand_t >= lo) & (cand_t < hi)]
        if len(sub) == 0:
            continue
        d = np.abs(sub - b)
        j = int(np.argmin(d))
        if d[j] <= TOL:
            hits.append({"rule": f"他机{cname}==BLOCK", "target_col": cname,
                         "resid_s": int(sub[j] - b)})
            break
    return hits

dep["ADEP_use"] = dep["ADEP_mvt"].fillna(dep["ADEP_flt"])
anom["ADEP_use"] = anom["ADEP_mvt"].fillna(anom["ADEP_flt"])
anom["c3_hits"] = [match_c3(i, r) for i, r in anom.iterrows()]

# ---------------- 组装明细表 ----------------
rows = []
for _, r in anom.iterrows():
    c1 = r["c1_hits"]; c2 = r["c2_hits"]; c3 = r["c3_hits"]
    allhits = [f"1:{h['rule']}" for h in c1] + [f"2:{h['rule']}" for h in c2] + \
              [f"3:{h['rule']}" for h in c3]
    rows.append({
        "MVT_ID": int(r["MVT_ID_mvt"]) if pd.notna(r["MVT_ID_mvt"]) else None,
        "month": int(r["month"]),
        "ADEP": r["ADEP_use"] if isinstance(r["ADEP_use"], str) else None,
        "RUNWAY": r["RUNWAY_mvt"], "STAND": r["STAND_mvt"],
        "FLIGHT": r["FLIGHT_mvt"],
        "operator": (r["AIRCRAFT_OPERATOR_flt"] if isinstance(r["AIRCRAFT_OPERATOR_flt"], str) else None),
        "ACTYPE": r["AIRCRAFT_TYPE_mvt"],
        "TAXI_s": int(r["TAXITIME_SEC_mvt"]),
        "MVT_s": int(r["MVT_TIME_UTC_mvt_s"]) if np.isfinite(r["MVT_TIME_UTC_mvt_s"]) else None,
        "BLOCK_s": int(r["BLOCK_TIME_UTC_mvt_s"]) if np.isfinite(r["BLOCK_TIME_UTC_mvt_s"]) else None,
        "AOBT_s": int(r["AOBT_3_flt_s"]) if np.isfinite(r["AOBT_3_flt_s"]) else None,
        "LOBT_s": int(r["LOBT_flt_s"]) if np.isfinite(r["LOBT_flt_s"]) else None,
        "IOBT_s": int(r["IOBT_flt_s"]) if np.isfinite(r["IOBT_flt_s"]) else None,
        "EOBT_s": int(r["EOBT_1_flt_s"]) if np.isfinite(r["EOBT_1_flt_s"]) else None,
        "SCHED_s": int(r["SCHED_TIME_UTC_mvt_s"]) if np.isfinite(r["SCHED_TIME_UTC_mvt_s"]) else None,
        "d_BLOCK_AOBT": int(r["BLOCK-AOBT"]) if np.isfinite(r["BLOCK-AOBT"]) else None,
        "d_BLOCK_EOBT": int(r["BLOCK-EOBT"]) if np.isfinite(r["BLOCK-EOBT"]) else None,
        "d_BLOCK_IOBT": int(r["BLOCK-IOBT"]) if np.isfinite(r["BLOCK-IOBT"]) else None,
        "d_BLOCK_SCHED": int(r["BLOCK-SCHED"]) if np.isfinite(r["BLOCK-SCHED"]) else None,
        "hit_c1": bool(c1), "hit_c2": bool(c2), "hit_c3": bool(c3),
        "hit_any": bool(allhits),
        "hits": allhits,
    })

# ---------------- 汇总 ----------------
def summ(hits, key):
    from collections import Counter
    c = Counter(h["rule"] for h in hits)
    return dict(c.most_common())

c1_all, c2_all, c3_all = [], [], []
for _, r in anom.iterrows():
    c1_all += r["c1_hits"]; c2_all += r["c2_hits"]; c3_all += r["c3_hits"]

n1 = int(anom["c1_hits"].apply(len).gt(0).sum())
n2 = int(anom["c2_hits"].apply(len).gt(0).sum())
n3 = int(anom["c3_hits"].apply(len).gt(0).sum())
nany = int((n1 > 0) * 0 + sum(1 for _, r in anom.iterrows()
                             if r["c1_hits"] or r["c2_hits"] or r["c3_hits"]))
_anom_any = anom["c1_hits"].apply(len).gt(0) | anom["c2_hits"].apply(len).gt(0) | \
            anom["c3_hits"].apply(len).gt(0)
anom["hit_any"] = _anom_any
none_rows = anom[~_anom_any]

diff_stats = {}
for name, _, _ in DIFFS:
    v = anom[name].dropna()
    if len(v):
        vc = v.value_counts()
        diff_stats[name] = {
            "n": int(len(v)), "min": int(v.min()), "p25": float(v.quantile(.25)),
            "median": float(v.median()), "p75": float(v.quantile(.75)), "max": int(v.max()),
            "top_values": [{"diff_s": int(k), "count": int(c)} for k, c in vc.head(5).items()],
            "max_cluster_count": int(vc.max()),
        }

RES = {
    "task": "anomaly_rule_match_train_taxitime_gt_10800",
    "read_only": True,
    "n_anomaly_rows": int(len(anom)),
    "identity_check_max_abs_err_s": float(idt.abs().max()),
    "timestamp_handling": "先 .astype('datetime64[ns]') 再 //1e9 转秒 (原始 us)",
    "n_hit_c1": n1, "n_hit_c2": n2, "n_hit_c3": n3, "n_hit_any": nany,
    "n_no_match": int(len(none_rows)),
    "c1_rule_summary": summ(c1_all, "rule"),
    "c2_rule_summary": summ(c2_all, "rule"),
    "c3_rule_summary": summ(c3_all, "rule"),
    "diff_stats": diff_stats,
    "no_match_MVT_IDs": [int(x) for x in none_rows["MVT_ID_mvt"].dropna().tolist()],
    "rows": rows,
}
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- 控制台报告 ----------------
log("\n" + "=" * 92)
log(f"汇总: 共 {len(anom)} 行 | 命中C1 {n1} | C2 {n2} | C3 {n3} | 任一 {nany} | 全不匹配 {len(none_rows)}")
log("=" * 92)
log("\n[检查1 规则汇总]")
for k, v in RES["c1_rule_summary"].items():
    log(f"  {v:>3d} 行  {k}")
log("\n[检查2 规则汇总]")
for k, v in RES["c2_rule_summary"].items():
    log(f"  {v:>3d} 行  {k}")
if not RES["c2_rule_summary"]:
    log("  (无命中)")
log("\n[检查3 规则汇总]")
for k, v in RES["c3_rule_summary"].items():
    log(f"  {v:>3d} 行  {k}")
if not RES["c3_rule_summary"]:
    log("  (无命中)")
log("\n[检查4 差值分布]")
for k, s in diff_stats.items():
    log(f"  {k:14s} n={s['n']:>3d} min={s['min']:>8d} p25={s['p25']:>9.0f} "
        f"med={s['median']:>9.0f} p75={s['p75']:>9.0f} max={s['max']:>8d} | "
        f"最大簇 {s['max_cluster_count']} 行")
log(f"\n→ {OUT}")
