"""
corrupt_replay2.py — 损坏行复现后处理实测 (只读, 不改 label)
规则: ADEP_flt 缺失(NM未匹配) 且 MVT-SCHED > T  → 预测替换为 MVT-SCHED
对照: 上一版仅用 MVT-SCHED>T (precision 极低), 本版加入 NM 缺失条件
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "corrupt_replay2.json"
NS = 1_000_000_000
SENT = -9223372037
TOL = 60
BASE_FULL, BASE_CORE = 567.61, 304.94

def mf(m: int) -> Path:
    ny, n2 = (2025, m + 1) if m < 12 else (2026, 1)
    return RAW / f"training_2025-{m:02d}-01_{ny}-{n2:02d}-01.parquet"

TC = ["MVT_ID_mvt", "ADEP_mvt", "ADEP_flt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
      "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"]
raw = pd.concat([pd.read_parquet(mf(m), columns=TC) for m in (1, 7)], ignore_index=True)
raw = raw[raw["PHASE_mvt"] == "DEP"].reset_index(drop=True)
for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]:
    v = pd.to_datetime(raw[c], utc=True, errors="coerce")
    raw[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    raw[c + "_s"] = raw[c].astype("int64") // NS

p = pd.read_parquet(D0 / "p0_holdout_predictions.parquet")
df = p.merge(raw[["MVT_ID_mvt", "ADEP_flt", "AOBT_3_flt_s", "BLOCK_TIME_UTC_mvt_s",
                  "SCHED_TIME_UTC_mvt_s", "MVT_TIME_UTC_mvt_s", "TAXITIME_SEC_mvt"]],
             on="MVT_ID_mvt", how="inner")
assert len(df) == len(p), f"join 行数不一致 {len(df)} vs {len(p)}"
assert (df["TAXI_TRUE_s"] == df["TAXITIME_SEC_mvt"]).all(), "标签不一致"

n = len(df)
y = df["TAXI_TRUE_s"].to_numpy(np.float64)
ens = df["TAXI_PRED_s_ENS"].to_numpy(np.float64)
mo = df["MONTH"].to_numpy()
ap = df["ADEP"].astype(str).to_numpy()
cand = (df["MVT_TIME_UTC_mvt_s"] - df["SCHED_TIME_UTC_mvt_s"]).to_numpy(np.float64)
blk = df["BLOCK_TIME_UTC_mvt_s"].to_numpy(np.float64)
sch = df["SCHED_TIME_UTC_mvt_s"].to_numpy(np.float64)
aobt = df["AOBT_3_flt_s"].to_numpy(np.float64)
adep_flt = df["ADEP_flt"]

# NM 缺失: ADEP_flt 空 或 AOBT 哨兵/空 (两者在 LIRF 极端行上完全等价)
adep_null = adep_flt.isna() | (adep_flt.astype(str).str.strip().isin(["", "None", "nan"]))
aobt_sent = aobt == SENT
aobt_null = np.isnan(aobt)
nm_missing = adep_null.to_numpy() | aobt_sent | aobt_null

# 真损坏定义: BLOCK==SCHED (±60s) → 此时 TAXI_TRUE 恒等于 MVT-SCHED
corrupt = np.isfinite(blk) & (np.abs(blk - sch) <= TOL)
resid = np.abs(cand - y)

print("=" * 100)
print("前置核对")
print("=" * 100)
print(f"holdout n={n:,}   ENS full={np.sqrt(np.mean((ens-y)**2)):.2f} "
      f"core={np.sqrt(np.mean((ens[(y>0)&(y<=10800)]-y[(y>0)&(y<=10800)])**2)):.2f}")
print(f"NM 缺失 (ADEP_flt空 或 AOBT哨兵/空): {int(nm_missing.sum()):,} "
      f"({nm_missing.mean()*100:.2f}%)   [ADEP_flt空={int(adep_null.sum()):,}, "
      f"AOBT哨兵={int(aobt_sent.sum()):,}]")
print(f"真损坏 BLOCK==SCHED: {int(corrupt.sum()):,} ({corrupt.mean()*100:.2f}%)")
print(f"损坏行上 |MVT-SCHED − TRUE| max = {resid[corrupt].max():.1f}s  "
      f"→ 复现值{'精确成立' if resid[corrupt].max()<=TOL else '不精确'}")
print(f"损坏行的 MVT-SCHED: 中位={np.median(cand[corrupt]):.0f}s  "
      f">10800 的仅 {int((cand[corrupt]>10800).sum()):,} 行 "
      f"({(cand[corrupt]>10800).mean()*100:.1f}%)")
print(f"非损坏行的 MVT-SCHED>10800: {int(((cand>10800)&~corrupt).sum()):,} 行")

core_m = (y > 0) & (y <= 10800)
R = lambda pr, m: float(np.sqrt(np.mean((pr[m] - y[m]) ** 2)))
base_ap = {a: R(ens, ap == a) for a in pd.unique(ap)}
BASE_JAN, BASE_JUL = R(ens, mo == 1), R(ens, mo == 7)
BASE_FULL_C = R(ens, np.ones(n, bool)); BASE_CORE_C = R(ens, core_m)
print(f"\n基线核对: full={BASE_FULL_C:.2f} (报 {BASE_FULL})  "
      f"core={BASE_CORE_C:.2f} (报 {BASE_CORE})  Jan={BASE_JAN:.2f} Jul={BASE_JUL:.2f}")

def evaluate(pred, name, extra=None):
    full = R(pred, np.ones(n, bool)); cr = R(pred, core_m)
    pa = {}
    for a in pd.unique(ap):
        m = ap == a
        pa[str(a)] = {"n": int(m.sum()), "full": round(R(pred, m), 2),
                      "base_full": round(base_ap[a], 2),
                      "delta": round(R(pred, m) - base_ap[a], 2)}
    ds = [v["delta"] for v in pa.values()]
    d = {"name": name, "full": round(full, 2), "core": round(cr, 2),
         "d_full": round(full - BASE_FULL_C, 2), "d_core": round(cr - BASE_CORE_C, 2),
         "jan": round(R(pred, mo == 1), 2), "jul": round(R(pred, mo == 7), 2),
         "d_jan": round(R(pred, mo == 1) - BASE_JAN, 2),
         "d_jul": round(R(pred, mo == 7) - BASE_JUL, 2),
         "n_ap_improve": sum(1 for x in ds if x < 0),
         "n_ap_worse": sum(1 for x in ds if x > 0),
         "worst_airport": max(pa.items(), key=lambda kv: kv[1]["delta"])[0],
         "worst_delta": round(max(ds), 2),
         "per_airport": pa}
    if extra:
        d.update(extra)
    return d

res = {"base": {"full": round(BASE_FULL_C, 2), "core": round(BASE_CORE_C, 2),
                "jan": round(BASE_JAN, 2), "jul": round(BASE_JUL, 2),
                "n": n, "n_corrupt": int(corrupt.sum()),
                "n_nm_missing": int(nm_missing.sum())},
       "runs": []}

print("\n" + "=" * 100)
print("A. 仅 LIRF, 规则 = NM缺失 且 MVT-SCHED > T")
print("=" * 100)
print(f"{'T':>8} {'命中':>6} {'判对':>6} {'误伤':>6} {'prec':>8} {'recall':>8} "
      f"{'full':>9} {'Δfull':>9} {'Δcore':>9} {'Jan':>8} {'Jul':>8} {'机场':>9}")
for T in [10800, 14400, 21600]:
    hit = nm_missing & (cand > T) & (ap == "LIRF")
    tp = int((hit & corrupt).sum()); fp = int((hit & ~corrupt).sum())
    fn = int((~hit & corrupt & (ap == "LIRF")).sum())
    nc = int((corrupt & (ap == "LIRF")).sum())
    pr = ens.copy(); pr[hit] = cand[hit]
    r = evaluate(pr, f"LIRF_T{T}",
                 {"T": T, "scope": "LIRF", "n_hit": int(hit.sum()), "tp": tp, "fp": fp,
                  "fn_vs_lirf_corrupt": fn, "n_lirf_corrupt": nc,
                  "precision": round(tp / max(int(hit.sum()), 1), 4),
                  "recall_lirf_corrupt": round(tp / max(nc, 1), 4)})
    res["runs"].append(r)
    print(f"{T:>8,} {r['n_hit']:>6} {tp:>6} {fp:>6} {r['precision']*100:>7.2f}% "
          f"{r['recall_lirf_corrupt']*100:>7.2f}% {r['full']:>9.2f} {r['d_full']:>+9.2f} "
          f"{r['d_core']:>+9.2f} {r['jan']:>8.2f} {r['jul']:>8.2f} "
          f"{r['n_ap_improve']}好/{r['n_ap_worse']}差")

print("\n" + "=" * 100)
print("B. 推广到全部机场, 规则 = NM缺失 且 MVT-SCHED > T")
print("=" * 100)
print(f"{'T':>8} {'命中':>6} {'判对':>6} {'误伤':>6} {'prec':>8} {'recall':>8} "
      f"{'full':>9} {'Δfull':>9} {'Δcore':>9} {'Jan':>8} {'Jul':>8} {'机场':>9}")
for T in [10800, 14400, 21600]:
    hit = nm_missing & (cand > T)
    tp = int((hit & corrupt).sum()); fp = int((hit & ~corrupt).sum())
    pr = ens.copy(); pr[hit] = cand[hit]
    r = evaluate(pr, f"ALL_T{T}",
                 {"T": T, "scope": "ALL", "n_hit": int(hit.sum()), "tp": tp, "fp": fp,
                  "fn": int((~hit & corrupt).sum()),
                  "precision": round(tp / max(int(hit.sum()), 1), 4),
                  "recall": round(tp / max(int(corrupt.sum()), 1), 4)})
    res["runs"].append(r)
    print(f"{T:>8,} {r['n_hit']:>6} {tp:>6} {fp:>6} {r['precision']*100:>7.2f}% "
          f"{r['recall']*100:>7.2f}% {r['full']:>9.2f} {r['d_full']:>+9.2f} "
          f"{r['d_core']:>+9.2f} {r['jan']:>8.2f} {r['jul']:>8.2f} "
          f"{r['n_ap_improve']}好/{r['n_ap_worse']}差")

print("\n" + "=" * 100)
print("C. 对照: 不加 NM 条件 (仅 MVT-SCHED > T), 全部机场")
print("=" * 100)
print(f"{'T':>8} {'命中':>6} {'判对':>6} {'误伤':>6} {'prec':>8} {'full':>9} {'Δfull':>9} {'Δcore':>9}")
for T in [10800, 14400, 21600]:
    hit = cand > T
    tp = int((hit & corrupt).sum()); fp = int((hit & ~corrupt).sum())
    pr = ens.copy(); pr[hit] = cand[hit]
    r = evaluate(pr, f"ALL_noNM_T{T}",
                 {"T": T, "scope": "ALL_noNM", "n_hit": int(hit.sum()), "tp": tp, "fp": fp,
                  "precision": round(tp / max(int(hit.sum()), 1), 4)})
    res["runs"].append(r)
    print(f"{T:>8,} {r['n_hit']:>6} {tp:>6} {fp:>6} {r['precision']*100:>7.2f}% "
          f"{r['full']:>9.2f} {r['d_full']:>+9.2f} {r['d_core']:>+9.2f}")

# ---- 最优组合明细 ----
best = min(res["runs"], key=lambda r: r["full"])
print("\n" + "=" * 100)
print(f"最优: {best['name']}  full={best['full']:.2f} (Δ{best['d_full']:+.2f})  "
      f"core={best['core']:.2f} (Δ{best['d_core']:+.2f})")
print("=" * 100)
T = best["T"]; scope = best["scope"]
if scope == "LIRF":
    hit = nm_missing & (cand > T) & (ap == "LIRF")
else:
    hit = nm_missing & (cand > T)
pr = ens.copy(); pr[hit] = cand[hit]
print(f"命中 {int(hit.sum())} 行 (判对 {int((hit&corrupt).sum())} / 误伤 {int((hit&~corrupt).sum())})")
print("\n--- 命中行明细 ---")
print(f"{'机场':>5} {'月':>3} {'TRUE':>9} {'ENS原值':>9} {'新预测':>10} {'MVT-SCHED':>10} "
      f"{'真损坏':>7} {'旧误差':>10} {'新误差':>10}")
rows = []
for i in np.where(hit)[0]:
    rows.append({"ap": str(ap[i]), "month": int(mo[i]), "true": float(y[i]),
                 "ens": float(ens[i]), "new": float(pr[i]), "cand": float(cand[i]),
                 "corrupt": bool(corrupt[i]), "old_err": float(ens[i] - y[i]),
                 "new_err": float(pr[i] - y[i])})
det = pd.DataFrame(rows).sort_values("true", ascending=False)
print(det.to_string(index=False, max_rows=80))
print("\n--- 误伤行(非损坏却被替换) ---")
fp = det[~det["corrupt"]]
if len(fp):
    print(fp.to_string(index=False, max_rows=40))
    print(f"\n误伤合计 {len(fp)} 行 | 其 SSE 变化 = "
          f"{float((fp['new_err']**2).sum() - (fp['old_err']**2).sum()):.4e}")
print("\n--- 分机场 (最优组合) ---")
for a, v in sorted(best["per_airport"].items(), key=lambda kv: kv[1]["delta"]):
    flag = "  <-- 恶化" if v["delta"] > 0.5 else ""
    print(f"  {a:>5} n={v['n']:>7,} base={v['base_full']:>8.2f} → {v['full']:>8.2f}  "
          f"Δ={v['delta']:>+8.2f}{flag}")
res["best"] = best
res["best_detail"] = det.to_dict("records")
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
print(f"\n→ {OUT}")
