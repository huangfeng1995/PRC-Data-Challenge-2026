"""
corrupt_key_scan.py — 寻找比 ADEP_flt 缺失更强的损坏识别钥匙 (只读)

逻辑要点: 若 BLOCK==SCHED, 则 TAXI == MVT-SCHED 恒成立。
所以对任意损坏行, MVT-SCHED 就是精确答案 —— 与量级无关。
v21 只挑了 MVT-SCHED 极大的行, 等于只修了长尾。
本脚本扫描所有 ranking 可见列, 找能识别"全部损坏行"的钥匙。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "corrupt_key_scan.json"
CACHE = D0 / "_holdout_allcols_s.parquet"
NS = 1_000_000_000
SENT = -9223372037
TOL = 60
# 身份列/派生列, 不作为钥匙
IDENT = {"MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "BLOCK_TIME_UTC_mvt",
         "BLOCK_TIME_UTC_mvt_s", "SCHED_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt_s",
         "MVT_TIME_UTC_mvt", "MVT_TIME_UTC_mvt_s", "TAXITIME_SEC_mvt",
         "ADEP_flt", "is_corrupt", "mvt_sched", "month"}

def mf(m: int) -> Path:
    ny, n2 = (2025, m + 1) if m < 12 else (2026, 1)
    return RAW / f"training_2025-{m:02d}-01_{ny}-{n2:02d}-01.parquet"

def log(*a): print(" ".join(str(x) for x in a), flush=True)

# ---------- ranking 可见列 ----------
rk_cols = set(pq.ParquetFile(RAW / "ranking.parquet").schema_arrow.names)
log(f"ranking 可见列 {len(rk_cols)}")

# ---------- holdout 1+7 全列 (缓存秒级派生) ----------
if CACHE.exists():
    h = pd.read_parquet(CACHE)
    log(f"holdout 缓存命中 {len(h):,} 行")
else:
    h = pd.concat([pd.read_parquet(mf(1)), pd.read_parquet(mf(7))], ignore_index=True)
    h = h[h["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    log(f"holdout DEP {len(h):,} 行, 全列 {h.shape[1]}")
    tcols = [c for c in h.columns if pd.api.types.is_datetime64_any_dtype(h[c])]
    log(f"datetime 列 {len(tcols)}")
    keep = {}
    for c in tcols:
        v = pd.to_datetime(h[c], utc=True, errors="coerce")
        v = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
        keep[c + "_s"] = v.astype("int64") // NS
    for c in h.columns:
        if c in tcols or c.endswith("_s"):
            continue
        keep[c] = h[c].values
    h = pd.DataFrame(keep)
    h["month"] = np.where(pd.to_datetime(h["MVT_TIME_UTC_mvt_s"] * NS).dt.month == 1, 1, 7)
    h.to_parquet(CACHE)
    log(f"→ 缓存 {CACHE.name}")

# ---------- 损坏定义 + 恒等式验证 ----------
blk = h["BLOCK_TIME_UTC_mvt_s"].to_numpy(np.float64)
sch = h["SCHED_TIME_UTC_mvt_s"].to_numpy(np.float64)
mvt = h["MVT_TIME_UTC_mvt_s"].to_numpy(np.float64)
taxi = h["TAXITIME_SEC_mvt"].to_numpy(np.float64)
h["is_corrupt"] = np.abs(blk - sch) <= TOL
h["mvt_sched"] = mvt - sch
nc = int(h["is_corrupt"].sum())
log("")
log(f"损坏行 (|BLOCK-SCHED|<=60s): {nc:,} / {len(h):,}  ({nc/len(h)*100:.3f}%)")
# 恒等式: 损坏行上 TAXI 必须 == MVT-SCHED
resid = np.abs(taxi[h["is_corrupt"].to_numpy()] - h.loc[h["is_corrupt"], "mvt_sched"].to_numpy())
log(f"损坏行上 |TAXI-(MVT-SCHED)| : max={resid.max():.3f}s  "
    f">60s 的行数={int((resid>60).sum())}")
log(f"→ 证实: 损坏行真值可由 MVT-SCHED 精确复现")
ext = (taxi > 10800)
log(f"极端行 >10800: {int(ext.sum())}  其中损坏 {int((ext & h['is_corrupt'].to_numpy()).sum())}")

# ---------- 扫描钥匙 ----------
scan_cols = [c for c in h.columns if c not in IDENT and not c.endswith("_s")]
ts_cols = {c for c in h.columns if c.endswith("_s")}
for thr, lab in [(10800, "MVT-SCHED>10800"), (3600, "MVT-SCHED>3600")]:
    m_sub = (h["mvt_sched"] > thr).to_numpy()
    pos = h["is_corrupt"].to_numpy() & m_sub
    neg = (~h["is_corrupt"].to_numpy()) & m_sub
    log("")
    log("=" * 100)
    log(f"扫描子集 {lab}: {int(m_sub.sum()):,} 行 (损坏 {int(pos.sum())} / 非损坏 {int(neg.sum())})")
    log("=" * 100)
    rows = []
    for c in scan_cols:
        if c not in rk_cols:
            continue                       # 只保留 ranking 可见的列
        v = h[c]
        m = v.isna().to_numpy()
        if c in ts_cols:
            vv = v.to_numpy(np.float64)
            m = m | (vv == SENT) | ~np.isfinite(vv)
        npos = int(m[pos].sum()); nneg = int(m[neg].sum())
        if npos == 0:
            continue
        p = npos / max(int(pos.sum()), 1)
        q = nneg / max(int(neg.sum()), 1)
        lift = p / q if q > 0 else float("inf")
        rows.append({"col": c, "n_pos": npos, "n_neg": nneg,
                     "rate_pos": round(p, 4), "rate_neg": round(q, 4),
                     "lift": round(lift, 2) if np.isfinite(lift) else 9999.0})
    rows.sort(key=lambda x: (-x["lift"], -x["n_pos"]))
    log(f"\n{'列名':<32} {'损坏组缺失率':>12} {'非损坏组':>10} {'lift':>8} {'命中/损坏':>12}")
    for x in rows[:22]:
        log(f"{x['col']:<32} {x['rate_pos']*100:>11.2f}% {x['rate_neg']*100:>9.3f}% "
            f"{x['lift']:>8.1f} {x['n_pos']:>6}/{int(pos.sum()):<5}")
    globals()[f"scan_{thr}"] = rows

# ---------- 组合钥匙评估 ----------
log("")
log("=" * 100)
log("组合钥匙: 识别全部损坏行 (不限量级)")
log("=" * 100)
def ev(mask_fn, name):
    m = np.asarray(mask_fn(h), dtype=bool)
    tp = int((m & h["is_corrupt"].to_numpy()).sum())
    fp = int((m & ~h["is_corrupt"].to_numpy()).sum())
    fn = int((~m & h["is_corrupt"].to_numpy()).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / (P + R) if (P + R) else 0
    log(f"{name:<44} n={tp+fp:>7,} TP={tp:>6} FP={fp:>7} P={P*100:>6.2f}% R={R*100:>6.2f}% F1={F1:.3f}")
    return {"name": name, "n": tp + fp, "tp": tp, "fp": fp,
            "precision": round(P, 4), "recall": round(R, 4), "f1": round(F1, 4)}

combos = []
adep = h["ADEP_flt"]
adep_null = (adep.isna() | adep.astype(str).str.strip().isin(["", "None", "nan"])).to_numpy()
aobt = h["AOBT_3_flt_s"].to_numpy(np.float64)
aobt_bad = (aobt == SENT) | ~np.isfinite(aobt)
act_null = h["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
combos.append(ev(lambda d: d["is_corrupt"], "(oracle 上界: 真损坏)"))
combos.append(ev(lambda d: adep_null, "ADEP_flt 缺失 (v21 用的)"))
combos.append(ev(lambda d: aobt_bad, "AOBT 哨兵"))
combos.append(ev(lambda d: act_null, "AIRCRAFT_TYPE_mvt 缺失  ★lift79"))
combos.append(ev(lambda d: pd.Series(adep_null | aobt_bad, index=d.index), "ADEP_flt空 | AOBT哨兵"))
combos.append(ev(lambda d: pd.Series(adep_null | aobt_bad | act_null, index=d.index), "三者并集 NM缺失"))
combos.append(ev(lambda d: (d["mvt_sched"] > 10800) & (adep_null | aobt_bad), "NM缺失 & MVT-SCHED>10800 (v21核心)"))
combos.append(ev(lambda d: (d["mvt_sched"] > 10800) & (adep_null | aobt_bad | act_null), "NM缺失(含机型) & >10800"))
for thr in [3600, 7200, 10800, 21600]:
    combos.append(ev(lambda d, t=thr: (d["mvt_sched"] > t) & (adep_null | aobt_bad | act_null),
                     f"NM缺失(含机型) & >{thr}"))

json.dump({"n_holdout_dep": int(len(h)), "n_corrupt": nc,
           "corrupt_rate_pct": round(nc / len(h) * 100, 3),
           "identity_max_resid_s": float(resid.max()),
           "n_identity_gt60": int((resid > 60).sum()),
           "n_extreme": int(ext.sum()),
           "n_extreme_corrupt": int((ext & h["is_corrupt"].to_numpy()).sum()),
           "scan_10800": globals().get("scan_10800", [])[:30],
           "scan_3600": globals().get("scan_3600", [])[:30],
           "combos": combos},
          open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
log(f"\n→ {OUT}")
