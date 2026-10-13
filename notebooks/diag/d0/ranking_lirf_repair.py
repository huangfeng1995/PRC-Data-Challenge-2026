"""
ranking_lirf_repair.py — ranking 后处理预测生成 (只读推理, 不改 label)

基础: submissions/jubilant-puzzle_v15.parquet (v15 全年训练, 真榜 532 基线)
规则: 仅 DEP 阶段 且 ADEP=='LIRF' 且 ADEP_flt 缺失(哨兵/空) 且 MVT-SCHED > 21600
      → 预测改为 MVT-SCHED
      否则保持原预测。不推广任何其他机场。
输出: 与 v15 提交列结构完全一致的 parquet (344,841 行)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
BASE = ROOT / "submissions" / "jubilant-puzzle_v15.parquet"
OUT = ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair.parquet"
RPT = D0 / "ranking_lirf_repair.json"
NS = 1_000_000_000
SENT = -9223372037
T = 21600.0
AP = "LIRF"

_t0 = 0.0
import time
_t0 = time.time()
def log(*a): print(f"[{time.time()-_t0:6.1f}s]", " ".join(str(x) for x in a), flush=True)

# ---------- 1. 基线提交 ----------
base = pd.read_parquet(BASE)
log(f"基线 {BASE.name}: {len(base):,} 行, 列={list(base.columns)}")
assert list(base.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"], "基线列结构不符"
assert len(base) == 344_841, f"基线行数 {len(base)} != 344841"
pred0 = base["TAXITIME_SEC_mvt"].to_numpy(np.float64).copy()
log(f"  原预测: min={pred0.min():.2f} max={pred0.max():.2f} mean={pred0.mean():.2f}")

# ---------- 2. ranking 可见列 ----------
rk = pd.read_parquet(RAW / "ranking.parquet",
                     columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt",
                              "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"])
log(f"ranking 读取 {len(rk):,} 行")
for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]:
    v = pd.to_datetime(rk[c], utc=True, errors="coerce")
    # R5: 显式统一 ns 再取 int64
    rk[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    rk[c + "_s"] = rk[c].astype("int64") // NS
log("  时间戳已统一为 datetime64[ns] 并取 int64 秒")

# ---------- 3. 规则 ----------
is_dep = (rk["PHASE_mvt"] == "DEP").to_numpy()
is_lirf = (rk["ADEP_mvt"].astype(str) == AP).to_numpy()
adep_flt = rk["ADEP_flt"]
adep_null = (adep_flt.isna() | adep_flt.astype(str).str.strip().isin(["", "None", "nan"])).to_numpy()
aobt_s = rk["AOBT_3_flt_s"].to_numpy(np.float64)
aobt_bad = (aobt_s == SENT) | np.isnan(aobt_s)
nm_missing = adep_null | aobt_bad
cand = (rk["MVT_TIME_UTC_mvt_s"] - rk["SCHED_TIME_UTC_mvt_s"]).to_numpy(np.float64)
cand = np.where(np.isfinite(cand), cand, -np.inf)

hit = is_dep & is_lirf & nm_missing & (cand > T)
log("")
log("规则分解 (ranking):")
log(f"  DEP 行                 : {int(is_dep.sum()):,}")
log(f"  ADEP=='{AP}'            : {int(is_lirf.sum()):,}")
log(f"  DEP 且 {AP}            : {int((is_dep & is_lirf).sum()):,}")
log(f"  ADEP_flt 缺失          : {int(adep_null.sum()):,}  (AOBT 哨兵/空: {int(aobt_bad.sum()):,})")
log(f"  NM 缺失(并集)          : {int(nm_missing.sum()):,}")
log(f"  NM 缺失 且 MVT-SCHED>{int(T)}: {int((nm_missing & (cand > T)).sum()):,}")
log(f"  >>> 最终命中 (DEP+{AP}+NM缺失+阈值): {int(hit.sum()):,} 行")

# ---------- 4. 应用 ----------
rk["_hit"] = hit
rk["_cand"] = cand
m = base.merge(rk[["MVT_ID_mvt", "_hit", "_cand"]], on="MVT_ID_mvt", how="left")
assert len(m) == len(base), f"merge 行数变化 {len(m)} vs {len(base)}"
assert m["_hit"].notna().all(), "有行未匹配到 ranking"
n_hit = int(m["_hit"].sum())
newp = m["TAXITIME_SEC_mvt"].to_numpy(np.float64).copy()
old = newp.copy()
newp[m["_hit"].to_numpy()] = m.loc[m["_hit"], "_cand"].to_numpy(np.float64)
changed = ~np.isclose(newp, old, rtol=0, atol=1e-9)

print()
print("=" * 92)
print("命中行明细")
print("=" * 92)
print(f"{'MVT_ID':>12} {'原预测':>10} {'新预测':>12} {'变化':>12}")
hi = m[m["_hit"]].copy()
hi["旧"] = old[hi.index.to_numpy()] if False else hi["TAXITIME_SEC_mvt"]
for _, r in hi.iterrows():
    print(f"{int(r['MVT_ID_mvt']):>12} {r['TAXITIME_SEC_mvt']:>10.2f} {r['_cand']:>12.1f} "
          f"{r['_cand']-r['TAXITIME_SEC_mvt']:>+12.1f}")

print()
print("=" * 92)
print("改动汇总")
print("=" * 92)
print(f"  命中行数            : {n_hit}")
print(f"  实际数值改变行数    : {int(changed.sum())}")
print(f"  原预测区间          : [{old.min():.2f}, {old.max():.2f}]")
print(f"  新预测区间          : [{newp.min():.2f}, {newp.max():.2f}]")
print(f"  改动行 原预测均值   : {old[changed].mean():.2f}")
print(f"  改动行 新预测均值   : {newp[changed].mean():.2f}")
print(f"  未改动行数          : {int((~changed).sum()):,}")

# 校验: 非命中行必须逐行不变
assert np.array_equal(newp[~changed], old[~changed]), "非命中行被意外修改"
assert len(newp) == 344_841
print("\n✓ 校验通过: 非命中行逐行未变; 总行数 344,841")

# ---------- 5. 落盘 ----------
out = pd.DataFrame({"MVT_ID_mvt": base["MVT_ID_mvt"].to_numpy(),
                    "TAXITIME_SEC_mvt": newp})
assert list(out.columns) == list(base.columns)
assert len(out) == len(base)
out.to_parquet(OUT, index=False)
log(f"\n→ {OUT}  ({OUT.stat().st_size:,} bytes)")

# 回读校验
chk = pd.read_parquet(OUT)
assert len(chk) == 344_841 and list(chk.columns) == list(base.columns)
assert chk["MVT_ID_mvt"].equals(base["MVT_ID_mvt"]), "MVT_ID 顺序不一致"
d = np.abs(chk["TAXITIME_SEC_mvt"].to_numpy() - old)
print(f"✓ 回读校验: {len(chk):,} 行, 与基线差异行数={int((d>1e-9).sum())} (应={n_hit})")

json.dump({"base_file": BASE.name, "out_file": OUT.name,
           "n_rows": int(len(out)), "n_cols": list(out.columns),
           "rule": {"phase": "DEP", "adep": AP, "nm_missing": True, "T": T,
                    "new_pred": "MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt"},
           "n_dep": int(is_dep.sum()), "n_lirf_dep": int((is_dep & is_lirf).sum()),
           "n_adep_flt_null": int(adep_null.sum()), "n_aobt_bad": int(aobt_bad.sum()),
           "n_nm_missing": int(nm_missing.sum()),
           "n_nm_missing_and_thr": int((nm_missing & (cand > T)).sum()),
           "n_hit": n_hit, "n_changed": int(changed.sum()),
           "pred_before": {"min": float(old.min()), "max": float(old.max()),
                           "mean": float(old.mean())},
           "pred_after": {"min": float(newp.min()), "max": float(newp.max()),
                          "mean": float(newp.mean())},
           "hit_rows": [{"MVT_ID": int(r["MVT_ID_mvt"]),
                         "old": float(r["TAXITIME_SEC_mvt"]),
                         "new": float(r["_cand"])} for _, r in hi.iterrows()]},
          open(RPT, "w"), ensure_ascii=False, indent=2)
print(f"→ {RPT}")
