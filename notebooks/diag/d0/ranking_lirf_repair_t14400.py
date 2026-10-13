"""
ranking_lirf_repair_t14400.py — T=14400 版 ranking 后处理 (只读推理, 不改 label)

基础: submissions/jubilant-puzzle_v15_lirf_repair.parquet (v21, 真榜 363.51)
规则: ADEP=='LIRF' AND ADEP_flt缺失 AND MVT-SCHED > 14400 → 预测改为 MVT-SCHED
      其余行逐行保持 v21 不变
输出: 与 v21 列结构完全一致
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
BASE = ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair.parquet"   # v21
OUT = ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair_t14400.parquet"
RPT = D0 / "ranking_lirf_repair_t14400.json"
NS = 1_000_000_000
T = 14400.0
AP = "LIRF"

_t0 = time.time()
def log(*a): print(f"[{time.time()-_t0:6.1f}s]", " ".join(str(x) for x in a), flush=True)

# ---------- 1. 基线 (v21) ----------
base = pd.read_parquet(BASE)
log(f"基线 {BASE.name}: {len(base):,} 行, 列={list(base.columns)}")
assert list(base.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
assert len(base) == 344_841
v21 = base["TAXITIME_SEC_mvt"].to_numpy(np.float64).copy()
log(f"  v21 预测: min={v21.min():.2f} max={v21.max():.2f} mean={v21.mean():.2f}")

# ---------- 2. ranking 可见列 ----------
rk = pd.read_parquet(RAW / "ranking.parquet",
                     columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt",
                              "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"])
for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]:
    v = pd.to_datetime(rk[c], utc=True, errors="coerce")
    # R5: 显式统一 ns 再取 int64 秒
    rk[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    rk[c + "_s"] = rk[c].astype("int64") // NS
log(f"ranking 读取 {len(rk):,} 行, 时间戳已统一 datetime64[ns]")

# ---------- 3. 规则 ----------
is_dep = (rk["PHASE_mvt"] == "DEP").to_numpy()
is_lirf = (rk["ADEP_mvt"].astype(str) == AP).to_numpy()
adep_flt = rk["ADEP_flt"]
adep_null = (adep_flt.isna() | adep_flt.astype(str).str.strip().isin(["", "None", "nan"])).to_numpy()
aobt_s = rk["AOBT_3_flt_s"].to_numpy(np.float64)
aobt_bad = (aobt_s == -9223372037) | np.isnan(aobt_s)
cand = (rk["MVT_TIME_UTC_mvt_s"] - rk["SCHED_TIME_UTC_mvt_s"]).to_numpy(np.float64)
cand = np.where(np.isfinite(cand), cand, -np.inf)

hit = is_dep & is_lirf & adep_null & (cand > T)
log("")
log(f"规则分解 (T={int(T)}):")
log(f"  DEP 且 {AP}            : {int((is_dep & is_lirf).sum()):,}")
log(f"  ADEP_flt 缺失          : {int(adep_null.sum()):,}")
log(f"  ADEP_flt空 与 AOBT哨兵一致: {bool((adep_null == aobt_bad).all())}")
log(f"  LIRF 内 ADEP_flt缺     : {int((is_dep & is_lirf & adep_null).sum()):,}")
log(f"  >>> 命中 (加阈值>{int(T)}) : {int(hit.sum()):,} 行")

# ---------- 4. 应用 ----------
rk["_hit"] = hit
rk["_cand"] = cand
m = base.merge(rk[["MVT_ID_mvt", "_hit", "_cand"]], on="MVT_ID_mvt", how="left")
assert len(m) == len(base)
assert m["_hit"].notna().all()

n_hit = int(m["_hit"].sum())
newp = v21.copy()
hitmask = m["_hit"].to_numpy()
newp[hitmask] = m.loc[hitmask, "_cand"].to_numpy(np.float64)
changed = ~np.isclose(newp, v21, rtol=0, atol=1e-9)

print()
print("=" * 92)
print("新增命中行 (v21 未改 → 本次改)")
print("=" * 92)
prev = pd.read_parquet(ROOT / "submissions" / "jubilant-puzzle_v15_lirf_repair.parquet")
hit_ids = set(m.loc[hitmask, "MVT_ID_mvt"].astype(int))
v21_map = dict(zip(base["MVT_ID_mvt"].astype(int), v21))
new_map = dict(zip(base["MVT_ID_mvt"].astype(int), newp))
added = [i for i in hit_ids if not np.isclose(v21_map[i], min(
    [v21_map[i]], default=0), atol=1e-9)]
# 真正的"新增"= 本次命中但 v21 预测 != MVT-SCHED (即 v21 未改过的行)
newly = [i for i in sorted(hit_ids)
         if abs(new_map[i] - v21_map[i]) > 1e-9]
print(f"本次命中 {n_hit} 行 | 相对 v21 实际新增改动 {len(newly)} 行")
print(f"\n{'MVT_ID':>12} {'v21原值':>10} {'T14400新值':>12} {'增量':>11}")
for i in newly:
    print(f"{i:>12} {v21_map[i]:>10.2f} {new_map[i]:>12.1f} {new_map[i]-v21_map[i]:>+11.1f}")

print()
print("=" * 92)
print("汇总")
print("=" * 92)
print(f"  命中行数 (LIRF+ADEP_flt缺+>{int(T)}) : {n_hit}")
print(f"  相对 v21 新增改动行数              : {len(newly)}")
print(f"  相对 v21 总改动行数                : {int(changed.sum())}")
print(f"  v21   区间 : [{v21.min():.2f}, {v21.max():.2f}]  mean={v21.mean():.2f}")
print(f"  T14400 区间: [{newp.min():.2f}, {newp.max():.2f}]  mean={newp.mean():.2f}")
print(f"  逐行未变行数: {int((~changed).sum()):,}")

assert np.array_equal(newp[~changed], v21[~changed]), "非命中行被意外修改"
assert len(newp) == 344_841
print("\n✓ 校验通过: 非命中行与 v21 逐行完全一致; 总行数 344,841")

# ---------- 5. 落盘 ----------
out = pd.DataFrame({"MVT_ID_mvt": base["MVT_ID_mvt"].to_numpy(),
                    "TAXITIME_SEC_mvt": newp})
assert list(out.columns) == list(base.columns)
out.to_parquet(OUT, index=False)
log(f"\n→ {OUT}  ({OUT.stat().st_size:,} bytes)")

chk = pd.read_parquet(OUT)
assert len(chk) == 344_841 and list(chk.columns) == list(base.columns)
assert chk["MVT_ID_mvt"].equals(base["MVT_ID_mvt"]), "MVT_ID 顺序不一致"
d = np.abs(chk["TAXITIME_SEC_mvt"].to_numpy() - v21)
print(f"✓ 回读校验: {len(chk):,} 行, 与 v21 差异 {int((d>1e-9).sum())} 行 (应={len(newly)})")
print(f"  低于30的行数: {int((chk['TAXITIME_SEC_mvt']<30).sum())}  NaN: {int(chk['TAXITIME_SEC_mvt'].isna().sum())}")

json.dump({"base_file": BASE.name, "out_file": OUT.name,
           "n_rows": int(len(out)), "n_cols": list(out.columns),
           "rule": {"adep": AP, "cond": "ADEP_flt 缺失", "T": T,
                    "new_pred": "MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt"},
           "n_dep_lirf": int((is_dep & is_lirf).sum()),
           "n_adep_flt_null": int(adep_null.sum()),
           "adep_null_equals_aobt_bad": bool((adep_null == aobt_bad).all()),
           "n_hit": n_hit, "n_newly_changed_vs_v21": len(newly),
           "n_changed_vs_v21": int(changed.sum()),
           "v21_stats": {"min": float(v21.min()), "max": float(v21.max()), "mean": float(v21.mean())},
           "new_stats": {"min": float(newp.min()), "max": float(newp.max()), "mean": float(newp.mean())},
           "newly_changed": [{"MVT_ID": int(i), "v21": float(v21_map[i]), "new": float(new_map[i])}
                             for i in newly]},
          open(RPT, "w"), ensure_ascii=False, indent=2)
print(f"→ {RPT}")
