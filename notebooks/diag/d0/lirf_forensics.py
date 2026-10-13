"""
lirf_forensics.py — LIRF 极端滑行行取证 (只读)
第一步: holdout(2025-01+07) LIRF DEP 且 TAXITIME>10800 的逐行机制裁定
第二步: 检验 "航司 × 无NM匹配" 假设, 算 precision/recall + 跨月稳定性
第三步: 2026 ranking LIRF DEP 的可识别性
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "lirf_forensics.json"
NS = 1_000_000_000
TOL = 60
SENT = -9223372037          # 哨兵: 1677-09-21 (int64 下界, 负方向)
EXTREME = 10800
AP = "LIRF"

TCOLS = ["MVT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt",
         "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
         "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
         "TAXITIME_SEC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt",
         "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt",
         "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "ADEP_flt"]
TS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
      "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]

def mf(m: int) -> Path:
    ny, n2 = (2025, m + 1) if m < 12 else (2026, 1)
    return RAW / f"training_2025-{m:02d}-01_{ny}-{n2:02d}-01.parquet"

def load(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=TCOLS)
    for c in TS:
        # R5: 先统一 ns 再取 int64 (原始为 us)
        v = pd.to_datetime(d[c], utc=True, errors="coerce")
        d[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
        d[c + "_s"] = d[c].astype("int64") // NS
    return d

def nm_flags(d: pd.DataFrame) -> pd.DataFrame:
    """NM 匹配代理: _flt 列是否可用。AOBT_3_flt 有哨兵(-9223372037) 视为无 NM。"""
    aobt = d["AOBT_3_flt_s"].to_numpy(np.float64)
    d["aobt_is_sentinel"] = aobt == SENT
    d["aobt_is_null"] = d["AOBT_3_flt_s"].isna().to_numpy()
    d["no_nm"] = d["aobt_is_sentinel"] | d["aobt_is_null"]
    return d

R: dict = {"meta": {"tolerance_s": TOL, "sentinel": SENT, "extreme_thr": EXTREME,
                    "airport": AP, "note": "单机场取证, 非建模"}}

# ================= 第一步 + 第二步: holdout 1+7 月 =================
print("=" * 100)
print(f"第一步 · holdout(2025-01+07) {AP} 极端行取证")
print("=" * 100)
parts = []
for m in (1, 7):
    d = load(mf(m)); d["month"] = m
    parts.append(d[d["ADEP_flt"].astype(str) == AP])
hd = pd.concat(parts, ignore_index=True)
hd = nm_flags(hd)
dep = hd[hd["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
print(f"{AP} DEP 行 (1+7月) = {len(dep):,}")
ext = dep[dep["TAXITIME_SEC_mvt"] > EXTREME].copy().reset_index(drop=True)
print(f"其中 TAXITIME>{EXTREME} = {len(ext)} 行")
R["n_lirf_dep_holdout"] = int(len(dep)); R["n_extreme"] = int(len(ext))

ext["block_eq_sched"] = (ext["BLOCK_TIME_UTC_mvt_s"] - ext["SCHED_TIME_UTC_mvt_s"]).abs() <= TOL
ext["block_eq_aobt"] = (ext["BLOCK_TIME_UTC_mvt_s"] - ext["AOBT_3_flt_s"]).abs() <= TOL
ext["d_mvt_sched"] = ext["MVT_TIME_UTC_mvt_s"] - ext["SCHED_TIME_UTC_mvt_s"]
ext["d_block_sched"] = ext["BLOCK_TIME_UTC_mvt_s"] - ext["SCHED_TIME_UTC_mvt_s"]
ext["d_block_aobt"] = ext["BLOCK_TIME_UTC_mvt_s"] - ext["AOBT_3_flt_s"]
ext["d_mvt_block"] = ext["MVT_TIME_UTC_mvt_s"] - ext["BLOCK_TIME_UTC_mvt_s"]
ext["carrier"] = ext["AIRCRAFT_OPERATOR_flt"].astype("string")
ext["flight_pref"] = ext["FLIGHT_mvt"].astype("string").str.slice(0, 2)
ext["hour"] = pd.to_datetime(ext["MVT_TIME_UTC_mvt"]).dt.hour

print(f"\nBLOCK==SCHED (±{TOL}s): {int(ext['block_eq_sched'].sum())}/{len(ext)}")
print(f"BLOCK==AOBT  (±{TOL}s): {int(ext['block_eq_aobt'].sum())}/{len(ext)}")
print(f"AOBT 哨兵: {int(ext['aobt_is_sentinel'].sum())}/{len(ext)}   "
      f"AOBT 空: {int(ext['aobt_is_null'].sum())}/{len(ext)}   无NM合计: {int(ext['no_nm'].sum())}/{len(ext)}")
print(f"\nMVT-SCHED 分布: min={ext['d_mvt_sched'].min():.0f} "
      f"p25={ext['d_mvt_sched'].quantile(.25):.0f} 中位={ext['d_mvt_sched'].median():.0f} "
      f"p75={ext['d_mvt_sched'].quantile(.75):.0f} max={ext['d_mvt_sched'].max():.0f}")
print(f"BLOCK-SCHED 分布: min={ext['d_block_sched'].min():.0f} 中位={ext['d_block_sched'].median():.0f} max={ext['d_block_sched'].max():.0f}")
print(f"\n机制裁定: TAXITIME = MVT - BLOCK, 恒等式最大误差 = "
      f"{(ext['TAXITIME_SEC_mvt'] - ext['d_mvt_block']).abs().max():.6f}s")
verdict = ("BLOCK==SCHED 标签损坏" if ext["block_eq_sched"].mean() > 0.5
           else "BLOCK 正常 → 真实极端延误")
print(f"→ 裁定: {verdict}")
R["step1"] = {"n_extreme": int(len(ext)),
              "block_eq_sched": int(ext["block_eq_sched"].sum()),
              "block_eq_aobt": int(ext["block_eq_aobt"].sum()),
              "aobt_sentinel": int(ext["aobt_is_sentinel"].sum()),
              "aobt_null": int(ext["aobt_is_null"].sum()),
              "no_nm": int(ext["no_nm"].sum()),
              "mvt_sched_min": int(ext["d_mvt_sched"].min()),
              "mvt_sched_median": float(ext["d_mvt_sched"].median()),
              "mvt_sched_max": int(ext["d_mvt_sched"].max()),
              "identity_max_err": float((ext["TAXITIME_SEC_mvt"] - ext["d_mvt_block"]).abs().max()),
              "verdict": verdict}

print("\n--- 54 行明细 (前 60 行全列) ---")
cols_show = ["month", "FLIGHT_mvt", "carrier", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
             "STAND_mvt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "hour",
             "TAXITIME_SEC_mvt", "d_mvt_sched", "d_block_sched", "no_nm",
             "block_eq_sched", "block_eq_aobt"]
det = ext[cols_show].copy()
det["TAXITIME_h"] = (det["TAXITIME_SEC_mvt"] / 3600).round(2)
print(det.drop(columns=["TAXITIME_SEC_mvt"]).to_string(max_rows=60, index=False))
R["detail"] = det.assign(TAXITIME_SEC_mvt=det["TAXITIME_SEC_mvt"]).to_dict("records")

# ---- 第二步: 假设检验 ----
print("\n" + "=" * 100)
print("第二步 · 检验「航司 × 无NM匹配」")
print("=" * 100)
dep = dep.copy()
dep["carrier"] = dep["AIRCRAFT_OPERATOR_flt"].astype("string")
dep["flight_pref"] = dep["FLIGHT_mvt"].astype("string").str.slice(0, 2)
dep["extreme"] = dep["TAXITIME_SEC_mvt"] > EXTREME

print(f"\n极端行按航司:")
for c, g in ext.groupby("carrier", dropna=False):
    print(f"  {str(c):>8} n={len(g):>3}  无NM={int(g['no_nm'].sum()):>3}  "
          f"BLOCK==SCHED={int(g['block_eq_sched'].sum()):>3}  "
          f"机型={g['AIRCRAFT_TYPE_mvt'].mode().iat[0] if len(g) else '-'}")
print(f"\n极端行按机型: {ext['AIRCRAFT_TYPE_mvt'].value_counts().to_dict()}")
print(f"极端行按跑道: {ext['RUNWAY_mvt'].value_counts().to_dict()}")
print(f"极端行按MARKET: {ext['MARKET_SEGMENT_flt'].value_counts().to_dict()}")
print(f"极端行按FLIGHT_RULE: {ext['FLIGHT_RULE_flt'].value_counts().to_dict()}")
print(f"极端行按小时: {ext['hour'].value_counts().sort_index().to_dict()}")

# 全体 LIRF DEP 的 (carrier, no_nm) 组合
dep["nm_carrier"] = dep["carrier"].astype(str) + "|" + np.where(dep["no_nm"], "noNM", "hasNM")
comb = dep.groupby("nm_carrier").agg(n=("extreme", "size"), n_ext=("extreme", "sum"))
comb["precision"] = comb["n_ext"] / comb["n"]
comb["recall"] = comb["n_ext"] / max(int(dep["extreme"].sum()), 1)
comb = comb.sort_values("n_ext", ascending=False)
print(f"\n{AP} 全部 DEP 按 (航司|无NM) 组合 —— 按极端行数排序 (共 {len(comb)} 组, 显示前 12):")
print(f"{'组合':>28} {'n':>8} {'极端':>6} {'precision':>10} {'recall':>8}")
for k, r in comb.head(12).iterrows():
    print(f"{k:>28} {int(r['n']):>8,} {int(r['n_ext']):>6} {r['precision']*100:>9.2f}% {r['recall']*100:>7.2f}%")
print(f"\n{AP} DEP 总数={len(dep):,} 极端={int(dep['extreme'].sum())}  "
      f"整体极端率={dep['extreme'].mean()*100:.3f}%")
R["step2"] = {"n_lirf_dep": int(len(dep)), "n_ext": int(dep["extreme"].sum()),
              "base_rate_pct": round(float(dep["extreme"].mean() * 100), 3),
              "carrier_extreme": {str(k): int(v) for k, v in ext["carrier"].value_counts(dropna=False).items()},
              "ac_extreme": {str(k): int(v) for k, v in ext["AIRCRAFT_TYPE_mvt"].value_counts().items()},
              "rwy_extreme": {str(k): int(v) for k, v in ext["RUNWAY_mvt"].value_counts().items()},
              "hour_extreme": {str(k): int(v) for k, v in ext["hour"].value_counts().sort_index().items()},
              "combos": [{"combo": k, "n": int(r["n"]), "n_ext": int(r["n_ext"]),
                          "precision": round(float(r["precision"]), 4),
                          "recall": round(float(r["recall"]), 4)} for k, r in comb.head(30).iterrows()]}

# ---- 跨月稳定性 ----
print("\n--- 跨月稳定性 (1月 vs 7月) ---")
for m in (1, 7):
    g = ext[ext["month"] == m]
    gd = dep[dep["month"] == m]
    print(f"  M{m}: 极端 {len(g):>3} 行 / DEP {len(gd):>7,}  极端率={gd['extreme'].mean()*100:.3f}%  "
          f"无NM={int(g['no_nm'].sum())}  BLOCK==SCHED={int(g['block_eq_sched'].sum())}  "
          f"航司={g['carrier'].value_counts().head(3).to_dict()}")
c1 = set(ext[ext["month"] == 1]["carrier"].dropna().astype(str))
c7 = set(ext[ext["month"] == 7]["carrier"].dropna().astype(str))
R["step2"]["month_stability"] = {
    "m1_carriers": sorted(c1), "m7_carriers": sorted(c7),
    "intersection": sorted(c1 & c7), "jaccard": round(len(c1 & c7) / max(len(c1 | c7), 1), 3)}

# ================= 第三步: ranking 可识别性 =================
print("\n" + "=" * 100)
print("第三步 · 2026 ranking LIRF DEP 可识别性")
print("=" * 100)
rk = pd.read_parquet(RAW / "ranking.parquet", columns=TCOLS)
for c in TS:
    v = pd.to_datetime(rk[c], utc=True, errors="coerce")
    rk[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    rk[c + "_s"] = rk[c].astype("int64") // NS
rk = nm_flags(rk)
rd = rk[(rk["ADEP_mvt"].astype(str) == AP) & (rk["PHASE_mvt"] == "DEP")].copy().reset_index(drop=True)
print(f"ranking {AP} DEP = {len(rd):,} 行")
print(f"  BLOCK 非空率 = {rd['BLOCK_TIME_UTC_mvt_s'].notna().mean()*100:.2f}%  "
      f"(确认: ranking 侧 BLOCK 不可用)")
print(f"  TAXITIME 非空率 = {rd['TAXITIME_SEC_mvt'].notna().mean()*100:.2f}%")
print(f"  AOBT 哨兵率 = {rd['aobt_is_sentinel'].mean()*100:.2f}%  "
      f"AOBT 空率 = {rd['aobt_is_null'].mean()*100:.2f}%  无NM = {rd['no_nm'].mean()*100:.2f}%")
rd["carrier"] = rd["AIRCRAFT_OPERATOR_flt"].astype("string")
rd["d_mvt_sched"] = rd["MVT_TIME_UTC_mvt_s"] - rd["SCHED_TIME_UTC_mvt_s"]
rd["hour"] = pd.to_datetime(rd["MVT_TIME_UTC_mvt"]).dt.hour

train_carr = set(ext["carrier"].dropna().astype(str))
rk_carr = set(rd["carrier"].dropna().astype(str))
print(f"\n  2025 极端行航司集合 = {sorted(train_carr)}")
print(f"  ranking {AP} 航司集合 = {sorted(rk_carr)}")
print(f"  交集 = {sorted(train_carr & rk_carr)}")
R["step3"] = {"n_ranking_lirf_dep": int(len(rd)),
              "block_nonnull_pct": round(float(rd["BLOCK_TIME_UTC_mvt_s"].notna().mean() * 100), 2),
              "taxitime_nonnull_pct": round(float(rd["TAXITIME_SEC_mvt"].notna().mean() * 100), 2),
              "aobt_sentinel_pct": round(float(rd["aobt_is_sentinel"].mean() * 100), 2),
              "no_nm_pct": round(float(rd["no_nm"].mean() * 100), 2),
              "mvt_sched_median": float(rd["d_mvt_sched"].median()),
              "mvt_sched_p99": float(rd["d_mvt_sched"].quantile(0.99)),
              "mvt_sched_max": float(rd["d_mvt_sched"].max()),
              "train_extreme_carriers": sorted(train_carr),
              "ranking_carriers": sorted(rk_carr),
              "carrier_intersection": sorted(train_carr & rk_carr)}

# 可见特征能否复现极端: 用 MVT-SCHED 作代理
print("\n--- 用可见特征 (MVT-SCHED 大) 统计 ranking 候选 ---")
print(f"{'阈值':>10} {'ranking命中':>12} {'占比':>9}   参照: 2025同阈值下极端/总")
ref = dep
for thr in [3600, 7200, 10800, 14400]:
    n = int((rd["d_mvt_sched"] > thr).sum())
    hit25 = int((ref["MVT_TIME_UTC_mvt_s"] - ref["SCHED_TIME_UTC_mvt_s"] > thr).sum())
    print(f"{thr:>10,} {n:>12,} {n/max(len(rd),1)*100:>8.2f}%   "
          f"{hit25:,}/{len(ref):,} = {hit25/max(len(ref),1)*100:.2f}%")
R["step3"]["mvt_sched_screen"] = {
    str(t): {"ranking": int((rd["d_mvt_sched"] > t).sum()),
             "ranking_pct": round(float((rd["d_mvt_sched"] > t).mean() * 100), 2),
             "train2025_hits": int(((ref["MVT_TIME_UTC_mvt_s"] - ref["SCHED_TIME_UTC_mvt_s"]) > t).sum()),
             "train2025_pct": round(float(((ref["MVT_TIME_UTC_mvt_s"] - ref["SCHED_TIME_UTC_mvt_s"]) > t).mean() * 100), 2)}
    for t in [3600, 7200, 10800, 14400]}

json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
print(f"\n→ {OUT}")
