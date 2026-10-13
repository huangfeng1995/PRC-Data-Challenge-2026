"""
lirf_forensics2.py — LIRF 极端滑行行取证 (修正版, 只读)

修正: 上一版用 ADEP_flt 过滤机场, 但 60 行极端行中 56 行 ADEP_flt=None。
      正确键是 ADEP_mvt(整数编码)。本版用 ADEP_mvt 定位 LIRF。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "lirf_forensics2.json"
NS = 1_000_000_000
TOL = 60
SENT = -9223372037
EXTREME = 10800
AP = "LIRF"

TCOLS = ["MVT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt",
         "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
         "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
         "TAXITIME_SEC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt",
         "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt",
         "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "ADEP_flt", "ADES_FILED_flt",
         "CALLSIGN_flt"]
TS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
      "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ADES_FILED_flt"]

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
    d["no_nm"] = d["aobt_sentinel"] | d["aobt_null"]
    # NM/计划数据匹配失败的Overall信号: ADEP_flt / ADES_FILED_flt 缺失
    d["adep_flt_null"] = d["ADEP_flt"].isna() | (d["ADEP_flt"].astype(str).str.strip().isin(["", "None", "nan"]))
    d["ades_filed_null"] = d["ADES_FILED_flt_s"].isna()
    d["nm_missing"] = d["no_nm"] | d["adep_flt_null"]
    d["carrier"] = d["AIRCRAFT_OPERATOR_flt"].astype("string")
    d["flight_pref"] = d["FLIGHT_mvt"].astype("string").str.slice(0, 2)
    d["taxi"] = d["MVT_TIME_UTC_mvt_s"] - d["BLOCK_TIME_UTC_mvt_s"]
    d["d_mvt_sched"] = d["MVT_TIME_UTC_mvt_s"] - d["SCHED_TIME_UTC_mvt_s"]
    d["d_block_sched"] = d["BLOCK_TIME_UTC_mvt_s"] - d["SCHED_TIME_UTC_mvt_s"]
    d["d_block_aobt"] = d["BLOCK_TIME_UTC_mvt_s"] - d["AOBT_3_flt_s"]
    d["hour"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"]).dt.hour
    d["extreme"] = d["taxi"] > EXTREME
    return d

cm = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
# raw 侧 ADEP_mvt 已经是字符串机场码(编码发生在 build_features 之后)
rev = {int(v): k for k, v in cm.get("ADEP_mvt", {}).items()}
ap_code = AP
print(f"ADEP_mvt 原始值即为机场码; LIRF 编码值={rev.get(1027)} (仅记录)")

R = {"meta": {"ap": AP, "apep_code": ap_code, "tol": TOL, "sentinel": SENT,
              "extreme_thr": EXTREME, "note": "单机场数据取证, 非建模"}}

# ================= 载入 holdout 1+7 =================
hd = pd.concat([load(mf(m)) for m in (1, 7)], ignore_index=True)
hd["month"] = np.where(hd["MVT_TIME_UTC_mvt"].dt.month == 1, 1, 7)
hd = enrich(hd)
dep = hd[hd["PHASE_mvt"] == "DEP"].reset_index(drop=True)
lirf = dep[dep["ADEP_mvt"] == ap_code].reset_index(drop=True)
ext = lirf[lirf["extreme"]].reset_index(drop=True)
print(f"holdout DEP={len(dep):,}  LIRF DEP={len(lirf):,}  LIRF 极端={len(ext)}")

print("\n" + "=" * 104)
print("第一步 · 机制裁定")
print("=" * 104)
n_bs = int((ext["d_block_sched"].abs() <= TOL).sum())
n_ba = int((ext["d_block_aobt"].abs() <= TOL).sum())
print(f"BLOCK==SCHED (±{TOL}s) : {n_bs}/{len(ext)}  ({n_bs/len(ext)*100:.1f}%)")
print(f"BLOCK==AOBT  (±{TOL}s) : {n_ba}/{len(ext)}")
print(f"ADEP_flt 缺失          : {int(ext['adep_flt_null'].sum())}/{len(ext)}")
print(f"AOBT 哨兵/空           : {int(ext['aobt_sentinel'].sum())}/{int(ext['aobt_null'].sum())}")
print(f"NM 缺失(no_nm 或 ADEP_flt空): {int(ext['nm_missing'].sum())}/{len(ext)}")
print(f"恒等式 max|TAXI-(MVT-BLOCK)| = {(ext['taxi']-ext['d_mvt_block' if 'd_mvt_block' in ext else 'taxi']).abs().max() if False else (ext['taxi']-(ext['MVT_TIME_UTC_mvt_s']-ext['BLOCK_TIME_UTC_mvt_s'])).abs().max():.6f}")
print(f"\nMVT-SCHED: min={ext['d_mvt_sched'].min():.0f} p25={ext['d_mvt_sched'].quantile(.25):.0f} "
      f"中位={ext['d_mvt_sched'].median():.0f} p75={ext['d_mvt_sched'].quantile(.75):.0f} max={ext['d_mvt_sched'].max():.0f}")
print(f"BLOCK-SCHED: min={ext['d_block_sched'].min():.0f} 中位={ext['d_block_sched'].median():.0f} max={ext['d_block_sched'].max():.0f}")
print(f"\nTAXI 分布: min={ext['taxi'].min():.0f} 中位={ext['taxi'].median():.0f} max={ext['taxi'].max():.0f}")
verdict = "BLOCK==SCHED 标签损坏(推出时刻被计划时刻顶替)" if n_bs / len(ext) > 0.5 else "BLOCK 正常 → 真实极端延误"
print(f"→ 裁定: {verdict}")

print("\n--- 54 行明细 ---")
cs = ["month", "FLIGHT_mvt", "flight_pref", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
      "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "hour", "taxi", "d_mvt_sched",
      "d_block_sched", "adep_flt_null", "no_nm", "aobt_sentinel"]
show = ext[cs].copy()
show["taxi_h"] = (show["taxi"] / 3600).round(1)
show["MVT-SCHED_h"] = (show["d_mvt_sched"] / 3600).round(1)
print(show.drop(columns=["taxi", "d_mvt_sched"]).to_string(max_rows=60, index=False))
R["step1"] = {"n_lirf_dep": int(len(lirf)), "n_ext": int(len(ext)),
              "block_eq_sched": n_bs, "block_eq_aobt": n_ba,
              "adep_flt_null": int(ext["adep_flt_null"].sum()),
              "aobt_sentinel": int(ext["aobt_sentinel"].sum()), "aobt_null": int(ext["aobt_null"].sum()),
              "nm_missing": int(ext["nm_missing"].sum()),
              "taxi_min": int(ext["taxi"].min()), "taxi_median": float(ext["taxi"].median()),
              "taxi_max": int(ext["taxi"].max()),
              "mvt_sched_median": float(ext["d_mvt_sched"].median()),
              "mvt_sched_max": int(ext["d_mvt_sched"].max()),
              "verdict": verdict,
              "carrier_top": {str(k): int(v) for k, v in ext["carrier"].value_counts(dropna=False).head(8).items()},
              "ac": {str(k): int(v) for k, v in ext["AIRCRAFT_TYPE_mvt"].value_counts().items()},
              "rwy": {str(k): int(v) for k, v in ext["RUNWAY_mvt"].value_counts().items()},
              "market": {str(k): int(v) for k, v in ext["MARKET_SEGMENT_flt"].value_counts().items()},
              "flight_rule": {str(k): int(v) for k, v in ext["FLIGHT_RULE_flt"].value_counts().items()},
              "hour": {str(k): int(v) for k, v in ext["hour"].value_counts().sort_index().items()},
              "flight_pref": {str(k): int(v) for k, v in ext["flight_pref"].value_counts().items()}}

# ================= 第二步 =================
print("\n" + "=" * 104)
print("第二步 · 识别条件 precision / recall")
print("=" * 104)
def score(name, mask_l, mask_all, pos_all):
    """mask_l: LIRF 内规则命中; mask_all: 全LIRF; pos_all: LIRF 极端行"""
    tp = int((mask_l & pos_all).sum()); fp = int((mask_l & ~pos_all).sum())
    n = int(mask_l.sum()); P = int(pos_all.sum())
    prec = tp / n if n else 0.0
    rec = tp / P if P else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"{name:>44} n={n:>6,} TP={tp:>4} FP={fp:>6,} P={prec*100:>6.2f}% R={rec*100:>6.2f}% F1={f1:.3f}")
    return {"rule": name, "n": n, "tp": tp, "fp": fp, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}

pos = lirf["extreme"]
base = float(pos.mean())
print(f"{'基准: LIRF 全部 DEP':>44} n={len(lirf):>6,} P={base*100:>6.3f}%  (极端 {int(pos.sum())} 行)")
rules = []
rules.append(score("ADEP_flt 缺失", lirf["adep_flt_null"], lirf, pos))
rules.append(score("AOBT 哨兵", lirf["aobt_sentinel"], lirf, pos))
rules.append(score("no_nm(AOBT哨兵或空)", lirf["no_nm"], lirf, pos))
rules.append(score("NM缺失(AOBT异常 或 ADEP_flt空)", lirf["nm_missing"], lirf, pos))
for thr in [3600, 10800, 14400, 21600]:
    rules.append(score(f"MVT-SCHED > {thr}s", lirf["d_mvt_sched"] > thr, lirf, pos))
rules.append(score("MVT-SCHED > 10800s", lirf["d_mvt_sched"] > 10800, lirf, pos))
rules.append(score("NM缺失 且 MVT-SCHED>10800",
                   lirf["nm_missing"] & (lirf["d_mvt_sched"] > 10800), lirf, pos))
rules.append(score("ADEP_flt缺失 且 MVT-SCHED>10800",
                   lirf["adep_flt_null"] & (lirf["d_mvt_sched"] > 10800), lirf, pos))
# 组合: 逐个放宽
for thr in [10800, 14400, 21600, 28800, 43200]:
    rules.append(score(f"ADEP_flt缺失 且 MVT-SCHED>{thr}",
                       lirf["adep_flt_null"] & (lirf["d_mvt_sched"] > thr), lirf, pos))
R["step2"] = {"base_rate_pct": round(base * 100, 4), "n_ext": int(pos.sum()),
              "rules": rules}

print("\n--- 航司分组(仅极端行) ---")
FL = ext["FLIGHT_mvt"].astype(str).str.slice(0, 2)
for c, g in ext.assign(_pf=FL.values).groupby("_pf"):
    ac = g["AIRCRAFT_TYPE_mvt"].dropna()
    print(f"  {c:>4} n={len(g):>3}  ADEP_flt空={int(g['adep_flt_null'].sum()):>3}  "
          f"无NM={int(g['no_nm'].sum()):>3}  MVT-SCHED中位={g['d_mvt_sched'].median():>8.0f}  "
          f"机型={ac.mode().iat[0] if len(ac) else 'None'}  跑道={g['RUNWAY_mvt'].mode().iat[0]}")

print("\n--- 跨月稳定性 ---")
for m in (1, 7):
    g = lirf[lirf["month"] == m]
    ge = g[g["extreme"]]
    print(f"  M{m}: DEP={len(g):>6,} 极端={len(ge):>3} 极端率={g['extreme'].mean()*100:.3f}%  "
          f"ADEP_flt空={int(ge['adep_flt_null'].sum())}  无NM={int(ge['no_nm'].sum())}  "
          f"BLOCK==SCHED={int((ge['d_block_sched'].abs()<=TOL).sum())}")
# 全 12 月稳定性
print("\n--- 全 12 个月 LIRF 极端行分布(跨月稳定性) ---")
ally = pd.concat([enrich(load(mf(m))) for m in range(1, 13)], ignore_index=True)
ally = ally[(ally["PHASE_mvt"] == "DEP") & (ally["ADEP_mvt"] == ap_code)].reset_index(drop=True)
ally["month"] = pd.to_datetime(ally["MVT_TIME_UTC_mvt"]).dt.month
mon = ally.groupby("month").agg(n=("extreme", "size"), n_ext=("extreme", "sum"),
                                n_adepnull=("adep_flt_null", "sum"),
                                n_nm=("nm_missing", "sum"))
mon["rate_pct"] = (mon["n_ext"] / mon["n"] * 100).round(4)
print(mon.to_string())
R["step2"]["month_stability"] = {str(k): {"n": int(v["n"]), "n_ext": int(v["n_ext"]),
                                          "rate_pct": float(v["rate_pct"]),
                                          "n_adep_flt_null": int(v["n_adepnull"]),
                                          "n_nm_missing": int(v["n_nm"])}
                                 for k, v in mon.iterrows()}

# BLOCK==SCHED 在全 LIRF 的基线率
bs_all = (ally["d_block_sched"].abs() <= TOL)
print(f"\nLIRF 全 12 月 DEP={len(ally):,}  BLOCK==SCHED 基线率={bs_all.mean()*100:.3f}%  "
      f"({int(bs_all.sum()):,} 行)")
print(f"  极端行中 BLOCK==SCHED 率={float((ext['d_block_sched'].abs()<=TOL).mean())*100:.1f}%")
print(f"  非极端行中 BLOCK==SCHED 率={float(bs_all[~ally['extreme']].mean())*100:.3f}%")
R["step2"]["block_eq_sched_baseline_lirf_pct"] = round(float(bs_all.mean() * 100), 3)
R["step2"]["block_eq_sched_extreme_pct"] = round(float((ext["d_block_sched"].abs() <= TOL).mean() * 100), 2)
R["step2"]["block_eq_sched_nonextreme_pct"] = round(float(bs_all[~ally["extreme"]].mean() * 100), 4)
R["step2"]["n_lirf_dep_12m"] = int(len(ally))
R["step2"]["n_ext_12m"] = int(ally["extreme"].sum())

# ================= 第三步: ranking =================
print("\n" + "=" * 104)
print("第三步 · 2026 ranking LIRF 可识别性")
print("=" * 104)
rk = load(RAW / "ranking.parquet")
rk = enrich(rk)
rd = rk[(rk["PHASE_mvt"] == "DEP") & (rk["ADEP_mvt"] == ap_code)].reset_index(drop=True)
print(f"ranking LIRF DEP = {len(rd):,}")
print(f"  BLOCK 非空率     = {rd['BLOCK_TIME_UTC_mvt_s'].notna().mean()*100:.2f}%   ← 确认不可用")
print(f"  TAXITIME 非空率  = {rd['TAXITIME_SEC_mvt'].notna().mean()*100:.2f}%")
print(f"  ADEP_flt 缺失率  = {rd['adep_flt_null'].mean()*100:.2f}%")
print(f"  AOBT 哨兵率      = {rd['aobt_sentinel'].mean()*100:.2f}%   AOBT 空率 = {rd['aobt_null'].mean()*100:.2f}%")
print(f"  NM 缺失率        = {rd['nm_missing'].mean()*100:.2f}%")
print(f"  MVT-SCHED: 中位={rd['d_mvt_sched'].median():.0f} p99={rd['d_mvt_sched'].quantile(.99):.0f} max={rd['d_mvt_sched'].max():.0f}")
print(f"\n  2025 LIRF 参照: ADEP_flt缺失率={ally['adep_flt_null'].mean()*100:.2f}%  "
      f"NM缺失率={ally['nm_missing'].mean()*100:.2f}%  MVT-SCHED p99={ally['d_mvt_sched'].quantile(.99):.0f}")

print(f"\n{'筛选规则':>46} {'ranking命中':>12} {'占比':>8}   2025LIRF同规则命中")
for thr in [10800, 14400, 21600, 28800, 43200]:
    a = int((rd["d_mvt_sched"] > thr).sum())
    b = int(((ally["d_mvt_sched"] > thr)).sum())
    print(f"{'MVT-SCHED>'+str(thr):>46} {a:>12,} {a/len(rd)*100:>7.2f}%   {b:,}/{len(ally):,} ({b/len(ally)*100:.2f}%)")
a = int((rd["adep_flt_null"] & (rd["d_mvt_sched"] > 10800)).sum())
b = int((ally["adep_flt_null"] & (ally["d_mvt_sched"] > 10800)).sum())
print(f"{'ADEP_flt缺失 且 MVT-SCHED>10800':>46} {a:>12,} {a/len(rd)*100:>7.2f}%   {b:,}/{len(ally):,} ({b/len(ally)*100:.2f}%)")
R["step3"] = {"n_ranking_lirf_dep": int(len(rd)),
              "block_nonnull_pct": round(float(rd["BLOCK_TIME_UTC_mvt_s"].notna().mean() * 100), 2),
              "taxitime_nonnull_pct": round(float(rd["TAXITIME_SEC_mvt"].notna().mean() * 100), 2),
              "adep_flt_null_pct": round(float(rd["adep_flt_null"].mean() * 100), 2),
              "aobt_sentinel_pct": round(float(rd["aobt_sentinel"].mean() * 100), 2),
              "nm_missing_pct": round(float(rd["nm_missing"].mean() * 100), 2),
              "mvt_sched_median": float(rd["d_mvt_sched"].median()),
              "mvt_sched_p99": float(rd["d_mvt_sched"].quantile(0.99)),
              "mvt_sched_max": float(rd["d_mvt_sched"].max()),
              "train12m_adep_flt_null_pct": round(float(ally["adep_flt_null"].mean() * 100), 2),
              "train12m_nm_missing_pct": round(float(ally["nm_missing"].mean() * 100), 2),
              "screen": {str(t): {"ranking": int((rd["d_mvt_sched"] > t).sum()),
                                  "ranking_pct": round(float((rd["d_mvt_sched"] > t).mean() * 100), 2),
                                  "train12m": int((ally["d_mvt_sched"] > t).sum())}
                         for t in [10800, 14400, 21600, 28800, 43200]}}
json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
print(f"\n→ {OUT}")
