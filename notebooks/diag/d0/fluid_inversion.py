"""
fluid_inversion.py — 流体队列反演可行性验证 (holdout 只读, 不训练)
对每条离港跑道做一维队列反演, 与 P0 ENS(LGB) 在各滑行分档对比 RMSE。
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "fluid_inversion.json"
NS = 1_000_000_000
QUEUE_GAP = 300.0          # 队列清空判定阈值 (s)
AP = "EGLL"                # 本次主机场 (脚本参数可覆盖)

def mf(m: int) -> Path:
    ny, n2 = (2025, m + 1) if m < 12 else (2026, 1)
    return RAW / f"training_2025-{m:02d}-01_{ny}-{n2:02d}-01.parquet"

def log(*a): print(" ".join(str(x) for x in a), flush=True)

# ---------- holdout 预测 + 真值 + ranking 侧可见列 ----------
p = pd.read_parquet(D0 / "p0_holdout_predictions.parquet")
TC = ["MVT_ID_mvt", "ADEP_mvt", "ADEP_flt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
      "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "RUNWAY_mvt",
      "AIRCRAFT_TYPE_mvt", "STAND_mvt"]
raw = pd.concat([pd.read_parquet(mf(m), columns=TC) for m in (1, 7)], ignore_index=True)
raw = raw[raw["PHASE_mvt"] == "DEP"].reset_index(drop=True)
for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]:
    v = pd.to_datetime(raw[c], utc=True, errors="coerce")
    raw[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    raw[c + "_s"] = raw[c].astype("int64") // NS

# lookup_t_unimp 需要 PHASE_mvt / ADEP_mvt / STAND_mvt / RUNWAY_mvt, 必须全部保留
NEED = ["MVT_ID_mvt", "ADEP_mvt", "ADEP_flt", "PHASE_mvt", "RUNWAY_mvt",
        "STAND_mvt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt_s",
        "SCHED_TIME_UTC_mvt_s", "BLOCK_TIME_UTC_mvt_s"]
df = p.merge(raw[NEED], on="MVT_ID_mvt", how="inner")
assert len(df) == len(p), f"join {len(df)} != {len(p)}"
for _c in ["PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]:
    assert _c in df.columns, f"缺列 {_c}"

# ---------- t_unimp_r7: 用训练月 LUT (只用训练折, 不含 1/7) ----------
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
import _r7_lib as L
from src.data_loader import load_month, parse_time_columns
from v15_clean_retrain import TRAIN_COLS, TIME_COLS
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
LUT_C = D0 / "_fluid_unimp_lut.pkl"
if LUT_C.exists():                      # 缓存复用, 避免重复拟合 350 万行
    with open(LUT_C, "rb") as f:
        lut = pickle.load(f)
    log(f"LUT 命中缓存 {LUT_C.name} (训练月 {TRAIN_MONTHS}, 不含 1/7)")
else:
    tr_raw = pd.concat([pd.read_parquet(mf(m), columns=TRAIN_COLS) for m in TRAIN_MONTHS],
                       ignore_index=True)
    tr_p = parse_time_columns(tr_raw.copy(), cols=TIME_COLS)
    lut = L.build_t_unimp_lut(tr_p[(tr_p["PHASE_mvt"] == "DEP")])
    with open(LUT_C, "wb") as f:
        pickle.dump(lut, f)
    log(f"LUT 用 {len(tr_p):,} 训练月行拟合并缓存 (不含 1/7 月)")
t_unimp, _ = L.lookup_t_unimp(df, lut)
df["t_unimp"] = t_unimp
df["MVT_s"] = df["MVT_TIME_UTC_mvt_s"]
# 校验恒等式
taxi_true = df["MVT_s"].to_numpy(float) - df["BLOCK_TIME_UTC_mvt_s"].to_numpy(float)
assert np.abs(taxi_true - df["TAXI_TRUE_s"].to_numpy(float)).max() < 1e-6
log(f"恒等式校验通过 max|Δ|={np.abs(taxi_true - df['TAXI_TRUE_s'].to_numpy(float)).max():.6f}")

df["adep_flt_null"] = (df["ADEP_flt"].isna()
                      | df["ADEP_flt"].astype(str).str.strip().isin(["", "None", "nan"]))
df["d_mvt_sched"] = df["MVT_s"] - df["SCHED_TIME_UTC_mvt_s"]
df["corrupt"] = df["adep_flt_null"] & (df["d_mvt_sched"] > 14400)

# ---------- 队列反演 (逐跑道) ----------
def invert_queue(g: pd.DataFrame) -> pd.DataFrame:
    """对单跑道: 按 MVT 排序, 用 <300s 的相邻间隔判队列连续性, 算排位 k"""
    g = g.sort_values("MVT_s", kind="stable")
    mv = g["MVT_s"].to_numpy(float)
    if len(mv) < 2:
        g = g.copy(); g["k"] = 0; g["svc"] = np.nan; g["pred_phys"] = g["t_unimp"].to_numpy(float)
        return g
    gaps = np.diff(mv)                       # gap[i] = mv[i+1] - mv[i]
    tight = gaps[gaps < QUEUE_GAP]
    s = float(np.median(tight)) if len(tight) else QUEUE_GAP
    # k[i] = 从 i 往前数, 连续 tight 间隔的个数
    k = np.zeros(len(mv), dtype=np.int64)
    cur = 0
    for i in range(1, len(mv)):
        if gaps[i - 1] < QUEUE_GAP:
            cur += 1
        else:
            cur = 0
        k[i] = cur
    g = g.copy()
    g["k"] = k
    g["svc"] = s
    g["pred_phys"] = g["t_unimp"].to_numpy(float) + k * s
    return g

BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, float("inf"))]
BLAB = ["[0,600)", "[600,1800)", "[1800,3600)", "[3600,10800]", ">10800"]

def run_airport(ap: str, exclude_corrupt: bool = True):
    sub = df[df["ADEP_mvt"] == ap].copy()
    if exclude_corrupt:
        n_before = len(sub)
        sub = sub[~sub["corrupt"]]
        log(f"\n{ap}: DEP {n_before:,} → 排除损坏行 {n_before-len(sub):,} → {len(sub):,}")
    else:
        log(f"\n{ap}: DEP {len(sub):,} (未排除损坏行)")
    # 逐跑道反演
    parts = []
    for rw, g in sub.groupby("RUNWAY_mvt", dropna=True):
        if len(g) < 5:
            gg = g.copy(); gg["k"] = 0; gg["svc"] = np.nan
            gg["pred_phys"] = gg["t_unimp"].to_numpy(float); parts.append(gg)
        else:
            parts.append(invert_queue(g))
    r = pd.concat(parts, ignore_index=True)
    y = r["TAXI_TRUE_s"].to_numpy(float)
    lgb = r["TAXI_PRED_s_ENS"].to_numpy(float)
    phy = np.clip(r["pred_phys"].to_numpy(float), 30.0, 10800.0)
    log(f"  跑道数={r['RUNWAY_mvt'].nunique()}  服务间隔 s 中位={r['svc'].median():.1f}s  "
        f"排位k: 中位={r['k'].median():.0f} p95={r['k'].quantile(.95):.0f} max={r['k'].max():.0f}")
    rows = []
    for (lo, hi), lab in zip(BINS, BLAB):
        m = (y >= lo) & (y < hi) if hi != float("inf") else (y >= lo)
        if m.sum() < 5:
            rows.append({"bin": lab, "n": int(m.sum()), "phys_rmse": None,
                         "lgb_rmse": None, "delta": None, "pct": None,
                         "bias_phys": None, "bias_lgb": None}); continue
        rp = float(np.sqrt(np.mean((phy[m] - y[m]) ** 2)))
        rl = float(np.sqrt(np.mean((lgb[m] - y[m]) ** 2)))
        rows.append({"bin": lab, "n": int(m.sum()), "phys_rmse": round(rp, 2),
                     "lgb_rmse": round(rl, 2), "delta": round(rp - rl, 2),
                     "pct": round((rp - rl) / rl * 100, 2),
                     "bias_phys": round(float(np.mean(phy[m] - y[m])), 2),
                     "bias_lgb": round(float(np.mean(lgb[m] - y[m])), 2)})
    corr = float(np.corrcoef(phy, lgb)[0, 1])
    # 长滑行段门槛
    key = next((x for x in rows if x["bin"] == "[1800,3600)"), None)
    verdict = None
    if key and key["phys_rmse"] is not None:
        verdict = ("继续精细化" if key["pct"] <= -20 else
                   ("放弃" if key["pct"] >= -10 else "不确定(介于-20%~-10%)"))
    log(f"\n  {'档':>14} {'n':>8} {'物理RMSE':>11} {'LGB RMSE':>11} {'Δ':>10} {'Δ%':>9} "
        f"{'bias_phys':>11} {'bias_lgb':>10}")
    for x in rows:
        if x["phys_rmse"] is None:
            log(f"  {x['bin']:>14} {x['n']:>8,} {'—':>11}")
            continue
        log(f"  {x['bin']:>14} {x['n']:>8,} {x['phys_rmse']:>11.2f} {x['lgb_rmse']:>11.2f} "
            f"{x['delta']:>+10.2f} {x['pct']:>+8.2f}% {x['bias_phys']:>11.2f} {x['bias_lgb']:>10.2f}")
    log(f"\n  corr(物理, LGB) = {corr:.4f}    判定: {verdict}")
    return {"ap": ap, "n": int(len(r)), "excluded_corrupt": bool(exclude_corrupt),
            "svc_median": float(r["svc"].median()),
            "k_median": float(r["k"].median()), "k_p95": float(r["k"].quantile(.95)),
            "k_max": int(r["k"].max()), "n_rwy": int(r["RUNWAY_mvt"].nunique()),
            "corr_phys_lgb": round(corr, 4), "bins": rows, "verdict": verdict}

print("=" * 100)
print("流体队列反演 · 简化版可行性验证 (holdout 2025-01+07)")
print("=" * 100)
res = {}
for ap in ["EGLL", "LTFM", "LIRF"]:
    res[ap] = run_airport(ap, exclude_corrupt=True)
res["LIRF_with_corrupt"] = run_airport("LIRF", exclude_corrupt=False)
res["_meta"] = {"queue_gap_s": QUEUE_GAP, "holdout": "2025-01 + 2025-07",
                "n_total": int(len(df)), "t_unimp_source": "训练月(2-6,8-12) LUT",
                "clip": [30.0, 10800.0]}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
print(f"\n→ {OUT}")
