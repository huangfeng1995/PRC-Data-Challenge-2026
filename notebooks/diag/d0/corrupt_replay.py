"""
corrupt_replay.py — holdout 上离线验证"损坏行复现"策略
只读: 不训练、不改 label、不碰 ranking
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "corrupt_replay.json"
NS = 1_000_000_000
SENT = -9223372037          # AOBT 哨兵 (1677-09-21, int64 下界)
TOL = 60                    # 判定"真损坏"容差 ±60s
CLIP_LO, CLIP_HI = 30.0, 10800.0
BASE_FULL, BASE_CORE = 567.61, 304.94

def log(*a): print(" ".join(str(x) for x in a), flush=True)

# ---------- 载入 holdout ----------
P = pd.read_parquet(D0 / "p0_holdout_predictions.parquet")
C = ["MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt",
     "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "ADEP_mvt"]
raw = pd.concat([pd.read_parquet(ROOT / f"data/raw/{f}", columns=C) for f in
                 ["training_2025-01-01_2025-02-01.parquet",
                  "training_2025-07-01_2025-08-01.parquet"]], ignore_index=True)
raw = raw[raw["PHASE_mvt"] == "DEP"]
for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]:
    raw[c] = pd.to_datetime(raw[c], utc=True, errors="coerce")
    raw[c + "_s"] = (raw[c].dt.tz_convert("UTC").dt.tz_localize(None)
                     .astype("datetime64[ns]").astype("int64") // NS)

df = P.merge(raw[["MVT_ID_mvt", "MVT_TIME_UTC_mvt_s", "BLOCK_TIME_UTC_mvt_s",
                  "SCHED_TIME_UTC_mvt_s", "AOBT_3_flt_s"]], on="MVT_ID_mvt", how="left")
assert len(df) == len(P) and df["MVT_TIME_UTC_mvt_s"].notna().all()

y = df["TAXI_TRUE_s"].to_numpy(np.float64)
ens = df["TAXI_PRED_s_ENS"].to_numpy(np.float64)
mo = df["MONTH"].to_numpy()
ap = df["ADEP"].to_numpy()
cand = (df["MVT_TIME_UTC_mvt_s"] - df["SCHED_TIME_UTC_mvt_s"]).to_numpy(np.float64)
aobt_sent = (df["AOBT_3_flt_s"].to_numpy(np.float64) == SENT)
block = df["BLOCK_TIME_UTC_mvt_s"].to_numpy(np.float64)
sched = df["SCHED_TIME_UTC_mvt_s"].to_numpy(np.float64)
n = len(y)

# ---------- 真损坏定义: BLOCK==SCHED (±60s), 即标签本身已被 SCHED 顶替 ----------
TRUE_CORRUPT = np.isfinite(block) & (np.abs(block - sched) <= TOL)
log("=" * 96)
log(f"holdout n={n:,}  (Jan {int((mo==1).sum()):,} / Jul {int((mo==7).sum()):,})")
log(f"基线 ENS      : full={BASE_FULL:.2f}  core={BASE_CORE:.2f}")
log(f"真损坏行(BLOCK==SCHED ±60s): {int(TRUE_CORRUPT.sum()):,}")
core_m = y <= 10800
log(f"core 行: {int(core_m.sum()):,}   >10800 行: {int((~core_m).sum()):,}")

def R(p, m):
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.sum() else float("nan")

def base_full(m=None):
    return BASE_FULL if m is None else R(ens, m if m is not None else np.ones(n, bool))

BASE_JAN, BASE_JUL = R(ens, mo == 1), R(ens, mo == 7)
BASE_AP = {a: R(ens, (ap == a)) for a in pd.unique(ap)}
log(f"基线 Jan={BASE_JAN:.2f}  Jul={BASE_JUL:.2f}")

# ---------- T 网格 ----------
TS = [3600, 5400, 7200, 9000, 10800, 14400]
res = []
for T in TS:
    fired = np.isfinite(cand) & (cand > T)
    for use_sent in (False, True):
        sel = fired & aobt_sent if use_sent else fired
        for do_clip in (False, True):
            pred = ens.copy()
            if sel.sum():
                v = cand[sel]
                pred[sel] = np.clip(v, CLIP_LO, CLIP_HI) if do_clip else v
            tp = int((sel & TRUE_CORRUPT).sum())
            fp = int((sel & ~TRUE_CORRUPT).sum())
            fn = int((~sel & TRUE_CORRUPT).sum())
            # 误判行的误差代价
            fp_cost = float(np.sum((pred[sel & ~TRUE_CORRUPT] - y[sel & ~TRUE_CORRUPT]) ** 2)) if fp else 0.0
            tp_gain = float(np.sum((ens[sel & TRUE_CORRUPT] - y[sel & TRUE_CORRUPT]) ** 2)
                            - np.sum((pred[sel & TRUE_CORRUPT] - y[sel & TRUE_CORRUPT]) ** 2)) if tp else 0.0
            r = {
                "T": T, "use_aobt_sentinel": use_sent, "clip": do_clip,
                "n_fired": int(sel.sum()), "n_true_corrupt": int(TRUE_CORRUPT.sum()),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / max(int(sel.sum()), 1), 4),
                "recall": round(tp / max(int(TRUE_CORRUPT.sum()), 1), 4),
                "fp_sse_cost": round(fp_cost, 1), "tp_sse_gain": round(tp_gain, 1),
                "rmse_full": round(R(pred, np.ones(n, bool)), 2),
                "rmse_core": round(R(pred, core_m), 2),
                "delta_full": round(R(pred, np.ones(n, bool)) - BASE_FULL, 2),
                "delta_core": round(R(pred, core_m) - BASE_CORE, 2),
                "rmse_jan": round(R(pred, mo == 1), 2),
                "rmse_jul": round(R(pred, mo == 7), 2),
                "delta_jan": round(R(pred, mo == 1) - BASE_JAN, 2),
                "delta_jul": round(R(pred, mo == 7) - BASE_JUL, 2),
            }
            # 分机场
            apr = {}
            for a in pd.unique(ap):
                m = ap == a
                apr[str(a)] = {"n": int(m.sum()), "full": round(R(pred, m), 2),
                               "base_full": round(BASE_AP[a], 2),
                               "delta": round(R(pred, m) - BASE_AP[a], 2)}
            r["per_airport"] = apr
            ds = [v["delta"] for v in apr.values()]
            r["n_airport_improve"] = int(sum(1 for d in ds if d < 0))
            r["n_airport_worse"] = int(sum(1 for d in ds if d > 0))
            r["same_sign"] = bool(r["n_airport_improve"] == 0 or r["n_airport_worse"] == 0)
            res.append(r)

RES = {"base": {"full": BASE_FULL, "core": BASE_CORE,
                "jan": round(BASE_JAN, 2), "jul": round(BASE_JUL, 2),
                "per_airport": {k: round(v, 2) for k, v in BASE_AP.items()}},
       "n_true_corrupt": int(TRUE_CORRUPT.sum()),
       "aobt_sentinel_rate_corrupt": round(float(aobt_sent[TRUE_CORRUPT].mean()) * 100, 2),
       "aobt_sentinel_rate_clean": round(float(aobt_sent[~TRUE_CORRUPT].mean()) * 100, 2),
       "n_airports": int(len(BASE_AP)), "grid": res}
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 打印 ----------
log("")
log("=" * 150)
log("T 网格 (sent=AOBT哨兵辅助, clip=对复现值做[30,10800])")
log("=" * 150)
hdr = f"{'T':>6} {'sent':>5} {'clip':>5} {'fired':>6} {'TP':>5} {'FP':>5} {'prec':>7} {'rec':>7} {'full':>9} {'Δfull':>9} {'core':>8} {'Δcore':>8} {'Jan':>8} {'Jul':>8} {'改善/变差':>10}"
log(hdr)
for r in res:
    log(f"{r['T']:>6} {str(r['use_aobt_sentinel']):>5} {str(r['clip']):>5} {r['n_fired']:>6} "
        f"{r['tp']:>5} {r['fp']:>5} {r['precision']:>7.4f} {r['recall']:>7.4f} "
        f"{r['rmse_full']:>9.2f} {r['delta_full']:>+9.2f} {r['rmse_core']:>8.2f} {r['delta_core']:>+8.2f} "
        f"{r['rmse_jan']:>8.2f} {r['rmse_jul']:>8.2f} {r['n_airport_improve']:>4}/{r['n_airport_worse']:<4}")

log("")
log("=" * 96)
log("最优组合 (按 full RMSE 排序, 前 6)")
log("-" * 96)
for r in sorted(res, key=lambda x: x["rmse_full"])[:6]:
    log(f"  T={r['T']:>6} sent={str(r['use_aobt_sentinel']):5s} clip={str(r['clip']):5s} "
        f"full={r['rmse_full']:8.2f} (Δ{r['delta_full']:+7.2f})  core={r['rmse_core']:7.2f} "
        f"(Δ{r['delta_core']:+6.2f})  机场 {r['n_airport_improve']}改善/{r['n_airport_worse']}变差 "
        f"{'同号✓' if r['same_sign'] else '同号✗'}")

best = min(res, key=lambda x: x["rmse_full"])
log("")
log("=" * 96)
log(f"最优: T={best['T']} sent={best['use_aobt_sentinel']} clip={best['clip']}  "
    f"full={best['rmse_full']:.2f} (Δ{best['delta_full']:+.2f})  core={best['rmse_core']:.2f} (Δ{best['delta_core']:+.2f})")
log("分机场明细 (最优组合):")
for a, v in sorted(best["per_airport"].items(), key=lambda kv: kv[1]["delta"]):
    log(f"  {a:>5s} n={v['n']:>7,} base={v['base_full']:8.2f} → new={v['full']:8.2f}  Δ={v['delta']:+8.2f}")
log(f"\n→ {OUT}")
