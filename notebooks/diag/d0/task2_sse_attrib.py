"""
task2_sse_attrib.py — full SSE 完整归因 (只读)
P0 ENS 预测 (full 567.61) 在 holdout 344,336 行上的 分档 × 机场 交叉归因。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "task2_sse_attrib.json"

BINS = [(30, 300), (300, 600), (600, 1200), (1200, 1800),
        (1800, 3600), (3600, 10800), (10800, float("inf"))]
LABELS = ["[30,300)", "[300,600)", "[600,1200)", "[1200,1800)",
          "[1800,3600)", "[3600,10800]", "(10800,inf]"]

p = pd.read_parquet(D0 / "p0_holdout_predictions.parquet")
cm = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
rev = {int(v): k for k, v in cm.get("ADEP_mvt", {}).items()}

y = p["TAXI_TRUE_s"].to_numpy(np.float64)
pred = p["TAXI_PRED_s_ENS"].to_numpy(np.float64)
mo = p["MONTH"].to_numpy()
ap = p["ADEP"].to_numpy()
# ADEP 可能是字符串(已还原) 或整数编码
if ap.dtype.kind in "iu":
    apn = np.array([rev.get(int(a), f"UNK{int(a)}") for a in ap], dtype=object)
else:
    apn = ap.astype(object)

n = len(y)
full_rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
sse_all = float(np.sum((pred - y) ** 2))
core = (y > 0) & (y <= 10800)
core_rmse = float(np.sqrt(np.mean((pred[core] - y[core]) ** 2)))
print(f"holdout n={n:,}  ENS full={full_rmse:.4f}  core={core_rmse:.4f}")
print(f"总 SSE = {sse_all:.4e}   (= full_rmse^2 * n = {full_rmse**2 * n:.4e})")

# 分档
bi = np.digitize(y, [300, 600, 1200, 1800, 3600, 10800], right=False)
# digitize: y==300 -> 1 (落在[300,600)) ; y<30 的行归 0
print(f"分档行数: " + "  ".join(f"{LABELS[k]}={int((bi==k).sum()):,}" for k in range(7)))
print(f"  注: y<30 的行 = {int((y < 30).sum()):,}")

sse_cell = (pred - y) ** 2
top_ap = ["LIRF", "EGLL", "LTFM", "LFPG", "EHAM", "LEBL", "EDDF", "EDMF", "LIRF", "LSZH"]
ap_list = [a for a in pd.Series(apn).value_counts().index]
print(f"\n机场数 = {len(ap_list)} (全部): {sorted(ap_list)}")

rows = []
for ai, a in enumerate(sorted(ap_list)):
    ma = apn == a
    for k in range(7):
        m = ma & (bi == k)
        ni = int(m.sum())
        if ni == 0:
            rows.append({"ap": a, "bin": LABELS[k], "n": 0, "sse": 0.0,
                         "pct": 0.0, "rmse": None})
            continue
        s = float(sse_cell[m].sum())
        rows.append({"ap": a, "bin": LABELS[k], "n": ni, "sse": s,
                     "pct": 100.0 * s / sse_all,
                     "rmse": float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))})

# ---- assert: 格子 SSE 加总 = 总 SSE ----
sum_sse = sum(r["sse"] for r in rows)
sum_n = sum(r["n"] for r in rows)
assert abs(sum_sse - sse_all) / sse_all < 1e-9, f"SSE 对不上: {sum_sse} vs {sse_all}"
assert sum_n == n, f"行数对不上: {sum_n} vs {n}"
print(f"\n✓ assert 通过: 格子 SSE 加总 {sum_sse:.6e} == 总 SSE {sse_all:.6e}")
print(f"✓ assert 通过: 格子行数加总 {sum_n:,} == {n:,}")

# ---- 按 bin 汇总 ----
print("\n" + "=" * 104)
print("按真值分档汇总 (全机场)")
print("=" * 104)
print(f"{'档':>14} {'n':>9} {'SSE':>14} {'占SSE':>8} {'该档RMSE':>10} {'档内平均误差':>12}")
bin_rows = []
for k in range(7):
    m = bi == k
    s = float(sse_cell[m].sum())
    bin_rows.append({"bin": LABELS[k], "n": int(m.sum()), "sse": s,
                     "pct": 100.0 * s / sse_all,
                     "rmse": float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))})
    r = bin_rows[-1]
    print(f"{r['bin']:>14} {r['n']:>9,} {r['sse']:>14.4e} {r['pct']:>7.2f}% {r['rmse']:>10.2f}")

# ---- top 杠杆格子 ----
nz = [r for r in rows if r["n"] > 0]
nz.sort(key=lambda r: -r["sse"])
print("\n" + "=" * 104)
print("SSE 贡献 top 20 格子")
print("=" * 104)
print(f"{'机场':>6} {'档':>14} {'n':>8} {'SSE':>14} {'占SSE':>8} {'RMSE':>9}")
for r in nz[:20]:
    print(f"{r['ap']:>6} {r['bin']:>14} {r['n']:>8,} {r['sse']:>14.4e} {r['pct']:>7.2f}% {r['rmse']:>9.2f}")
cum = 0.0
for i, r in enumerate(nz[:10], 1):
    cum += r["pct"]
    print(f"  top{i} 累计 {cum:.2f}%")

# ---- 机场汇总 ----
print("\n" + "=" * 104)
print("按机场汇总 (core 口径, 与前几轮可比)")
print("=" * 104)
ap_rows = []
for a in sorted(ap_list):
    ma = apn == a
    mc = ma & core
    if mc.sum() < 100:
        continue
    ap_rows.append({"ap": a, "n_core": int(mc.sum()),
                    "rmse_core": float(np.sqrt(np.mean((pred[mc] - y[mc]) ** 2))),
                    "sse_full": float(sse_cell[ma].sum()),
                    "pct_full": 100.0 * float(sse_cell[ma].sum()) / sse_all})
    r = ap_rows[-1]
    print(f"{r['ap']:>6} n_core={r['n_core']:>7,}  core_RMSE={r['rmse_core']:8.2f}  "
          f"占full SSE={r['pct_full']:6.2f}%")

# ---- >10800 与 3600-10800 专项 ----
print("\n" + "=" * 104)
print("专项: >10800 与 [3600,10800] 的机场分布")
print("=" * 104)
for k, nm in [(6, ">10800"), (5, "[3600,10800]")]:
    m = bi == k
    tot = float(sse_cell[m].sum())
    print(f"\n--- {nm} ---  n={int(m.sum()):,}  SSE={tot:.4e}  占总SSE={100*tot/sse_all:.2f}%")
    sub = (pd.DataFrame({"ap": apn[m], "sse": sse_cell[m]})
           .groupby("ap").agg(n=("sse", "size"), sse=("sse", "sum"))
           .sort_values("sse", ascending=False))
    sub["pct_of_bin"] = sub["sse"] / tot * 100
    sub["pct_of_all"] = sub["sse"] / sse_all * 100
    for a, r in sub.iterrows():
        print(f"  {a:>5} n={int(r['n']):>4}  SSE={r['sse']:.4e}  占该档={r['pct_of_bin']:5.1f}%  占总={r['pct_of_all']:5.2f}%")

# ---- Jan/Jul ----
print("\n" + "=" * 104)
print("Jan / Jul 对比")
print("=" * 104)
for m_, nm in [(1, "Jan"), (7, "Jul")]:
    m = mo == m_
    mc = m & core
    print(f"{nm}: n={int(m.sum()):,}  full={float(np.sqrt(np.mean((pred[m]-y[m])**2))):.2f}  "
          f"core={float(np.sqrt(np.mean((pred[mc]-y[mc])**2))):.2f}  "
          f"占SSE={100*float(sse_cell[m].sum())/sse_all:.2f}%")

json.dump({"n": n, "full_rmse": full_rmse, "core_rmse": core_rmse,
           "sse_all": sse_all, "assert_sse_ok": True, "assert_n_ok": True,
           "n_below30": int((y < 30).sum()),
           "cells": rows, "by_bin": bin_rows, "by_airport_full": ap_rows},
          open(OUT, "w"), ensure_ascii=False, indent=2)
print(f"\n→ {OUT}")
