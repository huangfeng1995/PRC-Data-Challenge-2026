"""
Step1: LOBT 窗口阈值标定
========================
口径:
- 只在 LOBT 有效行 (约 206 万) 统计 d = |BLOCK - LOBT|
- 哨兵/NaT 行 (22,470) 单独标记, 不参与阈值统计, 投影时跳过
- 自己定阈值, 不照抄 3606

输出 lobt_window.json
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

D0 = Path("/Users/openclaw/Documents/PRC_data_challenge/notebooks/diag/d0")
TS = D0 / "_ts_cache.pkl"
OUT = D0 / "lobt_window.json"

FIT_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD = [1, 2, 6, 7]


def main():
    ts = pickle.load(open(TS, "rb"))
    blk, lobt, lobt_valid = ts["blk_s"], ts["lobt_s"], ts["lobt_valid"]
    mo, y, ap = ts["mo"], ts["y"], ts["ap"]
    mvt, sch = ts["mvt_s"], ts["sch_s"]
    n = len(y)

    m = lobt_valid & ~np.isnan(blk)
    print(f"LOBT 有效行: {int(m.sum()):,} / {n:,}  ({m.mean()*100:.2f}%)")
    print(f"LOBT 无效行: {int((~m).sum()):,}  (哨兵/NaT, 不参与标定)")

    d = np.abs(blk[m] - lobt[m])
    print(f"\nd = |BLOCK - LOBT|  (n={len(d):,})")
    qs = [50, 90, 95, 99, 99.5, 99.9, 99.95, 99.99, 100]
    print("  分位      值(s)")
    vals = {}
    for q in qs:
        v = float(np.percentile(d, q))
        vals[str(q)] = round(v, 1)
        print(f"  p{q:<8} {v:12.1f}")

    # T_lobt: 覆盖 99.9% / 100%
    t_999 = float(np.ceil(np.percentile(d, 99.9)))
    t_100 = float(np.ceil(d.max()))
    print(f"\n  T_lobt(99.9%) = {t_999:.0f}s  (覆盖 {(d <= t_999).mean()*100:.4f}%)")
    print(f"  T_lobt(100%)  = {t_100:.0f}s  (覆盖 {(d <= t_100).mean()*100:.4f}%)")
    print(f"  max d = {d.max():.0f}s")

    # 分月看 d 尾部是否稳定
    mo_m = mo[m]
    print("\n  分月 d 分位 (检查尾部是否月间一致):")
    per_mo = {}
    for mm in range(1, 13):
        sel = mo_m == mm
        if sel.sum() == 0:
            continue
        dd = d[sel]
        per_mo[str(mm)] = {
            "n": int(sel.sum()),
            "p50": round(float(np.percentile(dd, 50)), 1),
            "p90": round(float(np.percentile(dd, 90)), 1),
            "p99": round(float(np.percentile(dd, 99)), 1),
            "p99.9": round(float(np.percentile(dd, 99.9)), 1),
            "max": round(float(dd.max()), 1),
        }
        print(f"    mo={mm:2d} n={sel.sum():>7,} p50={per_mo[str(mm)]['p50']:>8.1f} "
              f"p90={per_mo[str(mm)]['p90']:>9.1f} p99={per_mo[str(mm)]['p99']:>9.1f} "
              f"p99.9={per_mo[str(mm)]['p99.9']:>9.1f} max={per_mo[str(mm)]['max']:>10.1f}")

    # d > 阈值 的行都是什么 (检验 T=3606 的合理性 + 更大阈值的代价)
    print("\n  若用 T=3606 (既有值) 会夹掉多少行 / 涉及多少 SSE:")
    y_m = y[m]
    for T in [1800, 3606, 7200, t_999, t_100]:
        far = d > T
        if far.sum() == 0:
            print(f"    T={T:>8.0f}s  窗外行 0")
            continue
        true_t = mvt[m] - lobt[m]     # 真实 LOBT 推出的滑行时间
        print(f"    T={T:>8.0f}s  窗外行 {int(far.sum()):>6,} ({far.mean()*100:.4f}%)  "
              f"窗外 |MVT-LOBT| 中位 {np.median(true_t[far]):>8.1f}s  "
              f"max {true_t[far].max():>9.0f}s")

    # 分机场 d 分位 (确认阈值不需要写死机场)
    print("\n  分机场 d p99 / p99.9 / max (阈值不应写死机场):")
    ap_m = ap[m]
    df_ap = pd.DataFrame({"ap": ap_m, "d": d})
    g = df_ap.groupby("ap")["d"]
    ap_tab = g.agg(n="size", p50=lambda s: np.percentile(s, 50),
                   p99=lambda s: np.percentile(s, 99),
                   p999=lambda s: np.percentile(s, 99.9),
                   max="max").sort_values("n", ascending=False)
    print(ap_tab.to_string(float_format=lambda v: f"{v:,.1f}"))

    # 有效行里的 eq 关系 (给 Step2 标签口径做交叉核对)
    d_bs = np.abs(blk[m] - sch[m])
    eq = d_bs <= 60
    print(f"\n  [交叉核对] LOBT 有效行中 eq(|BLOCK-SCHED|<=60s) 阳性率 = {eq.mean()*100:.3f}%")

    result = {
        "quantiles_sec": vals,
        "T_lobt_999": t_999,
        "T_lobt_100": t_100,
        "d_max": float(d.max()),
        "n_lobt_valid": int(m.sum()),
        "n_lobt_invalid": int((~m).sum()),
        "per_month": per_mo,
        "per_airport": {k: {kk: float(vv) for kk, vv in v.items()}
                        for k, v in ap_tab.to_dict("index").items()},
        "eq_rate_lobt_valid": round(float(eq.mean()), 5),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
