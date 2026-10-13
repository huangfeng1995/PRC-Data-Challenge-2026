"""
step3_identify.py — 候选批次的可识别性检验 (禁止写死机场)

要回答: 能否只用**通用特征**把候选批次挑出来?
允许的信号 (全部来自当前航班自身, 无未来信息):
  (a) 跨源缺失: ADEP_flt 缺失 / AOBT_3 哨兵
  (b) MVT 与某个计划时刻的差值落在某区间
不允许: if ADEP_mvt == '某机场'

输出: 每个候选批次的 precision / recall (整体 + 分机场), 以及误伤绝对数。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "step3_identify.json"
EXACT_S = 2
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
                precision=round(p, 5), recall=round(r, 5), f1=round(f, 5))


def main():
    M = np.load(D0 / "_step2_masks.npy")
    ap = np.load(D0 / "_step2_ap.npy", allow_pickle=True)
    mo = np.load(D0 / "_step2_month.npy")
    taxi = np.load(D0 / "_step2_taxi.npy")
    mvt = np.load(D0 / "_step2_mvt.npy")
    T = {t: np.load(D0 / f"_step2_T_{t}.npy") for t in ["SCHED", "LOBT", "EOBT", "IOBT"]}
    only_sched, only_lo, only_e, only_i, allm, anym, base, nm, sent, corrupt, pool = M
    n = len(base)
    lap(f"训练 DEP {n:,}")

    # ground truth: 各类"只匹配某时刻"的行
    cands = {"only_SCHED": only_sched, "only_LOBT": only_lo,
             "only_EOBT": only_e, "only_IOBT": only_i, "all4": allm}

    print("\n" + "=" * 104)
    print("第三步 · 可识别性: 通用信号能否挑出候选批次 (不写死机场)")
    print("=" * 104)
    print(f"{'候选批次':<12} {'n':>8} | {'通用信号':<44} {'命中':>8} {'误伤':>9} {'precision':>10} {'recall':>8}")
    print("-" * 104)
    res = {}

    for name, truth in cands.items():
        if truth.sum() == 0:
            res[name] = {"n": 0, "note": "候选批次为空"}
            print(f"{name:<12} {0:>8,} | (空批次, 跳过)")
            continue
        # 该批次对应"BLOCK 等于哪个计划时刻" —— 对 only_X 就是 X
        tgt = name.replace("only_", "").replace("all4", "SCHED")
        d = mvt - T[tgt]
        # 通用信号组合 (逐个报告, 不做无依据的调参)
        rules = {
            "A: ADEP_flt缺失": nm,
            "B: AOBT哨兵": sent,
            "C: ADEP缺 或 AOBT哨兵": (nm | sent),
            f"D: MVT-{tgt} > 10800": d > 10800,
            f"E: MVT-{tgt} > 14400": d > 14400,
            f"F: (A或B) 且 MVT-{tgt}>10800": ((nm | sent) & (d > 10800)),
            f"G: (A或B) 且 MVT-{tgt}>14400": ((nm | sent) & (d > 14400)),
            f"H: (A或B) 且 MVT-{tgt}>21600": ((nm | sent) & (d > 21600)),
        }
        res[name] = {"n": int(truth.sum()), "target_time": tgt, "rules": {}}
        for rn, rm in rules.items():
            tp = int((rm & truth).sum()); fp = int((rm & ~truth).sum())
            fn = int((~rm & truth).sum())
            e = prf(tp, fp, fn)
            res[name]["rules"][rn] = e
            flag = "" if e["precision"] >= 0.5 else "   ← precision 不可用"
            print(f"{name:<12} {int(truth.sum()):>8,} | {rn:<44} {e['tp']:>8,} "
                  f"{e['fp']:>9,} {e['precision']:>10.4f} {e['recall']:>8.4f}{flag}")
        print()

    # ---------- 分机场: 最好的通用规则是否在某机场特别准 ----------
    print("=" * 104)
    print("分机场检验 (规则 G: (A或B) 且 MVT-{tgt}>14400)")
    print("=" * 104)
    res["per_airport"] = {}
    for name, truth in cands.items():
        if truth.sum() == 0:
            continue
        tgt = name.replace("only_", "").replace("all4", "SCHED")
        d = mvt - T[tgt]
        rule = (nm | sent) & (d > 14400)
        sub = {"tp": 0, "fp": 0, "fn": 0}
        per = {}
        for a in np.unique(ap):
            mt = ap == a
            tp = int((rule & truth & mt).sum()); fp = int((rule & ~truth & mt).sum())
            fn = int((~rule & truth & mt).sum())
            per[str(a)] = prf(tp, fp, fn)
        res["per_airport"][name] = per
        bad = [a for a, e in per.items() if e["fp"] > 0 and e["precision"] < 0.5]
        print(f"  {name:<12} 规则 G 误伤最多的机场: "
              + (", ".join(f"{a}({per[a]['fp']:,}/{per[a]['precision']:.2f})" for a in bad[:6])
                 if bad else "无"))

    # ---------- 结论 ----------
    print("\n" + "=" * 104)
    print("结论判定")
    print("=" * 104)
    verdicts = []
    for name, truth in cands.items():
        if truth.sum() == 0:
            verdicts.append((name, 0, "候选批次为空"))
            continue
        tgt = name.replace("only_", "").replace("all4", "SCHED")
        d = mvt - T[tgt]
        best = max(res[name]["rules"].items(), key=lambda kv: kv[1]["precision"])
        rn, e = best
        ok = e["precision"] >= 0.5 and e["fp"] <= 500
        verdicts.append((name, int(truth.sum()), f"最优规则 {rn} precision={e['precision']:.4f} "
                                             f"误伤={e['fp']:,} → {'可用' if ok else '不可用'}"))
        print(f"  {name:<12} n={int(truth.sum()):>7,}  {verdicts[-1][2]}")
    res["verdicts"] = {v[0]: v[2] for v in verdicts}

    # only_EOBT 的关键否证: 它与 ADEP 缺失零相关
    if only_e.sum():
        res["only_EOBT_diagnosis"] = {
            "n": int(only_e.sum()),
            "ADEP_flt_missing": int(nm[only_e].sum()),
            "AOBT_sentinel": int(sent[only_e].sum()),
            "TAXI_p50": round(float(np.nanmedian(taxi[only_e])), 1),
            "baseline_TAXI_p50": round(float(np.nanmedian(taxi[base])), 1),
            "median_abs_BLOCK_minus_LOBT": round(
                float(np.nanmedian(np.abs((mvt - taxi) - T["LOBT"])[only_e])), 1),
            "median_abs_BLOCK_minus_IOBT": round(
                float(np.nanmedian(np.abs((mvt - taxi) - T["IOBT"])[only_e])), 1),
            "n_airports": int(len(np.unique(ap[only_e]))),
        }
        lap("\nonly_EOBT 诊断: " + json.dumps(res["only_EOBT_diagnosis"], ensure_ascii=False))

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    lap(f"→ {OUT}")


if __name__ == "__main__":
    main()
