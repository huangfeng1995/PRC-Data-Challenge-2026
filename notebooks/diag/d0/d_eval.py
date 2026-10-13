"""
阶段1: 验证最终点预测 D (复用已落盘模型, 不重训)
================================================
D 定义:
  - LIRF 行        : 期望混合 mu = p*(MVT-SCHED) + (1-p)*mu_N  (SCHED 落 LOBT 窗外 -> p=0)
  - 非 LIRF 行     : LGB 基线预测, 不混合
  - 全部 LOBT 有效行: 投影到 [MVT-LOBT-3606, MVT-LOBT+3606]; LOBT 缺失跳过
  - 最后 clip [30, 10800]

对照:
  A 基线 | B 基线+LOBT | C v22(基线+LIRF T14400) | D 本方案

门槛:
  1) D 分机场全部不劣化 (非 LIRF 部分应 ≈ B, 预期 9/10 改善)
  2) LIRF 上 D vs C 谁低、差多少
  3) D 相对 C 的总体 Δ 是否超 seed 抖动 (±0.5s)
  若 1) 不满足 → 停下报告, 不进阶段2

红线: 不改 label、不用外部数据、模型先落盘(已落盘)、不写 submissions/、不自动上传
"""
from __future__ import annotations
import sys, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(D0))
from ngb_probe_reduced import load_cache  # noqa: E402
import lightgbm as lgb  # noqa: E402

TS = D0 / "_ts_cache.pkl"
MODEL_DIR = D0 / "cp_mixture_models"
OUT = D0 / "d_eval.json"

FIT_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD = [1, 2, 6, 7]
LIRF_AP = "LIRF"
T_LOBT = 3606.0
LIRF_T = 14400.0
CLIP_LO, CLIP_HI = 30.0, 10800.0
SEED_JITTER = 0.5

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def rmse(P, yt, m=None):
    if m is None:
        m = np.ones(len(yt), bool)
    return float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))


def main():
    lap("加载缓存与模型")
    X, y, mo, mvt_s, ap = load_cache()
    ts = pickle.load(open(TS, "rb"))
    mvt_s2, blk_s, sch_s = ts["mvt_s"], ts["blk_s"], ts["sch_s"]
    lobt_s, lobt_valid = ts["lobt_s"], ts["lobt_valid"]
    assert np.array_equal(mvt_s, mvt_s2)

    clf = pickle.load(open(MODEL_DIR / "clf.pkl", "rb"))
    reg = pickle.load(open(MODEL_DIR / "reg.pkl", "rb"))
    lap(f"  复用 clf.pkl / reg.pkl (best_iter {clf.best_iteration}/{reg.best_iteration})")

    i_hold = np.where(np.isin(mo, HOLD))[0]
    yh, moh, ap_h = y[i_hold], mo[i_hold], ap[i_hold]
    mvt_h, sch_h, lobt_h = mvt_s[i_hold], sch_s[i_hold], lobt_s[i_hold]
    lobt_ok = lobt_valid[i_hold]

    # 基线预测 (已落盘)
    lgb_h1 = np.load(D0 / "_ngb_lgb_h1.npy")
    lgb_h2 = np.load(D0 / "_ngb_lgb_h2.npy")
    m_h1 = np.isin(moh, [1, 7])
    base = np.empty(len(i_hold))
    base[m_h1] = lgb_h1
    base[~m_h1] = lgb_h2

    lap("预测 p / mu_N")
    p = np.clip(clf.predict(X.iloc[i_hold], num_iteration=clf.best_iteration), 0, 1)
    mu_N = reg.predict(X.iloc[i_hold], num_iteration=reg.best_iteration)

    mu_C = mvt_h - sch_h
    # ---- LOBT 窗口 (全部有效行) ----
    ctr = mvt_h - lobt_h
    lo_w, hi_w = ctr - T_LOBT, ctr + T_LOBT
    # SCHED 落窗外 (|SCHED-LOBT| > T) -> p = 0
    sched_out = lobt_ok & (np.abs(sch_h - lobt_h) > T_LOBT)
    p_adj = p.copy()
    p_adj[sched_out] = 0.0

    def proj(mu):
        out = np.clip(mu, CLIP_LO, CLIP_HI).copy()
        out[lobt_ok] = np.clip(out[lobt_ok], lo_w[lobt_ok], hi_w[lobt_ok])
        return out

    is_lirf = (ap_h == LIRF_AP)

    # ---- 四个方案 ----
    A = base.copy()
    B = proj(base)
    mix_all = np.maximum(p_adj * mu_C + (1 - p_adj) * mu_N, 0.0)   # 全机场混合 (已证伪, 仅参照)
    mix_lirf = np.where(is_lirf, mix_all, base)                    # 只 LIRF 混合
    Dv = proj(mix_lirf)
    # C: v22 基线 + LIRF T=14400
    lirf_hit = is_lirf & (mu_C > LIRF_T)
    C = base.copy()
    C[lirf_hit] = mu_C[lirf_hit]

    combos = {"A_基线": A, "B_基线+LOBT": B, "C_v22_T14400": C, "D_本方案": Dv}
    print("\n" + "=" * 84)
    print("[阶段1] 方案 D 验证 — 分月 full RMSE")
    print("=" * 84)
    print(f"{'方案':<16}" + "".join(f"{'mo='+str(mm):>10}" for mm in HOLD)
          + f"{'四个月':>11}{'Δ vs A':>10}")
    print("-" * 84)
    base4 = np.sqrt(np.mean((A - yh) ** 2))
    res = {}
    for k, P in combos.items():
        r4 = np.sqrt(np.mean((P - yh) ** 2))
        cells = "".join(f"{rmse(P, yh, moh == mm):>10.2f}" for mm in HOLD)
        print(f"{k:<16}{cells}{r4:>11.2f}{r4 - base4:>+10.2f}")
        res[k] = {f"mo{mm}": round(rmse(P, yh, moh == mm), 2) for mm in HOLD}
        res[k]["all4"] = round(float(r4), 2)
        res[k]["delta_vs_A"] = round(float(r4 - base4), 2)

    # ---- 分机场 ----
    print("\n" + "=" * 84)
    print("[分机场 full RMSE — 门槛1: D 全部不劣化]")
    print("=" * 84)
    aps = pd.Series(ap_h).value_counts().index.tolist()
    print(f"{'机场':<7}{'n':>9}{'A':>9}{'B':>9}{'C':>9}{'D':>9}"
          f"{'D-B':>9}{'D-C':>9}{'判定':>8}")
    print("-" * 84)
    ap_tab = {}
    gate_pass = True
    for a in aps:
        m = ap_h == a
        ra, rb, rc, rd = (rmse(P, yh, m) for P in (A, B, C, Dv))
        d_b, d_c = rd - rb, rd - rc
        ok = rd <= rb + 1e-6          # D 不劣于 B (同口径: 非LIRF应等于B)
        if not ok:
            gate_pass = False
        ap_tab[a] = {"n": int(m.sum()), "A": round(ra, 2), "B": round(rb, 2),
                     "C": round(rc, 2), "D": round(rd, 2),
                     "D_minus_B": round(d_b, 2), "D_minus_C": round(d_c, 2),
                     "not_worse_than_B": bool(ok)}
        print(f"{a:<7}{int(m.sum()):>9,}{ra:>9.1f}{rb:>9.1f}{rc:>9.1f}{rd:>9.1f}"
              f"{d_b:>+9.2f}{d_c:>+9.2f}{'  OK' if ok else '  FAIL':>8}")
    n_ok = sum(v["not_worse_than_B"] for v in ap_tab.values())
    print(f"\n  门槛1 (D 分机场不劣于 B): {n_ok}/{len(aps)}  → "
          f"{'PASS' if gate_pass else 'FAIL'}")

    # ---- 门槛2: LIRF 上 D vs C ----
    print("\n" + "=" * 84)
    print("[门槛2] LIRF 上 D vs C")
    print("=" * 84)
    m = is_lirf
    ra, rb, rc, rd = (rmse(P, yh, m) for P in (A, B, C, Dv))
    print(f"  LIRF n={int(m.sum()):,}")
    print(f"    A={ra:.2f}  B={rb:.2f}  C={rc:.2f}  D={rd:.2f}")
    print(f"    D vs C: {rd - rc:+.2f}s  →  {'D 更低 (胜)' if rd < rc else 'C 更低 (负)'}")
    print(f"    D vs A: {rd - ra:+.2f}s")

    # ---- 门槛3: D 相对 C 总体 Δ ----
    rC, rD = np.sqrt(np.mean((C - yh) ** 2)), np.sqrt(np.mean((Dv - yh) ** 2))
    print("\n" + "=" * 84)
    print("[门槛3] D 相对 C 总体 Δ 是否超 seed 抖动")
    print("=" * 84)
    print(f"  C 四个月 = {rC:.2f}")
    print(f"  D 四个月 = {rD:.2f}")
    print(f"  Δ (D-C)  = {rD - rC:+.2f}s   |  门槛 ±{SEED_JITTER}s  → "
          f"{'PASS (超抖动)' if abs(rD - rC) > SEED_JITTER else 'FAIL (未超抖动)'}")

    # ---- 参照: 全机场混合 (已证伪, 确认 D 更优) ----
    rMix = np.sqrt(np.mean((proj(mix_all) - yh) ** 2))
    print(f"\n  [参照] 全机场混合+LOBT 四个月 = {rMix:.2f}  "
          f"(D 比它好 {rMix - rD:+.2f}s → 仅 LIRF 混合更优)")

    # ---- 非 LIRF 机场 D 是否严格等于 B ----
    non_lirf_aps = [a for a in aps if a != LIRF_AP]
    same = all(abs(ap_tab[a]["D"] - ap_tab[a]["B"]) < 1e-6 for a in non_lirf_aps)
    print(f"  [结构校验] 非 LIRF 机场 ({len(non_lirf_aps)} 个) 上 D 是否严格等于 B: {same}")

    result = {
        "T_lobt": T_LOBT, "LIRF_T": LIRF_T, "n_holdout": int(len(i_hold)),
        "combos": res,
        "per_airport": ap_tab,
        "gate1_D_not_worse_than_B": {"pass": bool(gate_pass), "n_ok": int(n_ok),
                                     "n_total": len(aps)},
        "gate2_lirf_D_vs_C": {"A": round(ra, 2), "B": round(rb, 2), "C": round(rc, 2),
                              "D": round(rd, 2), "D_minus_C": round(rd - rc, 2),
                              "D_wins": bool(rd < rc), "n_lirf": int(m.sum())},
        "gate3_D_vs_C_overall": {"C": round(float(rC), 2), "D": round(float(rD), 2),
                                 "delta": round(float(rD - rC), 2),
                                 "jitter": SEED_JITTER,
                                 "pass": bool(abs(rD - rC) > SEED_JITTER)},
        "ref_all_airport_mixture": round(float(rMix), 2),
        "non_lirf_D_equals_B": bool(same),
        "n_lirf_t14400_hit": int(lirf_hit.sum()),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # ---- 总裁决 ----
    all_pass = gate_pass and abs(rD - rC) > SEED_JITTER
    print("\n" + "=" * 84)
    print(f"总裁决: 门槛1 {'PASS' if gate_pass else 'FAIL'} | "
          f"门槛3 {'PASS' if abs(rD - rC) > SEED_JITTER else 'FAIL'}")
    print(f"→ {'✅ D 通过门槛, 可进阶段2 (生成 final parquet)' if all_pass else '⛔ 未过门槛, 停止, 不进阶段2'}")
    print(f"→ {OUT}")
    lap(f"总耗时 {(time.time() - _t0)/60:.1f} min")


if __name__ == "__main__":
    main()
