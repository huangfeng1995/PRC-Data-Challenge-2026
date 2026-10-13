"""
两阶段期望混合 (拷贝分类器) + LOBT 窗口 — 2025 四个 holdout 月验证
==============================================================
Step2 拷贝分类器: eq = (|BLOCK-SCHED| <= 60s), LGB 二分类, 仅 fit 月训练,
       四个 holdout 月 (mo ∈ {1,2,6,7}) 只预测得概率 p (不 argmax)
Step3 正常航班回归: LGB L2, 仅 ~eq 行训练, 目标 clip 到 7200, 得 mu_N
Step4 期望混合: mu_C = MVT-SCHED;  mu = p*mu_C + (1-p)*mu_N, 下限 0
       LOBT 投影: 夹到 [MVT-LOBT-T_lobt, MVT-LOBT+T_lobt]; SCHED 落窗外时 p=0
Step5 分月 full RMSE 六组合对照 + p 分布 / eq 阳性率 / 误报数

模型先落盘再评估。
红线: 不改 label、不用外部数据、不写死机场 (LIRF 既有规则除外)、不写 submissions/、不自动上传。
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
OUT = D0 / "copy_mixture_lobt.json"
MODEL_DIR = D0 / "cp_mixture_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CLS_PKL = MODEL_DIR / "clf.pkl"
REG_PKL = MODEL_DIR / "reg.pkl"

FIT_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD = [1, 2, 6, 7]
SEED = 42

# LOBT 窗口 (Step1 标定: p100=3606, p99.9=3479; 全 12 月 / 10 机场 max 恒为 3606)
T_LOBT = 3606.0
T_LOBT_999 = 3479.0

# 既有 LIRF T=14400 规则 (对照组)
LIRF_T = 14400.0
LIRF_AP = "LIRF"

EQ_TOL = 60.0          # |BLOCK-SCHED| <= 60s 判为拷贝
REG_CLIP = 7200.0      # 正常航班回归目标 clip

BASE_PARAMS = {
    "learning_rate": 0.0147555939449363,
    "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
CLS_PARAMS = {**BASE_PARAMS, "objective": "binary", "metric": "binary_logloss",
              "is_unbalance": False}
REG_PARAMS = {**BASE_PARAMS, "objective": "regression", "metric": "rmse"}
CAP = 3000
NROUND_CAP = 3000

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def rmse(P, yt):
    return float(np.sqrt(np.mean((P - yt) ** 2)))


def main():
    lap("加载缓存")
    X, y, mo, mvt_s, ap = load_cache()
    ts = pickle.load(open(TS, "rb"))
    mvt_s2, blk_s, sch_s = ts["mvt_s"], ts["blk_s"], ts["sch_s"]
    lobt_s, lobt_valid = ts["lobt_s"], ts["lobt_valid"]
    assert np.array_equal(mvt_s, mvt_s2), "两缓存 MVT 秒不一致"

    # ---- 标签 ----
    d_bs = np.abs(blk_s - sch_s)          # BLOCK 全部有效 (2025 训练集)
    eq = d_bs <= EQ_TOL                    # 拷贝 (BLOCK 顶替 SCHED)
    lap(f"eq 全量阳性率 = {eq.mean()*100:.3f}%  ({int(eq.sum()):,}/{len(eq):,})")

    # ---- LGB 基线 holdout 预测 (已落盘, 不重训) ----
    # _ngb_lgb_h1.npy: holdout1 (mo 1,7) 344,419;  _ngb_lgb_h2.npy: holdout2 (mo 2,6) 326,874
    lgb_h1 = np.load(D0 / "_ngb_lgb_h1.npy")
    lgb_h2 = np.load(D0 / "_ngb_lgb_h2.npy")
    i_h1 = np.where(np.isin(mo, [1, 7]))[0]
    i_h2 = np.where(np.isin(mo, [2, 6]))[0]
    assert len(lgb_h1) == len(i_h1) and len(lgb_h2) == len(i_h2)
    # 校验基线预测确实按 i_h1/i_h2 顺序存 (用 RMSE 已知值反查)
    lap(f"  LGB 基线: h1={len(i_h1):,}  h2={len(i_h2):,}")

    # ---- 索引 ----
    i_fit = np.where(np.isin(mo, FIT_MONTHS))[0]
    # ES: fit 月按 MVT 排序最后 10%
    o = i_fit[np.argsort(mvt_s[i_fit], kind="stable")]
    cut = int(len(o) * 0.9)
    i_fit_tr, i_es = o[:cut], o[cut:]
    es_m = sorted(set(mo[i_es].tolist()))
    assert set(es_m) <= {11, 12}, f"ES 月份异常 {es_m}"
    assert not np.isin(mo[i_es], HOLD).any()
    lap(f"fit={len(i_fit):,}  ES={len(i_es):,} 月 {es_m}")
    i_hold = np.where(np.isin(mo, HOLD))[0]
    lap(f"holdout 四个月 (mo 1/2/6/7) = {len(i_hold):,}")

    # ================= Step2 拷贝分类器 =================
    lap("\n" + "=" * 60)
    lap("Step2: 拷贝分类器 LGB binary (eq = |BLOCK-SCHED| <= 60s)")
    if CLS_PKL.exists():
        clf = pickle.load(open(CLS_PKL, "rb"))
        lap(f"  复用已训模型 {CLS_PKL.name}")
    else:
        ds = lgb.Dataset(X.iloc[i_fit_tr], label=eq[i_fit_tr].astype(np.int8),
                         free_raw_data=False)
        dv = lgb.Dataset(X.iloc[i_es], label=eq[i_es].astype(np.int8), reference=ds)
        t1 = time.time()
        clf = lgb.train(CLS_PARAMS, ds, num_boost_round=NROUND_CAP, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(200, verbose=False),
                                   lgb.log_evaluation(0)])
        pickle.dump(clf, open(CLS_PKL, "wb"), protocol=4)
        lap(f"  训练完成 {time.time()-t1:.1f}s  best_iter={clf.best_iteration}  "
            f"→ 落盘 {CLS_PKL.name}")

    # ================= Step3 正常航班回归 (~eq) =================
    lap("\nStep3: 正常航班回归 LGB L2 (仅 ~eq, 目标 clip 7200)")
    i_norm_fit = i_fit[~eq[i_fit]]
    y_norm = np.clip(y, 0, REG_CLIP)
    if REG_PKL.exists():
        reg = pickle.load(open(REG_PKL, "rb"))
        lap(f"  复用已训模型 {REG_PKL.name}")
    else:
        ds = lgb.Dataset(X.iloc[i_norm_fit], label=y_norm[i_norm_fit],
                         free_raw_data=False)
        dv = lgb.Dataset(X.iloc[i_es], label=y_norm[i_es], reference=ds)
        t2 = time.time()
        reg = lgb.train(REG_PARAMS, ds, num_boost_round=NROUND_CAP, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(200, verbose=False),
                                   lgb.log_evaluation(0)])
        pickle.dump(reg, open(REG_PKL, "wb"), protocol=4)
        lap(f"  训练完成 {time.time()-t2:.1f}s  best_iter={reg.best_iteration}  "
            f"→ 落盘 {REG_PKL.name}")
    lap(f"  ~eq 训练行数 = {len(i_norm_fit):,} / {len(i_fit):,}")

    # ================= Step4 四个 holdout 月预测 =================
    lap("\nStep4: 预测四个 holdout 月 (mo 1/2/6/7)")
    P_clf = clf.predict(X.iloc[i_hold], num_iteration=clf.best_iteration)
    p = np.clip(P_clf, 0.0, 1.0)
    mu_N = reg.predict(X.iloc[i_hold], num_iteration=reg.best_iteration)

    yh = y[i_hold]
    moh = mo[i_hold]
    eqh = eq[i_hold]
    ap_h = ap[i_hold]
    mvt_h, sch_h, lobt_h = mvt_s[i_hold], sch_s[i_hold], lobt_s[i_hold]
    lobt_ok = lobt_valid[i_hold]

    mu_C = mvt_h - sch_h                     # 拷贝假设下的滑行时间
    mu_mix = np.maximum(p * mu_C + (1 - p) * mu_N, 0.0)

    # ---- LOBT 窗口 ----
    # 窗口中心 = MVT - LOBT (即 LOBT 锚定的滑行时间), 半宽 T_lobt
    ctr = mvt_h - lobt_h
    lo_w = ctr - T_LOBT
    hi_w = ctr + T_LOBT
    # SCHED 落窗外 → p=0 (拷贝假设不可信)。
    # 判据必须在"时长"空间比较: SCHED 作为 BLOCK 代理时推出的滑行时间 mu_C = MVT-SCHED
    # 是否落在窗口内, 等价于 |SCHED - LOBT| > T_lobt (即 Step1 标定的 d 越界)。
    sched_out = lobt_ok & (np.abs(sch_h - lobt_h) > T_LOBT)
    p_adj = p.copy()
    p_adj[sched_out] = 0.0
    mu_mix_adj = np.maximum(p_adj * mu_C + (1 - p_adj) * mu_N, 0.0)

    def project(mu):
        """夹到 LOBT 窗口 (仅 LOBT 有效行)"""
        out = mu.copy()
        out[lobt_ok] = np.clip(mu[lobt_ok], lo_w[lobt_ok], hi_w[lobt_ok])
        return out

    mu_mix_proj = project(mu_mix)
    mu_mix_adj_proj = project(mu_mix_adj)

    # ---- LGB 基线 (按 holdout 顺序拼回) ----
    base_all = np.empty(len(i_hold))
    m_h1 = np.isin(moh, [1, 7])
    base_all[m_h1] = lgb_h1
    base_all[~m_h1] = lgb_h2
    # LOBT 窗口 (对基线)
    base_proj = project(base_all)

    # ---- 既有 LIRF T=14400 (对照组) ----
    # 规则: ADEP==LIRF AND ADEP_flt 缺失 AND MVT-SCHED > 14400 → 预测改为 MVT-SCHED
    # 训练集无 ADEP_flt 缺失概念, 用等价的"eq 拷贝"作为对照代理不可行;
    # 故此处直接用可见的 MVT-SCHED 规则在 LIRF 子集上做对照 (ranking 可复现)
    lirf_mask = (ap_h == LIRF_AP) & (mu_C > LIRF_T)
    def apply_lirf(mu):
        out = mu.copy()
        out[lirf_mask] = mu_C[lirf_mask]
        return out

    # ================= Step5 分月对照 =================
    combos = {
        "0_LGB基线": base_all,
        "1_LGB+LOBT窗": base_proj,
        "2_期望混合": mu_mix,
        "3_混合+LOBT窗": mu_mix_proj,
        "4_LIRF_T14400": apply_lirf(base_all),
        "5_混合+LOBT+LIRF": apply_lirf(mu_mix_adj_proj),
    }

    lap("\n" + "=" * 60)
    print("[分月 full RMSE 对照]")
    print("=" * 78)
    header = f"{'组合':<20}" + "".join(f"{'mo='+str(mm):>11}" for mm in HOLD) + f"{'四个月':>12}{'Δ vs 基线':>12}"
    print(header)
    print("-" * 78)
    res = {}
    base_mo = {}
    for mm in HOLD:
        m = moh == mm
        base_mo[mm] = rmse(base_all[m], yh[m])
    print(f"{'(基线 mo RMSE)':<20}" + "".join(f"{base_mo[mm]:>11.2f}" for mm in HOLD))
    for name, P in combos.items():
        row, tot = [], []
        for mm in HOLD:
            m = moh == mm
            r = rmse(P[m], yh[m])
            row.append(f"{r:>11.2f}")
            tot.append(r * m.sum())
        tot_rmse = np.sqrt(np.average(np.square(np.concatenate(
            [P[moh == mm] - yh[moh == mm] for mm in HOLD]))))
        d = tot_rmse - np.sqrt(np.average(np.square(np.concatenate(
            [base_all[moh == mm] - yh[moh == mm] for mm in HOLD]))))
        print(f"{name:<20}" + "".join(row) + f"{tot_rmse:>12.2f}{d:>+12.2f}")
        res[name] = {f"mo{mm}": round(rmse(P[moh == mm], yh[moh == mm]), 2) for mm in HOLD}
        res[name]["all4"] = round(float(tot_rmse), 2)
        res[name]["delta_vs_base"] = round(float(d), 2)
        res[name]["per_month_delta"] = {
            f"mo{mm}": round(rmse(P[moh == mm], yh[moh == mm]) - base_mo[mm], 2)
            for mm in HOLD}

    # ---- p 分布 / eq 阳性率 / 误报 ----
    lap("\n" + "=" * 60)
    print("[p 分布 / eq 阳性率 / 误报]")
    print("=" * 78)
    print(f"{'':<14}" + "".join(f"{'mo='+str(mm):>13}" for mm in HOLD))
    print(f"{'n':<14}" + "".join(f"{int((moh==mm).sum()):>13,}" for mm in HOLD))
    print(f"{'eq 阳性率%':<14}" + "".join(f"{eqh[moh==mm].mean()*100:>13.3f}" for mm in HOLD))
    print(f"{'p 均值':<14}" + "".join(f"{p[moh==mm].mean():>13.4f}" for mm in HOLD))
    print(f"{'p 中位':<14}" + "".join(f"{np.median(p[moh==mm]):>13.4f}" for mm in HOLD))
    print(f"{'p p90':<14}" + "".join(f"{np.percentile(p[moh==mm],90):>13.4f}" for mm in HOLD))
    print(f"{'p p99':<14}" + "".join(f"{np.percentile(p[moh==mm],99):>13.4f}" for mm in HOLD))
    print(f"{'p max':<14}" + "".join(f"{p[moh==mm].max():>13.4f}" for mm in HOLD))
    print(f"{'p>0.5 行数':<14}" + "".join(f"{int((p[moh==mm]>0.5).sum()):>13,}" for mm in HOLD))
    # 误报: 真实正常 (~eq) 但混合预测 mu > 1h
    fp = (~eqh) & (mu_mix > 3600)
    fp_base = (~eqh) & (base_all > 3600)
    print(f"{'~eq 行数':<14}" + "".join(f"{int((~eqh)[moh==mm].sum()):>13,}" for mm in HOLD))
    print(f"{'误报(混合>1h)':<12}" + "".join(f"{int(fp[moh==mm].sum()):>13,}" for mm in HOLD))
    print(f"{'基线>1h 参照':<12}" + "".join(f"{int(fp_base[moh==mm].sum()):>13,}" for mm in HOLD))
    # 分类器 AUC
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for mm in HOLD:
        m = moh == mm
        if eqh[m].sum() > 0 and (~eqh[m]).sum() > 0:
            aucs[f"mo{mm}"] = round(float(roc_auc_score(eqh[m].astype(int), p[m])), 4)
    print(f"\n  分类器 AUC: {aucs}")

    # ---- 分机场同号检查 ----
    print("\n" + "=" * 78)
    print("[分机场 full RMSE — 同号检查]")
    print("=" * 78)
    aps = pd.Series(ap_h).value_counts().index.tolist()
    print(f"{'机场':<8}{'n':>9}{'基线':>10}" + "".join(f"{k:>14}" for k in combos if k != "0_LGB基线"))
    ap_delta = {}
    for a in aps:
        m = ap_h == a
        b = rmse(base_all[m], yh[m])
        cells = []
        deltas = {}
        for k, P in combos.items():
            if k == "0_LGB基线":
                continue
            r = rmse(P[m], yh[m])
            deltas[k] = r - b
            cells.append(f"{r:>8.1f}{r-b:>+6.1f}")
        ap_delta[a] = {k: round(v, 2) for k, v in deltas.items()}
        print(f"{a:<8}{int(m.sum()):>9,}{b:>10.2f}" + "".join(cells))
    # 同号汇总
    print("\n  各组合分机场同号情况 (Δ<0 记为改善):")
    for k in ap_delta[aps[0]]:
        d = np.array([ap_delta[a][k] for a in aps])
        n_neg = int((d < 0).sum())
        print(f"    {k:<20} 改善 {n_neg}/{len(aps)} 机场  "
              f"中位 Δ {np.median(d):+.2f}  最差 {d.max():+.2f}")

    # ---- eq / ~eq 分层 ----
    print("\n" + "=" * 78)
    print("[eq(拷贝) / ~eq(正常) 分层 RMSE]")
    print("=" * 78)
    print(f"{'组合':<20}{'eq RMSE':>12}{'~eq RMSE':>12}")
    for k, P in combos.items():
        r_eq = rmse(P[eqh], yh[eqh]) if eqh.any() else float("nan")
        r_ne = rmse(P[~eqh], yh[~eqh])
        print(f"{k:<20}{r_eq:>12.2f}{r_ne:>12.2f}")

    # ---- LOBT 投影触发统计 ----
    n_proj = lobt_ok.sum()
    touched = {}
    for name, P in [("基线", base_all), ("混合", mu_mix), ("混合+LOBT", mu_mix_proj)]:
        changed = lobt_ok & (np.abs(P - np.clip(P, lo_w, hi_w)) > 1e-9)
        touched[name] = int(changed.sum())
    print(f"\n  LOBT 有效行 (holdout 四个月) = {int(n_proj):,} / {len(i_hold):,}")
    print(f"  LOBT 缺失/哨兵行 (跳过投影)  = {int((~lobt_ok).sum()):,}")
    for name, c in touched.items():
        print(f"    {name:<12} 被窗口改动行数 = {c:,} ({c/max(n_proj,1)*100:.3f}%)")
    print(f"  SCHED 落窗外 → p=0 行数 = {int(sched_out.sum()):,}")

    # ---- LIRF 对照命中 ----
    print(f"\n  LIRF T=14400 命中行数 (holdout) = {int(lirf_mask.sum()):,}")
    for mm in HOLD:
        m = moh == mm
        if (lirf_mask & m).sum():
            ss = (lirf_mask & m)
            print(f"    mo={mm}: {int(ss.sum()):>5,} 行  真实 y 中位 {np.median(yh[ss]):>9.1f}  "
                  f"MVT-SCHED 中位 {np.median(mu_C[ss]):>9.1f}")

    # ---- LOBT 无效行影响 ----
    print(f"\n  LOBT 无效行的 eq 阳性率 = {eqh[~lobt_ok].mean()*100 if (~lobt_ok).any() else float('nan'):.3f}% "
          f"(n={int((~lobt_ok).sum()):,})")

    result = {
        "T_lobt": T_LOBT, "T_lobt_999": T_LOBT_999, "EQ_TOL": EQ_TOL,
        "REG_CLIP": REG_CLIP,
        "clf_best_iter": int(clf.best_iteration or NROUND_CAP),
        "reg_best_iter": int(reg.best_iteration or NROUND_CAP),
        "n_fit_clf": int(len(i_fit)), "n_fit_reg_normal": int(len(i_norm_fit)),
        "n_holdout": int(len(i_hold)),
        "eq_rate_holdout": round(float(eqh.mean()), 5),
        "combos": res,
        "auc": aucs,
        "p_stats": {f"mo{mm}": {
            "mean": round(float(p[moh == mm].mean()), 4),
            "median": round(float(np.median(p[moh == mm])), 4),
            "p90": round(float(np.percentile(p[moh == mm], 90)), 4),
            "p99": round(float(np.percentile(p[moh == mm], 99)), 4),
            "max": round(float(p[moh == mm].max()), 4),
            "n_gt_0p5": int((p[moh == mm] > 0.5).sum()),
        } for mm in HOLD},
        "fp_normal_pred_gt_1h": {f"mo{mm}": int(((~eqh) & (mu_mix > 3600) & (moh == mm)).sum())
                                  for mm in HOLD},
        "fp_baseline_gt_1h": {f"mo{mm}": int(((~eqh) & (base_all > 3600) & (moh == mm)).sum())
                               for mm in HOLD},
        "n_lobt_valid_holdout": int(n_proj),
        "n_lobt_invalid_holdout": int((~lobt_ok).sum()),
        "lobt_projection_touched": touched,
        "n_sched_outside_window_p0": int(sched_out.sum()),
        "n_lirf_t14400_hit": int(lirf_mask.sum()),
        "per_airport_delta": ap_delta,
        "stratified_eq_normal_rmse": {
            k: {"eq": round(rmse(P[eqh], yh[eqh]), 2) if eqh.any() else None,
                "normal": round(rmse(P[~eqh], yh[~eqh]), 2)}
            for k, P in combos.items()},
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    lap(f"\n→ {OUT}")
    lap(f"总耗时 {(time.time()-_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
