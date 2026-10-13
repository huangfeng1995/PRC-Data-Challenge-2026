"""
NGBoost 缩小版 σ 信号探针 (不重跑 LGB 基线)
==========================================
- 基线 (LGB h1=568.49, h2=398.40) — 来自之前 v18/v24 全量训练, 用户已知, 不重训
- NGBoost 缩小版:
  · fit 月 2,3,4,5,6,8,9,10,11,12 分层抽样 ~30 万行 (按 ADEP×月比例)
  · TAXITIME>10800 全部纳入 (极端尾部不抽样)
  · 估: loc μ, scale σ (Normal, DecisionTreeRegressor max_depth=3, lr=0.03, n_est=500)
- ES: fit 月最后 10% (按 MVT_TIME 排序), 断言月份 ⊆ {11,12} 且不含 1,7
- 预测: h1 (Jan+Jul 全量 344,419 行)
- 硬断言: loc/scale 非有限, scale>0 才失败; scale max/1e4 只记录不拦截
- σ 信号主指标: σ vs |残差| 的 Spearman; σ 在 >1800 / <600 段的中位比
- 模型先 pickle 落盘再评估

红线: 不上传, 不改 label, 不写 submissions/
"""
from __future__ import annotations
import sys, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*ngboost.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="ngboost")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
CACHE = D0 / "_qinv_feats.pkl"
OUT = D0 / "ngb_probe_reduced.json"
NGB_PKL = D0 / "_ngb_reduced.pkl"

FIT_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD1 = [1, 7]
CLIP_LO, CLIP_HI = 30.0, 10800.0
SEED = 42
N_SUBSAMPLE = 300_000           # 缩小版抽样目标
N_ESTIMATORS = 500
LR = 0.03
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def load_cache():
    """加载 _qinv_feats.pkl, 复用 lgb_full_train.py 的 FEATS71 解析"""
    import re
    blob = pickle.load(open(CACHE, "rb"))
    X = blob["X"]; y = blob["y"].astype(np.float64)
    mo = blob["mo"].astype(np.int64)
    mvt_s = blob["mvt_s"].astype(np.float64)
    adep = blob["ap"]  # 字符串数组, 不是 category
    src = (D0 / "lgb_full_train.py").read_text()
    f71 = re.findall(r'"([^"]+)"',
                     re.search(r"FEATS71\s*=\s*\[(.*?)\]", src, re.S).group(1))
    assert len(f71) == 71
    X = X[[c for c in X.columns if not c.startswith("q_")]]
    assert list(X.columns) == f71
    X_num = X.copy()
    cats = [c for c in X_num.columns if str(X_num[c].dtype) == "category"]
    for c in cats:
        X_num[c] = X_num[c].cat.codes.astype(np.float32)
    nan_cols = [c for c in X_num.columns if X_num[c].isna().any()]
    if nan_cols:
        X_num[nan_cols] = X_num[nan_cols].fillna(-999.0)
    assert not X_num.isna().any().any()
    return X_num, y, mo, mvt_s, adep


def split_es(mo, mvt_s):
    """ES: fit 月最后 10% (按 MVT_TIME 排序), 断言 ⊆ {11,12} 且不含 1,7.
    fit_all 不剔 TAXITIME>10800 (NGBoost 用全部真实尾部, 包括极端长尾)"""
    fm = np.isin(mo, FIT_MONTHS)
    i_tr = np.where(fm)[0]
    o = i_tr[np.argsort(mvt_s[i_tr], kind="stable")]
    cut = int(len(o) * 0.9)
    i_fit_all, i_es = o[:cut], o[cut:]
    es_m = sorted(set(mo[i_es].tolist()))
    assert set(es_m) <= {11, 12}, f"ES 月份异常: {es_m}"
    assert not np.isin(mo[i_es], HOLD1).any(), "ES 泄漏到 holdout"
    assert not np.intersect1d(i_fit_all, i_es).size
    return i_fit_all, i_es, es_m


def stratified_subsample(i_fit_all, mo, adep, y, n_target, ext_mask, seed):
    """按 ADEP×月比例分层抽样 (i_fit_all 内), 全部 ext_mask 行强制纳入.
    ext_mask = (y > 10800) — 训练集中真实 >10800 极端尾部."""
    rng = np.random.RandomState(seed)
    i_ext = i_fit_all[ext_mask[i_fit_all]]
    i_rest = i_fit_all[~ext_mask[i_fit_all]]

    n_need = max(0, n_target - len(i_ext))
    adep_i = adep[i_rest]
    mo_i = mo[i_rest]
    keys = pd.DataFrame({"ADEP": adep_i, "mo": mo_i})
    grp = keys.groupby(["ADEP", "mo"]).indices
    sampled = []
    for key, idx_local in grp.items():
        quota = int(round(n_need * len(idx_local) / len(i_rest)))
        if quota <= 0:
            continue
        if len(idx_local) <= quota:
            sampled.extend(idx_local.tolist())
        else:
            sel = rng.choice(len(idx_local), size=quota, replace=False)
            sampled.extend(idx_local[sel].tolist())
    sampled_idx = i_rest[np.array(sampled)]
    final = np.concatenate([i_ext, sampled_idx])
    return final, len(i_ext), len(sampled_idx)


def rmse(P, yt, m=None):
    m = np.ones(len(yt), bool) if m is None else m
    return float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return round(float(r.statistic), 4), round(float(r.pvalue), 4)


def per_band_report(P, yt):
    """(>1800 / <600 / 其他) 段 RMSE"""
    m_hi = yt > 1800
    m_lo = yt < 600
    out = {
        "n_gt1800": int(m_hi.sum()),
        "n_lt600": int(m_lo.sum()),
        "rmse_gt1800": round(rmse(P, yt, m_hi), 2) if m_hi.sum() else None,
        "rmse_lt600": round(rmse(P, yt, m_lo), 2) if m_lo.sum() else None,
    }
    return out


def main():
    lap("步骤1: 加载缓存")
    X_num, y, mo, mvt_s, adep = load_cache()
    i_fit_all, i_es, es_m = split_es(mo, mvt_s)
    # fit 不剔 TX>10800 (NGBoost 训练用全部真实尾部, 包括极端长尾)
    # 仅剔 TX<=0 (无效数据)
    keep = (y[i_fit_all] > 0)
    i_fit_clean = i_fit_all[keep]
    n_dropped = int((~keep).sum())
    lap(f"fit 全量 {len(i_fit_all):,} (仅剔 TX<=0 后剩 {len(i_fit_clean):,}, 剔 {n_dropped:,})  "
        f"| ES {len(i_es):,} 月 {es_m}")

    i_h1 = np.where(np.isin(mo, HOLD1))[0]
    lap(f"holdout1 (Jan+Jul) {len(i_h1):,}")

    # 缩小版抽样
    lap("\n步骤2: 代表性缩小版抽样 (按 ADEP×月比例, 全部 TX>10800 纳入)")
    ext_mask = (y > 10800)
    i_sub, n_ext, n_sampled = stratified_subsample(
        i_fit_clean, mo, adep, y, N_SUBSAMPLE, ext_mask, seed=SEED)
    n_sub_total = len(i_sub)
    y_sub = y[i_sub]
    n_sub_ext = int((y[i_sub] > 10800).sum())
    # 检查 10 个机场都覆盖
    aps_present = pd.Series(adep[i_sub]).value_counts().index.tolist()
    top10_aps = ["LIRF", "EGLL", "LTFM", "LFPG", "EHAM",
                 "EDDF", "LFPG", "LEBL", "EDDM", "LEMD"]
    coverage_10 = [a for a in top10_aps if a in aps_present]
    lap(f"抽样结果: n={n_sub_total:,} (ext={n_sub_ext:,} TX>10800, "
        f"sampled={n_sampled:,})  覆盖10机场: {len(coverage_10)}/10")

    # NGBoost 训练 (先模型, 再报告 — 与 ngb_probe.py 一致)
    lap(f"\n步骤4: NGBoost 训练 ({N_ESTIMATORS} estimators, Normal, "
        f"DecisionTree max_depth=3, lr={LR})")
    from ngboost import NGBRegressor
    from ngboost.distns import Normal as Dist
    from sklearn.tree import DecisionTreeRegressor

    if NGB_PKL.exists():
        m = pickle.load(open(NGB_PKL, "rb"))
        fit_s = 0.0
        lap(f"复用已训模型 {NGB_PKL.name}")
    else:
        t0 = time.time()
        m = NGBRegressor(Dist=Dist,
                         Base=DecisionTreeRegressor(max_depth=3, random_state=SEED),
                         n_estimators=N_ESTIMATORS, learning_rate=LR, verbose=False)
        m.fit(X_num.iloc[i_sub], y_sub,
              X_val=X_num.iloc[i_es], Y_val=y[i_es])
        fit_s = time.time() - t0
        pickle.dump(m, open(NGB_PKL, "wb"), protocol=4)
        lap(f"NGBoost 训练完成 {fit_s:.1f}s  已落盘 {NGB_PKL.name}")

    # 预测 h1 (Jan+Jul 全量)
    lap("\n步骤5: 预测 holdout1 (Jan+Jul 全量 344k 行)")
    t_pred = time.time()
    D_h1 = m.pred_dist(X_num.iloc[i_h1])
    loc_h1 = D_h1.loc
    scale_h1 = D_h1.scale
    pred_s = time.time() - t_pred
    lap(f"  预测耗时 {pred_s:.1f}s")

    # 硬断言 (loc/scale 必须有限, scale 必须 >0)
    assert np.isfinite(loc_h1).all(), "loc 有非有限值"
    assert np.isfinite(scale_h1).all(), "scale 有非有限值"
    assert (scale_h1 > 0).all(), "scale 有非正值"
    # 软报告 (σ max/1e4 只报告不拦截)
    y_h1 = y[i_h1]
    P_h1 = np.clip(loc_h1, CLIP_LO, CLIP_HI)
    resid_h1 = np.abs(P_h1 - y_h1)
    rho, pv = spearman(scale_h1, resid_h1)

    m_gt1800 = y_h1 > 1800
    m_lt600 = y_h1 < 600
    sigma_med_gt = float(np.median(scale_h1[m_gt1800])) if m_gt1800.any() else None
    sigma_med_lt = float(np.median(scale_h1[m_lt600])) if m_lt600.any() else None
    sigma_ratio = (sigma_med_gt / sigma_med_lt) if (sigma_med_gt and sigma_med_lt) else None

    sigma_max = float(scale_h1.max())
    n_sigma_gt1e4 = int((scale_h1 > 1e4).sum())

    # loc RMSE (主指标, 仅主对比)
    loc_rmse_full = rmse(P_h1, y_h1)
    loc_rmse_core = rmse(P_h1, y_h1, (y_h1 > 0) & (y_h1 <= 10800))
    loc_rmse_short = rmse(P_h1, y_h1, (y_h1 > 0) & (y_h1 <= 1800))
    loc_rmse_long = rmse(P_h1, y_h1, (y_h1 > 1800) & (y_h1 <= 10800))

    band_report = per_band_report(P_h1, y_h1)

    # LOC 与 (best_estimates) 关系
    best_iter = getattr(m, "best_val_loss_itr", None)

    result = {
        "task": "ngb_probe_reduced_30w",
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LR,
        "fit_seconds": round(fit_s, 1),
        "n_fit": int(n_sub_total),
        "n_fit_ext_incl_gt_10800": n_sub_ext,
        "n_fit_sampled_random": n_sampled,
        "coverage_10_airports": len(coverage_10),
        "n_holdout1": int(len(i_h1)),
        "best_iter": int(best_iter) if best_iter is not None else None,
        "loc_RMSE_full": round(loc_rmse_full, 2),
        "loc_RMSE_core": round(loc_rmse_core, 2),
        "loc_RMSE_short": round(loc_rmse_short, 2),
        "loc_RMSE_long": round(loc_rmse_long, 2),
        "sigma_vs_absresid_spearman": rho,
        "sigma_vs_absresid_p": pv,
        "sigma_median_gt1800": round(sigma_med_gt, 3) if sigma_med_gt is not None else None,
        "sigma_median_lt600": round(sigma_med_lt, 3) if sigma_med_lt is not None else None,
        "sigma_ratio_gt1800_over_lt600": round(sigma_ratio, 3) if sigma_ratio is not None else None,
        "sigma_max": round(sigma_max, 1),
        "n_sigma_gt_1e4": n_sigma_gt1e4,
        "loc_median": round(float(np.median(loc_h1)), 1),
        "scale_median": round(float(np.median(scale_h1)), 3),
        "predict_seconds": round(pred_s, 1),
        "band_report": band_report,
        "baseline_lgb_full_h1_RMSE": 568.49,  # 用户给定, 不重训
        "signal_strong": bool(rho > 0.3),
        "model_path": str(NGB_PKL),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # 输出报告
    print("\n" + "=" * 70)
    print("[NGBoost 缩小版 σ 信号探针] holdout1 (Jan+Jul)")
    print("=" * 70)
    print(f"  抽样: n={n_sub_total:,} (ext TX>10800={n_sub_ext:,}, random={n_sampled:,})  "
          f"覆盖10机场: {len(coverage_10)}/10")
    print(f"  训练耗时: {fit_s:.1f}s (predict {pred_s:.1f}s)")
    print(f"  best_iter: {best_iter}")
    print(f"  模型: {NGB_PKL}")
    print()
    print(f"  --- 主指标: loc μ ---")
    print(f"  full RMSE = {loc_rmse_full:.2f}  (LGB 基线=568.49, +{(loc_rmse_full-568.49):.2f})")
    print(f"  core RMSE = {loc_rmse_core:.2f}")
    print(f"  short RMSE = {loc_rmse_short:.2f}")
    print(f"  long RMSE = {loc_rmse_long:.2f}")
    print(f"  分档 loc RMSE: >1800段={band_report['rmse_gt1800']}, <600段={band_report['rmse_lt600']}")
    print()
    print(f"  --- σ 信号 (主目标) ---")
    sig_marker = "✓ >0.3 信号强" if rho > 0.3 else "✗ ≤0.3 信号弱"
    print(f"  σ vs |残差| Spearman: {rho:+.4f}  (p={pv:.2e})  {sig_marker}")
    print(f"  σ 中位数: >1800 段 {sigma_med_gt:.3f}s vs <600 段 {sigma_med_lt:.3f}s  "
          f"→ 比值 {sigma_ratio:.2f}x")
    print(f"  σ max = {sigma_max:.1f}  (>1e4 行数: {n_sigma_gt1e4:,})")
    print(f"  σ 中位数 = {float(np.median(scale_h1)):.3f}")
    print()
    print(f"  --- 边界信息 (只报告, 不作硬断言) ---")
    print(f"  n_holdout1 = {len(i_h1):,}")
    print(f"  → {OUT}")
    print(f"  总耗时: {(time.time() - _t0)/60:.1f} min")


if __name__ == "__main__":
    main()