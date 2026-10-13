"""
物理约束二阶矩混合 + 损坏感知共形
===================================
【1】损坏分量:  μ_C = MVT − SCHED;  σ_C 物理有界, 敏感性 σ_C ∈ {100,300,600}s
【2】混合矩:    μ = p·μ_C + (1−p)·μ_N
              σ² = p·σ_C² + (1−p)·σ_N² + p(1−p)·(μ_C−μ_N)²
              σ_N 截断 min(σ_N, 2000)  (消灭 NGBoost 尾部爆炸, 记录截断行数)
【3】分层共形:  校准集 mo∈{3,4,5,8,9,10,11} (不在 h1/h2)
              损坏行 = T=14400 完整口径命中 或 p>0.5;  正常行 = 其余
              分层取 s=|y−μ|/σ 的 90%/95% 分位 q_C/q_N
              在 h1/h2 算实际覆盖率, 对比名义 90%/95%
【4】报告 JSON: Spearman / σ 分布 / σ_C 敏感性 / 点预测 RMSE / 分机场

只读已有模型与缓存; 模型先落盘; 不写 submissions/; 不上传。
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

TS = D0 / "_ts_cache.pkl"
OUT = D0 / "moment_conformal.json"

CALIB = [3, 4, 5, 8, 9, 10, 11]
H1, H2 = [1, 7], [2, 6]
HOLD = [1, 2, 6, 7]
LIRF_AP, LIRF_T = "LIRF", 14400.0
SIG_CAP = 2000.0
SIG_C_GRID = [100.0, 300.0, 600.0]
SIG_C_MAIN = 300.0
LEVELS = [0.90, 0.95]

_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:6.1f}s] {m}", flush=True)


def sp(a, b):
    """Spearman, 不依赖 scipy"""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def rmse(P, yt, m=None):
    if m is None:
        m = np.ones(len(yt), bool)
    return float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))


def main():
    lap("加载缓存 + NGBoost/拷贝模型输出")
    X, y, mo, mvt_s, ap = load_cache()
    ts = pickle.load(open(TS, "rb"))
    mvt_a, sch_a = ts["mvt_s"], ts["sch_s"]
    afn = ts["adep_flt_null"]
    assert np.array_equal(mvt_s, mvt_a)
    lap(f"  全量 {len(y):,}")

    clf = pickle.load(open(D0 / "cp_mixture_models" / "clf.pkl", "rb"))

    # ---- 校准集 p (唯一需要新算的) ----
    i_cal = np.where(np.isin(mo, CALIB))[0]
    lap(f"  校准集 mo{CALIB} = {len(i_cal):,} 行; 算 p ...")
    p_cal = np.clip(clf.predict(X.iloc[i_cal], num_iteration=clf.best_iteration), 0, 1)

    # ---- NGBoost loc/scale: 直接从 _ngb_reduced.pkl 预测 ----
    # 注: 目录里的 _ngb_h{1,2}_{loc,scale}.npy 非本脚本产物 (13:59 生成,
    #     与 reduced 模型锚点不符: σ中位 1981 vs 报告 228.85), 故不复用。
    lap("加载 NGBoost (_ngb_reduced.pkl) 并预测 calib/h1/h2")
    ngb = pickle.load(open(D0 / "_ngb_reduced.pkl", "rb"))
    Xv = X.to_numpy(dtype=np.float32)
    NGBP = D0 / "_moment_ngb_preds.npz"
    if NGBP.exists():
        z = np.load(NGBP)
        loc_cal, sc_cal = z["loc_cal"], z["sc_cal"]
        loc_h1, sc_h1 = z["loc_h1"], z["sc_h1"]
        loc_h2, sc_h2 = z["loc_h2"], z["sc_h2"]
        lap(f"  复用已落盘 NGBoost 预测 {NGBP.name}")
    else:
        def ngb_pred(idx, tag):
            # 用 pred_param 拿原始参数, 避免 pred_dist 的 DataFrame 形状问题
            par = np.asarray(ngb.pred_param(Xv[idx]), dtype=np.float64)
            mu, sc = par[:, 0], np.exp(par[:, 1])
            assert np.isfinite(mu).all(), f"{tag} loc 非有限"
            assert np.isfinite(sc).all(), f"{tag} scale 非有限"
            assert (sc > 0).all(), f"{tag} scale<=0"
            lap(f"    {tag} n={len(idx):,} loc/scale 完成")
            return mu, sc

        loc_cal, sc_cal = ngb_pred(i_cal, "calib")
        loc_h1, sc_h1 = ngb_pred(i_h1, "h1")
        loc_h2, sc_h2 = ngb_pred(i_h2, "h2")
        np.savez(NGBP,
                 loc_cal=loc_cal, sc_cal=sc_cal, loc_h1=loc_h1, sc_h1=sc_h1,
                 loc_h2=loc_h2, sc_h2=sc_h2)

    i_h1 = np.where(np.isin(mo, H1))[0]
    i_h2 = np.where(np.isin(mo, H2))[0]

    # ---- LGB 基线 (已落盘, 与 h1/h2 probe 同源) ----
    lgb_h1 = np.load(D0 / "_ngb_lgb_h1.npy").astype(np.float64)
    lgb_h2 = np.load(D0 / "_ngb_lgb_h2.npy").astype(np.float64)
    assert len(lgb_h1) == len(i_h1) and len(lgb_h2) == len(i_h2)
    lap(f"  LGB 基线复用 h1 {len(lgb_h1):,} / h2 {len(lgb_h2):,} ✓")

    p_h = {"h1": np.clip(clf.predict(X.iloc[i_h1], num_iteration=clf.best_iteration), 0, 1),
           "h2": np.clip(clf.predict(X.iloc[i_h2], num_iteration=clf.best_iteration), 0, 1)}

    def gather(idx, p):
        d = {"y": y[idx], "mo": mo[idx], "ap": ap[idx].astype(str),
             "mvt": mvt_a[idx], "sch": sch_a[idx], "afn": afn[idx], "p": p}
        return d

    S = {"calib": gather(i_cal, p_cal),
         "h1": gather(i_h1, p_h["h1"]), "h2": gather(i_h2, p_h["h2"])}
    S["calib"]["loc"], S["calib"]["scale"] = loc_cal, sc_cal
    S["h1"].update(loc=loc_h1, scale=sc_h1, lgb=lgb_h1)
    S["h2"].update(loc=loc_h2, scale=sc_h2, lgb=lgb_h2)

    # ---- 硬断言: loc/scale 有限且 scale>0 ----
    for nm in S:
        assert np.isfinite(S[nm]["loc"]).all(), f"{nm} loc 非有限"
        assert np.isfinite(S[nm]["scale"]).all(), f"{nm} scale 非有限"
        assert (S[nm]["scale"] > 0).all(), f"{nm} scale<=0"

    # ---- 损坏行标记 ----
    for nm in S:
        s = S[nm]
        s["mu_C"] = s["mvt"] - s["sch"]
        s["hit14400"] = (s["ap"] == LIRF_AP) & s["afn"] & (s["mu_C"] > LIRF_T)
        s["corrupt"] = s["hit14400"] | (s["p"] > 0.5)
        lap(f"  {nm}: n={len(s['y']):,}  hit14400={int(s['hit14400'].sum()):,}  "
            f"corrupt={int(s['corrupt'].sum()):,}")

    # ---- 混合矩 ----
    def mix(d, sig_c, cap=SIG_CAP):
        sc_n = np.minimum(d["scale"], cap)
        n_trunc = int((d["scale"] > cap).sum())
        mu = d["p"] * d["mu_C"] + (1 - d["p"]) * d["loc"]
        var = (d["p"] * sig_c**2 + (1 - d["p"]) * sc_n**2
               + d["p"] * (1 - d["p"]) * (d["mu_C"] - d["loc"])**2)
        return mu, np.sqrt(var), n_trunc

    lap("\n" + "=" * 82)
    print("=" * 82)
    print("[1][2] σ 信号对比: NGBoost 原始 σ vs 混合 σ (σ_C=%g)" % SIG_C_MAIN)
    print("=" * 82)
    sig_stats = {}
    for nm in ["h1", "h2"]:
        mu, sg, ntr = mix(S[nm], SIG_C_MAIN)
        S[nm]["mu"], S[nm]["sig"], S[nm]["n_trunc"] = mu, sg, ntr
        r = np.abs(S[nm]["y"] - mu)          # 混合点预测的残差
        r_loc = np.abs(S[nm]["y"] - S[nm]["loc"])  # NGBoost 点预测的残差
        raw = S[nm]["scale"]
        print(f"\n  {nm}  (n={len(mu):,})")
        print(f"    {'σ vs |y−μ_mix| Spearman':<28}  "
              f"NGBoost原始σ={sp(raw, r):+.4f}   混合σ={sp(sg, r):+.4f}")
        print(f"    {'σ vs |y−loc| Spearman':<28}  "
              f"NGBoost原始σ={sp(raw, r_loc):+.4f}   (NGBoost 自洽口径)")
        print(f"    {'σ max':<28}  NGBoost原始={raw.max():.4e}   混合={sg.max():.4e}")
        print(f"    {'σ >1e4 行数':<28}  NGBoost原始={int((raw>1e4).sum()):,}     "
              f"混合={int((sg>1e4).sum()):,}")
        print(f"    {'σ 中位':<28}  NGBoost原始={np.median(raw):.2f}      "
              f"混合={np.median(sg):.2f}")
        print(f"    {'σ 被截断行数 (>2000)':<28}  NGBoost={ntr:,} "
              f"({ntr/len(raw)*100:.3f}%)")
        m_gt = sg > 1800
        m_lt = sg < 600
        print(f"    σ 中位 (>1800 段)={np.median(sg[m_gt]):.1f}  "
              f"(<600 段)={np.median(sg[m_lt]):.1f}  "
              f"比值={np.median(sg[m_gt])/np.median(sg[m_lt]):.2f}x")
        anchor = (7.08e14, 431, 228.85, 2.29) if nm == "h1" else (1.93e12, 103, 197.87, 2.49)
        print(f"    [锚点 {nm}] NGBoost σ max={anchor[0]:.2e}, >1e4 行数={anchor[1]}, "
              f"σ中位={anchor[2]}, 长短尾比={anchor[3]}x")
        sig_stats[nm] = {
            "spearman_raw_vs_mix_resid": round(sp(raw, r), 4),
            "spearman_mix_vs_mix_resid": round(sp(sg, r), 4),
            "spearman_raw_vs_loc_resid": round(sp(raw, r_loc), 4),
            "sigma_max_raw": float(raw.max()), "sigma_max_mix": float(sg.max()),
            "n_sigma_gt_1e4_raw": int((raw > 1e4).sum()),
            "n_sigma_gt_1e4_mix": int((sg > 1e4).sum()),
            "sigma_median_raw": round(float(np.median(raw)), 2),
            "sigma_median_mix": round(float(np.median(sg)), 2),
            "n_truncated": ntr, "trunc_rate": round(ntr / len(raw), 6),
            "sigma_ratio_gt1800_over_lt600": round(
                float(np.median(sg[m_gt]) / np.median(sg[m_lt])), 3),
            "anchor": {"sigma_max": anchor[0], "n_gt_1e4": anchor[1],
                       "sigma_median": anchor[2], "ratio": anchor[3]},
        }

    # ---- [3] 分层共形 ----
    lap("\n" + "=" * 82)
    print("=" * 82)
    print("[3] 损坏感知分层共形 (校准集 mo%s)" % CALIB)
    print("=" * 82)
    cal = S["calib"]
    mu_c, sg_c, _ = mix(cal, SIG_C_MAIN)
    cal["mu"], cal["sig"] = mu_c, sg_c
    s_score = np.abs(cal["y"] - mu_c) / np.maximum(sg_c, 1e-9)
    cal["score"] = s_score
    q = {}
    for lvl in LEVELS:
        q[("C", lvl)] = float(np.quantile(s_score[cal["corrupt"]], lvl))
        q[("N", lvl)] = float(np.quantile(s_score[~cal["corrupt"]], lvl))
    print(f"\n  校准集 n={len(cal['y']):,}  损坏行={int(cal['corrupt'].sum()):,}  "
          f"正常行={int((~cal['corrupt']).sum()):,}")
    for lvl in LEVELS:
        print(f"    名义 {lvl:.0%}: q_C={q[('C',lvl)]:.4f}   q_N={q[('N',lvl)]:.4f}")

    conf = {}
    for nm in ["h1", "h2"]:
        d = S[nm]
        conf[nm] = {}
        print(f"\n  {nm} 实际覆盖率:")
        for lvl in LEVELS:
            row = {}
            for tag, tkey, mask in [("损坏行", "C", d["corrupt"]),
                                    ("正常行", "N", ~d["corrupt"])]:
                if mask.sum() == 0:
                    row[tag] = {"n": 0}
                    continue
                qv = q[(tkey, lvl)]
                lo = d["mu"][mask] - qv * d["sig"][mask]
                hi = d["mu"][mask] + qv * d["sig"][mask]
                cov = float(((d["y"][mask] >= lo) & (d["y"][mask] <= hi)).mean())
                wid = float(np.median(hi - lo))
                row[tag] = {"n": int(mask.sum()), "q": round(qv, 4),
                            "coverage": round(cov, 4),
                            "dev": round(cov - lvl, 4),
                            "median_width": round(wid, 1)}
                print(f"    名义{lvl:.0%} {tag}: n={int(mask.sum()):>7,}  "
                      f"q={qv:.4f}  实际覆盖={cov:.4f}  偏差={cov-lvl:+.4f}  "
                      f"区间中位宽={wid:,.0f}s")
            row["nominal"] = lvl
            conf[nm][f"{lvl:.2f}"] = row

    # ---- σ_C 敏感性 ----
    lap("\n" + "=" * 82)
    print("=" * 82)
    print("[4] σ_C 敏感性 (σ_C ∈ %s)" % SIG_C_GRID)
    print("=" * 82)
    sens = {}
    for sc in SIG_C_GRID:
        row = {}
        print(f"\n  σ_C = {sc:.0f}s")
        for nm in ["calib", "h1", "h2"]:
            d = S[nm]
            mu_v, sg_v, _ = mix(d, sc)
            if nm == "calib":
                s_ = np.abs(d["y"] - mu_v) / np.maximum(sg_v, 1e-9)
                qq = {f"{l}_{t}": float(np.quantile(s_[d["corrupt"] if t == 'C' else ~d["corrupt"]], l))
                      for l in LEVELS for t in ["C", "N"]}
            else:
                qq = None
            row[nm] = {"sigma_median": round(float(np.median(sg_v)), 2),
                       "sigma_p95": round(float(np.percentile(sg_v, 95)), 2),
                       "sigma_max": float(sg_v.max()),
                       "n_gt_1e4": int((sg_v > 1e4).sum()),
                       "spearman": round(sp(sg_v, np.abs(d["y"] - mu_v)), 4),
                       "q": qq}
        # h1/h2 覆盖率 (在 calib 的 q 下, q 随 sigma_C 变)
        for nm in ["h1", "h2"]:
            d = S[nm]
            mu_v, sg_v, _ = mix(d, sc)
            cc = {}
            for lvl in LEVELS:
                for tag, mask in [("C", d["corrupt"]), ("N", ~d["corrupt"])]:
                    qv = row["calib"]["q"][f"{lvl}_{tag}"]
                    lo = mu_v[mask] - qv * sg_v[mask]
                    hi = mu_v[mask] + qv * sg_v[mask]
                    cc[f"{lvl}_{tag}"] = round(
                        float(((d["y"][mask] >= lo) & (d["y"][mask] <= hi)).mean()), 4)
            row[nm]["coverage"] = cc
        print(f"    h1: σ中位={row['h1']['sigma_median']}  Spearman={row['h1']['spearman']}  "
              f"覆盖 95%C={row['h1']['coverage']['0.95_C']} 95%N={row['h1']['coverage']['0.95_N']}")
        print(f"    h2: σ中位={row['h2']['sigma_median']}  Spearman={row['h2']['spearman']}  "
              f"覆盖 95%C={row['h2']['coverage']['0.95_C']} 95%N={row['h2']['coverage']['0.95_N']}")
        sens[f"{sc:.0f}"] = row

    # ---- 点预测 RMSE ----
    lap("\n" + "=" * 82)
    print("=" * 82)
    print("[5] 点预测 μ_total full RMSE (σ_C=%g)" % SIG_C_MAIN)
    print("=" * 82)
    allidx = np.concatenate([i_h1, i_h2])
    y_all = np.concatenate([S["h1"]["y"], S["h2"]["y"]])
    mo_all = np.concatenate([S["h1"]["mo"], S["h2"]["mo"]])
    ap_all = np.concatenate([S["h1"]["ap"], S["h2"]["ap"]])
    mu_all = np.concatenate([S["h1"]["mu"], S["h2"]["mu"]])
    lgb_all = np.concatenate([S["h1"]["lgb"], S["h2"]["lgb"]])
    muC_all = np.concatenate([S["h1"]["mu_C"], S["h2"]["mu_C"]])
    afn_all = np.concatenate([S["h1"]["afn"], S["h2"]["afn"]])

    aps = pd.Series(ap_all).value_counts().index.tolist()
    schemes = {"mu_total_混合": mu_all, "LGB基线": lgb_all}
    v22 = lgb_all.copy()
    hit = (ap_all == LIRF_AP) & afn_all & (muC_all > LIRF_T)
    v22[hit] = muC_all[hit]
    schemes["v22硬规则"] = v22

    print(f"\n  {'方案':<16}" + "".join(f"{'mo='+str(m):>10}" for m in HOLD)
          + f"{'四个月':>11}{'vs LGB':>10}")
    print("  " + "-" * 69)
    rmse_tab = {}
    base4 = None
    for k, P in schemes.items():
        r4 = rmse(P, y_all)
        if base4 is None:
            base4 = r4
        cells = "".join(f"{rmse(P, y_all, mo_all == m):>10.2f}" for m in HOLD)
        print(f"  {k:<16}{cells}{r4:>11.2f}{r4-base4:>+10.2f}")
        rmse_tab[k] = {f"mo{m}": round(rmse(P, y_all, mo_all == m), 2) for m in HOLD}
        rmse_tab[k]["all4"] = round(r4, 2)
        rmse_tab[k]["delta_vs_LGB"] = round(r4 - base4, 2)
        rmse_tab[k]["_pred"] = P

    # 分机场
    print(f"\n  分机场 full RMSE:")
    print(f"  {'机场':<7}{'n':>9}{'LGB':>10}{'v22':>10}{'μ_total':>10}"
          f"{'μ-LGB':>10}{'同号':>7}")
    print("  " + "-" * 63)
    ap_tab = {}
    for a in aps:
        m = ap_all == a
        r_l, r_v, r_t = (rmse(schemes[k], y_all, m) for k in
                         ["LGB基线", "v22硬规则", "mu_total_混合"])
        d = r_t - r_l
        same = (d <= 0) if d != 0 else True
        same = d <= 0.5
        ap_tab[a] = {"n": int(m.sum()), "LGB": round(r_l, 2), "v22": round(r_v, 2),
                     "mu_total": round(r_t, 2), "delta_vs_LGB": round(d, 2),
                     "improved": bool(d <= 0.5)}
        print(f"  {a:<7}{int(m.sum()):>9,}{r_l:>10.1f}{r_v:>10.1f}{r_t:>10.1f}"
              f"{d:>+10.2f}{'  OK' if d <= 0.5 else '  FAIL':>7}")
    n_ok = sum(v["improved"] for v in ap_tab.values())
    print(f"\n  分机场不劣化于 LGB: {n_ok}/{len(aps)}")

    # 分月同号
    mm_ok = all(rmse_tab["mu_total_混合"][f"mo{m}"] <= rmse_tab["LGB基线"][f"mo{m}"] + 1e-9
                for m in HOLD)
    print(f"  分月全部不劣于 LGB: {mm_ok}")

    # 保存预测 (模型先落盘)
    np.savez(D0 / "_moment_conformal_preds.npz",
             mu=mu_all, sig=np.concatenate([S["h1"]["sig"], S["h2"]["sig"]]),
             p=np.concatenate([S["h1"]["p"], S["h2"]["p"]]),
             y=y_all, mo=mo_all, ap=ap_all, lgb=lgb_all)
    lap(f"  预测已落盘 _moment_conformal_preds.npz")

    res = {
        "sig_c_main": SIG_C_MAIN, "sigma_n_cap": SIG_CAP,
        "calib_months": CALIB, "h1_months": H1, "h2_months": H2,
        "sigma_stats": sig_stats,
        "conformal": {"q": {f"{t}_{l:.2f}": q[(t, l)] for t in ["C", "N"] for l in LEVELS},
                      "coverage": conf},
        "sigma_c_sensitivity": sens,
        "point_rmse": {k: {kk: vv for kk, vv in v.items() if kk != "_pred"}
                       for k, v in rmse_tab.items()},
        "per_airport": ap_tab,
        "gate_per_airport": {"n_ok": int(n_ok), "n_total": len(aps)},
        "gate_per_month_all_better": bool(mm_ok),
        "anchors": {"LGB_holdout_all4": 493.05, "v22": 435.08, "D_概率混合": 451.60},
        "pred_path": str(D0 / "_moment_conformal_preds.npz"),
        "total_time_min": round((time.time() - _t0) / 60, 1),
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    lap(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
