"""
gate_ab.py — 学习出来的双模态门控 (LGB + 时间恒等式反演)

p_data : 干净口径 LGB 回归 (train 2-6,8-12 / ES 11-12 / holdout 1+7)
p_phys : MVT - SCHED  (确定性, 无需模型)
w      : 二分类门控, label = (|p_phys-y| < |p_data_oof-y|)
pred   : w*p_phys + (1-w)*p_data

防泄漏:
  - p_data_oof 由"按月折"生成 (训练月留一月, 其余月训)
  - 门控只用训练月 (2-6,8-12) 的 OOF 行, 绝不含 1/7 月
  - 不写死 if ADEP=='LIRF'; ADEP 仅作普通类别特征
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "gate_ab.json"
CACHE = D0 / "_qinv_feats.pkl"
RAWCACHE = D0 / "_gate_rawcols.pkl"
NS = 1_000_000_000
CAP = 2000
SEEDS = [42, 123, 2024]
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, float("inf"))]
BLAB = ["[0,600)", "[600,1800)", "[1800,3600)", "[3600,10800]", ">10800"]
SENT = -9223372037
_t0 = time.time()
def log(*a): print(" ".join(str(x) for x in a), flush=True)
def lap(m=""): log(f"[{time.time()-_t0:7.1f}s] {m}")

def mf(m: int) -> Path:
    ny, n2 = (2025, m + 1) if m < 12 else (2026, 1)
    return RAW / f"training_2025-{m:02d}-01_{ny}-{n2:02d}-01.parquet"

# ============ 特征与原始列 ============
def load_rawcols():
    if RAWCACHE.exists():
        with open(RAWCACHE, "rb") as f: return pickle.load(f)
    cols = ["MVT_ID_mvt", "PHASE_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
            "AOBT_3_flt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"]
    d = {}
    for m in sorted(TRAIN_MONTHS + HOLD_MONTHS):
        g = pd.read_parquet(mf(m), columns=cols)
        g = g[g["PHASE_mvt"] == "DEP"]
        d[m] = g.drop(columns=["PHASE_mvt"]).reset_index(drop=True)
    with open(RAWCACHE, "wb") as f: pickle.dump(d, f)
    return d

def main():
    with open(CACHE, "rb") as f:
        blob = pickle.load(f)
    X = blob["X"]; y = blob["y"]; mo = blob["mo"]; ap = blob["ap"]; mvt_s = blob["mvt_s"]
    lap(f"X={X.shape} train={np.isin(mo,TRAIN_MONTHS).sum():,} holdout={np.isin(mo,HOLD_MONTHS).sum():,}")

    # 原始列按行序对齐 (qinv_ab 的 X 与 allraw 均由同一循环按月 concat, 行序一致)
    rc = load_rawcols()
    allraw = pd.concat([rc[m].assign(month=m) for m in sorted(rc)], ignore_index=True)
    assert len(allraw) == len(X), f"行数不一致 raw={len(allraw)} X={len(X)}"
    # mo 来自 MVT 时刻, month 来自文件名; 跨月航班会差 1-2 行, 允许
    n_mis = int((allraw["month"].to_numpy(np.int64) != mo).sum())
    assert n_mis <= 50, f"月份错位过多: {n_mis}"
    assert (np.isin(allraw["month"].to_numpy(np.int64), TRAIN_MONTHS).sum()
            == np.isin(mo, TRAIN_MONTHS).sum()), "train/holdout 划分不一致"
    lap(f"原始列按行序对齐 {len(allraw):,} 行 ✓ (月份差异 {n_mis} 行, 跨月航班)")

    def tos(s):
        # 与 qinv_ab 一致: 先转 naive datetime64[ns] 再取 int64 秒
        v = pd.to_datetime(s, utc=True, errors="coerce")
        v = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
        return v.astype("int64") // NS
    sched_s = tos(allraw["SCHED_TIME_UTC_mvt"]).to_numpy(np.float64)
    mvt2 = tos(allraw["MVT_TIME_UTC_mvt"]).to_numpy(np.float64)
    assert np.isfinite(mvt2).mean() > 0.99, "MVT 转换失败"
    assert np.nanmedian(mvt2) < 2e9, f"MVT 秒值异常 中位={np.nanmedian(mvt2):.3e}"
    p_phys = mvt2 - sched_s                 # 时间恒等式反演
    assert np.isfinite(p_phys).all() and (p_phys > 0).mean() > 0.99
    lap(f"p_phys 区间 [{np.nanmin(p_phys):.0f}, {np.nanmax(p_phys):.0f}]")

    # 门控用的显式可见标志 (ranking 侧同样可见)
    def _null(s):
        v = s.astype(str).str.strip()
        return s.isna().to_numpy() | v.isin(["", "None", "nan", "NaT"]).to_numpy()
    nm_adep = _null(allraw["ADEP_flt"])
    nm_actype = _null(allraw["AIRCRAFT_TYPE_mvt"])
    aobt = tos(allraw["AOBT_3_flt"])
    nm_aobt = ~np.isfinite(aobt) | (aobt <= SENT + 86400)
    mvtdelay = np.where(np.isfinite(p_phys), p_phys, 0.0)
    X["g_nm_adep"] = nm_adep.astype(np.int8)
    X["g_nm_actype"] = nm_actype.astype(np.int8)
    X["g_nm_aobt"] = nm_aobt.astype(np.int8)
    X["g_mvtdelay"] = mvtdelay.astype(np.float32)
    X["g_mvtdelay_x_nm"] = (mvtdelay * nm_adep).astype(np.float32)
    GATE_EXTRA = ["g_nm_adep", "g_nm_actype", "g_nm_aobt", "g_mvtdelay", "g_mvtdelay_x_nm"]
    # 损坏行判定 (仅用于诊断, 不作特征): |BLOCK-SCHED|<=60
    d_blk_sched = np.abs((p_phys - y))
    corrupt = np.isfinite(d_blk_sched) & (d_blk_sched <= 60)
    lap(f"损坏行(|BLOCK-SCHED|<=60): 全量 {corrupt.sum():,}  训练月 {np.isin(mo,TRAIN_MONTHS)[corrupt].sum():,}"
        f"  holdout {np.isin(mo,HOLD_MONTHS)[corrupt].sum():,}")

    tr = np.isin(mo, TRAIN_MONTHS)
    ho = np.isin(mo, HOLD_MONTHS)
    fit_idx = np.where(tr)[0][np.argsort(mvt_s[tr], kind="stable")]
    cut = int(len(fit_idx) * 0.9)
    fit_idx, es_idx = fit_idx[:cut], fit_idx[cut:]
    es_months = sorted(set(mo[es_idx].tolist()))
    assert 1 not in es_months and 7 not in es_months, f"ES 含 holdout 月 {es_months}"
    ho_idx = np.where(ho)[0]
    lap(f"拟合 {len(fit_idx):,} / ES {len(es_idx):,} {es_months} ✓")

    # ============ 1. 按月折 OOF (仅训练月) ============
    OOF = D0 / "_gate_oof.npy"
    if OOF.exists():
        p_oof = np.load(OOF); lap("OOF 缓存命中")
    else:
        p_oof = np.full(len(y), np.nan)
        fold_lb = []
        for k, fm in enumerate(TRAIN_MONTHS):
            t1 = time.time()
            va = np.where(mo == fm)[0]
            fit = np.setdiff1d(fit_idx, va, assume_unique=False)
            pm = dict(LGB_PARAMS); pm["seed"] = 42
            dtr = lgb.Dataset(X.iloc[fit], label=y[fit], categorical_feature=LOW_CARD_CATS)
            dva = lgb.Dataset(X.iloc[va], label=y[va], categorical_feature=LOW_CARD_CATS)
            b = lgb.train(pm, dtr, num_boost_round=CAP, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            p_oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
            fold_lb.append(b.best_iteration)
            lap(f"  OOF fold m{fm:02d}: n_va={len(va):,} best_iter={b.best_iteration} ({time.time()-t1:.0f}s)")
        np.save(OOF, p_oof)
    m_tr = tr & np.isfinite(p_oof)
    assert m_tr.sum() > 0
    lap(f"OOF 覆盖训练月 {m_tr.sum():,} 行 (月 {sorted(set(mo[m_tr].tolist()))})")

    # ============ 2. 门控标签与训练 ============
    e_phys = np.abs(p_phys - y)
    e_data = np.abs(p_oof - y)
    g_lab = (e_phys < e_data).astype(np.int8)
    gate = X.copy()
    gate["g_oof_pred"] = np.where(np.isfinite(p_oof), p_oof, np.nan).astype(np.float32)
    gate["g_oof_err"] = np.where(np.isfinite(p_oof), e_data, np.nan).astype(np.float32)
    gcols = list(X.columns) + ["g_oof_pred", "g_oof_err"]
    gtr = m_tr
    lap(f"门控训练集 {gtr.sum():,} 行, 正样本率 {g_lab[gtr].mean()*100:.2f}%")

    # 3a 逻辑回归 (仅数值列)
    num_cols = [c for c in gcols if c not in LOW_CARD_CATS + ["ADEP_mvt", "RUNWAY_mvt"]]
    A = gate.iloc[np.where(gtr)[0]][num_cols].to_numpy(np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(A)
    lr = LogisticRegression(max_iter=400, C=1.0, n_jobs=-1)
    lr.fit(sc.transform(A), g_lab[gtr])
    lap("逻辑回归门控已训练")

    # 3b LGB 分类器
    gp = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
          "num_leaves": 63, "max_depth": 9, "min_data_in_leaf": 200,
          "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
          "verbose": -1, "num_threads": -1, "seed": 42}
    t1 = time.time()
    dg = lgb.Dataset(gate.iloc[np.where(gtr)[0]][gcols], label=g_lab[gtr],
                     categorical_feature=LOW_CARD_CATS)
    dvg = lgb.Dataset(gate.iloc[es_idx][gcols], label=g_lab[es_idx], reference=dg)
    bg = lgb.train(gp, dg, num_boost_round=600, valid_sets=[dvg],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    lap(f"LGB 门控 best_iter={bg.best_iteration} ({time.time()-t1:.0f}s)")

    # ============ 4. holdout p_data (3 seed) ============
    preds = []
    for sd in SEEDS:
        t1 = time.time()
        pm = dict(LGB_PARAMS); pm["seed"] = sd
        pm["bagging_seed"] = sd + 1; pm["feature_fraction_seed"] = sd + 2
        dtr = lgb.Dataset(X.iloc[fit_idx], label=y[fit_idx], categorical_feature=LOW_CARD_CATS)
        dva = lgb.Dataset(X.iloc[es_idx], label=y[es_idx], reference=dtr)
        b = lgb.train(pm, dtr, num_boost_round=CAP, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        pr = np.clip(b.predict(X.iloc[ho_idx]).astype(np.float64), 30.0, 10800.0)
        preds.append(pr)
        lap(f"  p_data seed{sd}: best_iter={b.best_iteration} ({time.time()-t1:.0f}s)")

    # ============ 5. 门控权重 ============
    Gh = gate.iloc[ho_idx]
    w_lgb = bg.predict(Gh[gcols], num_iteration=bg.best_iteration).astype(np.float64)
    w_lr = lr.predict_proba(sc.transform(
        np.nan_to_num(Gh[num_cols].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)))[:, 1]
    lap(f"门控权重 holdout: LGB mean={w_lgb.mean():.4f}  LR mean={w_lr.mean():.4f}")

    # 门控判别力 (训练月, 折外)
    w_oof = bg.predict(gate.iloc[np.where(gtr)[0]][gcols],
                       num_iteration=bg.best_iteration).astype(np.float64)
    gate["w"] = np.nan
    gate.loc[np.where(gtr)[0], "w"] = w_oof
    auc_g = roc_auc_score(g_lab[gtr], w_oof)
    ap_g = average_precision_score(g_lab[gtr], w_oof)
    # 关键诊断: 损坏行 vs 正常行的 w 分布 (不点名机场)
    cw = gate.loc[corrupt & tr, "w"].to_numpy()
    nw = gate.loc[(~corrupt) & tr, "w"].to_numpy()
    diag = {"n_corrupt_train": int((corrupt & tr).sum()),
            "w_corrupt": {"n": int(len(cw)), "mean": float(np.mean(cw)),
                          "p10": float(np.percentile(cw, 10)), "p50": float(np.percentile(cw, 50)),
                          "p90": float(np.percentile(cw, 90))},
            "w_normal": {"n": int(len(nw)), "mean": float(np.mean(nw)),
                         "p50": float(np.percentile(nw, 50)),
                         "p90": float(np.percentile(nw, 90))},
            "w_sep_auc_corrupt_vs_normal": float(
                roc_auc_score(np.r_[np.ones(len(cw)), np.zeros(len(nw))], np.r_[cw, nw]))}
    lap(f"门控判别: 折外 AUC={auc_g:.4f} AP={ap_g:.4f}")
    lap(f"  损坏行 w: n={diag['w_corrupt']['n']} mean={diag['w_corrupt']['mean']:.4f} "
        f"p50={diag['w_corrupt']['p50']:.4f} p90={diag['w_corrupt']['p90']:.4f}")
    lap(f"  正常行 w: n={diag['w_normal']['n']} mean={diag['w_normal']['mean']:.4f} "
        f"p50={diag['w_normal']['p50']:.4f}")
    lap(f"  损坏 vs 正常 分离 AUC={diag['w_sep_auc_corrupt_vs_normal']:.4f}")

    # ============ 6. 评估 ============
    yt = y[ho_idx]; ap_h = ap[ho_idx]; mo_h = mo[ho_idx]
    core = (yt > 0) & (yt <= 10800)
    phys_h = p_phys[ho_idx]
    def ev(P, tag):
        r = lambda msk: float(np.sqrt(np.mean((P[msk] - yt[msk]) ** 2)))
        o = {"tag": tag, "full": round(r(np.ones(len(yt), bool)), 2), "core": round(r(core), 2),
             "jan": round(r(mo_h == 1), 2), "jul": round(r(mo_h == 7), 2), "bins": [],
             "per_airport_core": {}}
        for (lo, hi), lab in zip(BINS, BLAB):
            msk = (yt >= lo) & (yt < hi) if hi != float("inf") else (yt >= lo)
            o["bins"].append({"bin": lab, "n": int(msk.sum()),
                              "rmse": round(r(msk), 2) if msk.sum() >= 5 else None})
        for a in pd.Series(ap_h).value_counts().index:
            msk = (ap_h == a) & core
            if msk.sum() >= 500:
                o["per_airport_core"][str(a)] = {"n": int(msk.sum()), "rmse": round(r(msk), 2)}
        return o
    def manual(P, T=14400):
        m = nm_adep[ho_idx] & (phys_h > T)
        Q = P.copy(); Q[m] = phys_h[m]
        return Q, int(m.sum())

    res = {"gate_diag": diag, "gate_auc_oof": round(auc_g, 4), "gate_ap_oof": round(ap_g, 4),
           "n_holdout": int(len(yt)), "seeds": {}}
    A_full = []; A_core = []
    for i, sd in enumerate(SEEDS):
        P = preds[i]
        Qman, nman = manual(P)
        Fuse = {"lgb": w_lgb * phys_h + (1 - w_lgb) * P,
                "lr": w_lr * phys_h + (1 - w_lr) * P}
        e_raw = ev(P, f"p_data_raw_s{sd}")
        e_man = ev(Qman, f"manual_s{sd}")
        e_lg = ev(Fuse["lgb"], f"gate_lgb_s{sd}")
        e_lr = ev(Fuse["lr"], f"gate_lr_s{sd}")
        res["seeds"][str(sd)] = {"raw": e_raw, "manual": e_man, "gate_lgb": e_lg, "gate_lr": e_lr,
                                 "n_manual_changed": nman}
        A_full.append((e_raw["core"], e_man["core"], e_lg["core"], e_lr["core"]))
    # 三 seed 均值
    arr = np.array(A_full)
    res["mean3"] = {"raw_core": round(arr[:, 0].mean(), 2), "manual_core": round(arr[:, 1].mean(), 2),
                    "gate_lgb_core": round(arr[:, 2].mean(), 2), "gate_lr_core": round(arr[:, 3].mean(), 2),
                    "gate_lgb_core_std": round(arr[:, 2].std(), 2),
                    "gate_lr_core_std": round(arr[:, 3].std(), 2),
                    "manual_core_std": round(arr[:, 1].std(), 2), "raw_core_std": round(arr[:, 0].std(), 2)}

    log("\n" + "=" * 100); log("三 seed 平均 (core)"); log("=" * 100)
    log(f"  p_data 原始      {res['mean3']['raw_core']:>9.2f}  (std {res['mean3']['raw_core_std']:.2f})")
    log(f"  手工规则 T14400  {res['mean3']['manual_core']:>9.2f}  (std {res['mean3']['manual_core_std']:.2f})  改 {res['seeds']['42']['n_manual_changed']} 行")
    log(f"  门控 LGB         {res['mean3']['gate_lgb_core']:>9.2f}  (std {res['mean3']['gate_lgb_core_std']:.2f})")
    log(f"  门控 LR          {res['mean3']['gate_lr_core']:>9.2f}  (std {res['mean3']['gate_lr_core_std']:.2f})")
    log("\n" + "=" * 100); log("分档 (seed42)"); log("=" * 100)
    log(f"{'档':>14} {'p_data':>10} {'手工':>10} {'门控LGB':>10} {'门控LR':>10}")
    for a, b_, c_, d_ in zip(res["seeds"]["42"]["raw"]["bins"], res["seeds"]["42"]["manual"]["bins"],
                             res["seeds"]["42"]["gate_lgb"]["bins"], res["seeds"]["42"]["gate_lr"]["bins"]):
        if a["rmse"] is None: continue
        log(f"{a['bin']:>14} {a['rmse']:>10.2f} {b_['rmse']:>10.2f} {c_['rmse']:>10.2f} {d_['rmse']:>10.2f}")
    log("\n" + "=" * 100); log("分机场 core (seed42)"); log("=" * 100)
    pa = res["seeds"]["42"]["raw"]["per_airport_core"]; pmn = res["seeds"]["42"]["manual"]["per_airport_core"]
    pl = res["seeds"]["42"]["gate_lgb"]["per_airport_core"]; pr = res["seeds"]["42"]["gate_lr"]["per_airport_core"]
    log(f"{'机场':>6} {'p_data':>10} {'手工':>10} {'门控LGB':>10} {'门控LR':>10}")
    for a in sorted(pa):
        log(f"{a:>6} {pa[a]['rmse']:>10.2f} {pmn.get(a,{}).get('rmse',float('nan')):>10.2f} "
            f"{pl.get(a,{}).get('rmse',float('nan')):>10.2f} {pr.get(a,{}).get('rmse',float('nan')):>10.2f}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
    lap(f"→ {OUT}")

if __name__ == "__main__":
    main()
