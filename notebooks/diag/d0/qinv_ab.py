"""
qinv_ab.py — step2: 排队反演量作为特征的 A/B 增益验证

协议 (与基线一致):
  train = 2025 月 2-6,8-12 ; holdout = 1+7
  ES = 训练月内按 MVT 时间最后 10% (即 11-12 月), 断言不含 1/7
  cap 2000, seed 42/123/2024, FEATS71 + 12 新增反演量
红线: 不改 label、不 clip 真值、反演只用 MVT 之前信息
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
RAW = ROOT / "data/raw"
D0 = ROOT / "notebooks" / "diag" / "d0"
SRC = ROOT / "src"
OUT = D0 / "qinv_ab.json"
FCACHE = D0 / "_qinv_feats.pkl"
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
BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, float("inf"))]
BLAB = ["[0,600)", "[600,1800)", "[1800,3600)", "[3600,10800]", ">10800"]
NEW_FEATS = ["q_k_burst", "q_k_x_s", "q_svc_gap", "q_svc_p75", "q_svc_ratio",
             "q_idle_since_dep", "q_burst_len_min", "q_dep_cnt_60m", "q_dep_cnt_120m",
             "q_arr_cnt_30m", "q_arr_cnt_60m", "q_rwy_share_dep"]
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]

def log(*a): print(" ".join(str(x) for x in a), flush=True)
_t0 = time.time()
def lap(m=""): log(f"[{time.time()-_t0:7.1f}s] {m}")

def mf(m: int) -> Path:
    ny, n2 = (2025, m + 1) if m < 12 else (2026, 1)
    return RAW / f"training_2025-{m:02d}-01_{ny}-{n2:02d}-01.parquet"

# ================= 反演特征 (只用 MVT 之前信息) =================
def qinv_features(dep: pd.DataFrame) -> pd.DataFrame:
    """逐跑道按 MVT 排序做序列反演。全部统计量只看 [MVT-w, MVT) 与历史间隔。"""
    n = len(dep)
    o = {c: np.full(n, np.nan, dtype=np.float32) for c in NEW_FEATS}
    mvt = dep["MVT_TIME_UTC_mvt_s"].to_numpy(np.float64)
    rwy = dep["RUNWAY_mvt"].astype(str).to_numpy()
    ap = dep["ADEP_mvt"].astype(str).to_numpy()
    fin = np.isfinite(mvt)
    # 到港计划时刻 (只做计划, 不用实际落地)
    arr_s = dep["_arr_sched_s"].to_numpy(np.float64) if "_arr_sched_s" in dep.columns \
        else np.full(n, np.nan)
    mvt_safe = np.where(fin, mvt, np.inf)

    for rw in np.unique(rwy):
        m = (rwy == rw) & fin
        idx = np.where(m)[0]
        if len(idx) < 20:
            continue
        vo = mvt[idx]
        order = np.argsort(vo, kind="stable")
        idx = idx[order]; vo = vo[order]
        gaps = np.diff(vo)
        tight = gaps[gaps < 300.0]
        s_med = float(np.median(tight)) if len(tight) else 300.0
        thr = max(120.0, 3.0 * s_med)
        # 逐架局部服务间隔 + 反演 k + 连续段时长
        loc_s = np.empty(len(idx)); loc_s[0] = s_med
        k = np.zeros(len(idx), dtype=np.float64)
        burst_id = np.zeros(len(idx), dtype=np.int64)
        run_len = np.zeros(len(idx), dtype=np.float64)
        bs = 0; bl = 0.0
        for i in range(1, len(idx)):
            lo = max(0, i - 20)
            loc = gaps[lo:i]
            loc_t = loc[loc < thr]
            if len(loc_t) >= 3:
                loc_s[i] = float(np.median(loc_t))
            if gaps[i - 1] < thr:
                k[i] = k[i - 1] + 1
                bl += gaps[i - 1]
            else:
                k[i] = 0.0; bl = 0.0
            burst_id[i] = burst_id[i - 1] + (1 if gaps[i - 1] < thr else 0)
            run_len[i] = bl
        o["q_k_burst"][idx] = k
        o["q_svc_gap"][idx] = s_med
        o["q_svc_p75"][idx] = float(np.percentile(tight, 75)) if len(tight) else s_med
        o["q_k_x_s"][idx] = k * loc_s
        o["q_burst_len_min"][idx] = run_len / 60.0
        o["q_idle_since_dep"][idx] = np.concatenate([[np.nan], gaps])
        # 窗口计数 (MVT 之前)
        for w, nm in [(3600, "q_dep_cnt_60m"), (7200, "q_dep_cnt_120m")]:
            o[nm][idx] = (np.searchsorted(vo, vo - w, side="left")).astype(np.float32)
        # 机场级 30min 起飞数, 用于算跑道占比
        o["q_rwy_share_dep"][idx] = 0.0
    # 机场级 30min 起飞数
    ap_cnt = np.zeros(n, dtype=np.float32)
    for a in np.unique(ap):
        m = (ap == a) & fin
        idx = np.where(m)[0]
        if len(idx) < 20:
            continue
        vo = np.sort(mvt[idx])
        ap_cnt[idx] = (np.searchsorted(vo, vo, side="right")
                       - np.searchsorted(vo, vo - 1800, side="left")).astype(np.float32)
    o["q_rwy_share_dep"][:] = o["q_dep_cnt_60m"] / np.maximum(ap_cnt, 1.0)
    # q_svc_ratio: 反演排位占该跑道近 60min 起飞流量的比例 (拥堵压力占比)
    o["q_svc_ratio"][:] = o["q_k_burst"] / np.maximum(o["q_dep_cnt_60m"], 1.0)
    # 到港计数 (计划时刻, MVT 之前)
    for w, nm in [(1800, "q_arr_cnt_30m"), (3600, "q_arr_cnt_60m")]:
        c = np.zeros(n, dtype=np.float32)
        af = np.isfinite(arr_s)
        for rw in np.unique(rwy):
            src = np.where((rwy == rw) & af)[0]
            if len(src) < 20:
                continue
            tgt = np.where((rwy == rw) & fin)[0]
            if len(tgt) == 0:
                continue
            ss = np.sort(arr_s[src])
            c[tgt] = (np.searchsorted(ss, mvt[tgt], side="left")
                      - np.searchsorted(ss, mvt[tgt] - w, side="left")).astype(np.float32)
        o[nm][:] = c
    return pd.DataFrame(o)

# ================= 主流程 =================
def main():
    if FCACHE.exists():
        with open(FCACHE, "rb") as f:
            blob = pickle.load(f)
        X = blob["X"]; y = blob["y"]; mo = blob["mo"]; qf = blob["qf"]
        ap = blob["ap"]
        lap("特征缓存命中")
    else:
        sys.path.insert(0, str(SRC)); sys.path.insert(0, str(D0))
        import _r7_lib as L
        from oof_calibrate import FEATS71, build_features, encode_X
        from v15_clean_retrain import TRAIN_COLS, TIME_COLS
        from src.data_loader import parse_time_columns
        months = sorted(TRAIN_MONTHS + HOLD_MONTHS)
        lap(f"读取 {months} 月原始数据")
        raw = pd.concat([pd.read_parquet(mf(m), columns=TRAIN_COLS) for m in months],
                        ignore_index=True)
        raw = parse_time_columns(raw.copy(), cols=TIME_COLS)
        dep_all = raw[raw["PHASE_mvt"] == "DEP"].reset_index(drop=True)
        for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"]:
            v = pd.to_datetime(dep_all[c], utc=True, errors="coerce")
            dep_all[c] = v.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
            dep_all[c + "_s"] = dep_all[c].astype("int64") // NS
        dep_all["month"] = pd.to_datetime(dep_all["MVT_TIME_UTC_mvt"]).dt.month
        # 到港计划时刻表 (仅计划, 不用实际落地)
        arr = raw[raw["PHASE_mvt"] == "ARR"].copy()
        av = pd.to_datetime(arr["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
        arr["_arr_sched_s"] = av.dt.tz_convert("UTC").dt.tz_localize(None).astype("int64") // NS
        arr["_rwy"] = arr["ADES_mvt"].astype(str)
        lap(f"DEP {len(dep_all):,} / ARR {len(arr):,}")
        # 训练月子集 (LUT 与 drift 参考只用训练月, 防泄漏)
        tr_mask = dep_all["month"].isin(TRAIN_MONTHS).to_numpy()
        dep_tr = dep_all[tr_mask].reset_index(drop=True)
        lap(f"拟合 t_unimp LUT (仅训练月 {len(dep_tr):,})")
        lut = L.build_t_unimp_lut(dep_tr)
        lap("build_features (FEATS71, ref_dep_for_d=训练月)")
        X = build_features(dep_all.copy(), dep_tr, lut).reset_index(drop=True)
        t_unimp = X["t_unimp_r7"].to_numpy(np.float32)
        dep = X[["MVT_TIME_UTC_mvt_s", "RUNWAY_mvt", "ADEP_mvt", "month"]].copy()
        y = (dep_all["MVT_TIME_UTC_mvt_s"].to_numpy(np.float64)
             - dep_all["BLOCK_TIME_UTC_mvt_s"].to_numpy(np.float64))
        mo = dep_all["month"].to_numpy(np.int64)
        ap = dep_all["ADEP_mvt"].astype(str).to_numpy()
        # 到港计划时刻挂到 DEP 行 (按跑道最近计划到港, 仅用于窗口计数)
        arr_by_rwy = {r: np.sort(g["_arr_sched_s"].to_numpy(np.float64))
                      for r, g in arr.groupby("_rwy")}
        lap("计算 12 个新增反演特征")
        dep["_arr_sched_s"] = np.nan
        rwy_s = dep["RUNWAY_mvt"].astype(str)
        for r, ss in arr_by_rwy.items():
            m = (rwy_s == r).to_numpy()
            if m.sum() == 0 or len(ss) == 0:
                continue
            pos = np.clip(np.searchsorted(ss, dep.loc[m, "MVT_TIME_UTC_mvt_s"].to_numpy()),
                          0, len(ss) - 1)
            dep.loc[m, "_arr_sched_s"] = ss[pos]
        qf = qinv_features(dep)
        lap("对齐特征矩阵 + 整数编码")
        for c in qf.columns:
            X[c] = qf[c].to_numpy(np.float32)
        cols = [c for c in list(FEATS71) + NEW_FEATS if c in X.columns]
        # 注意: 不能用 np.asarray() —— 会把 category dtype 降成 object
        X = encode_X(X, cols)
        assert all(t.kind in "biuf" or str(t) == "category"
                   for t in X.dtypes), f"非法 dtype: {X.dtypes[~X.dtypes.isin(['int32','int64','float32','float64','category','bool','int8'])].to_dict()}"
        X = X.astype({c: "float32" for c in NEW_FEATS if c in X.columns})
        with open(FCACHE, "wb") as f:
            pickle.dump({"X": X, "y": y, "mo": mo, "qf": qf, "ap": ap,
                         "mvt_s": dep["MVT_TIME_UTC_mvt_s"].to_numpy(np.float64)}, f)
    lap(f"X={X.shape}  新增列非空率 " +
        ", ".join(f"{c}={X[c].notna().mean()*100:.1f}%" for c in NEW_FEATS))

    # ---- split ----
    tr = np.isin(mo, TRAIN_MONTHS)
    ho = np.isin(mo, HOLD_MONTHS)
    assert not (tr & ho).any()
    lap(f"train {tr.sum():,}  holdout {ho.sum():,}")
    # ES = 训练月内按 MVT 时间最后 10%
    with open(FCACHE, "rb") as f:
        mvt_all = pickle.load(f)["mvt_s"]
    idx_tr = np.where(tr)[0]
    order = idx_tr[np.argsort(mvt_all[idx_tr], kind="stable")]
    cut = int(len(order) * 0.9)
    fit_idx, es_idx = order[:cut], order[cut:]
    es_months = sorted(set(mo[es_idx].tolist()))
    assert 1 not in es_months and 7 not in es_months, f"R4 违规 ES 含 holdout 月 {es_months}"
    ho_idx = np.where(ho)[0]
    lap(f"拟合 {len(fit_idx):,} / ES {len(es_idx):,} 月份={es_months} ✓ 不含 1/7")

    def run(cols, tag):
        preds, infos = [], []
        for sd in SEEDS:
            t1 = time.time()
            p = dict(LGB_PARAMS); p["seed"] = sd
            p["bagging_seed"] = sd + 1; p["feature_fraction_seed"] = sd + 2
            dtr = lgb.Dataset(X.iloc[fit_idx][cols], label=y[fit_idx],
                              categorical_feature=LOW_CARD_CATS, free_raw_data=False)
            dva = lgb.Dataset(X.iloc[es_idx][cols], label=y[es_idx],
                              categorical_feature=LOW_CARD_CATS, reference=dtr)
            b = lgb.train(p, dtr, num_boost_round=CAP, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80, verbose=False),
                                     lgb.log_evaluation(0)])
            pr = np.clip(b.predict(X.iloc[ho_idx][cols]).astype(np.float64), 30.0, 10800.0)
            preds.append(pr)
            bi = b.best_iteration if b.best_iteration > 0 else CAP
            infos.append({"seed": sd, "best_iter": int(bi), "sec": round(time.time() - t1, 1)})
            lap(f"  {tag} seed{sd}: best_iter={bi} ({time.time()-t1:.0f}s)")
        P = np.mean(preds, axis=0)
        return P, infos

    def report(P, infos, tag):
        yt = y[ho]
        core = (yt > 0) & (yt <= 10800)
        r = lambda m_: float(np.sqrt(np.mean((P[m_] - yt[m_]) ** 2)))
        full, cr = r(np.ones(len(yt), bool)), r(core)
        out = {"tag": tag, "per_seed": infos, "full": round(full, 2), "core": round(cr, 2),
               "jan": round(r(mo[ho] == 1), 2), "jul": round(r(mo[ho] == 7), 2),
               "bins": [], "per_airport_core": {}}
        for (lo, hi), lab in zip(BINS, BLAB):
            m_ = (yt >= lo) & (yt < hi) if hi != float("inf") else (yt >= lo)
            out["bins"].append({"bin": lab, "n": int(m_.sum()),
                                "rmse": round(r(m_), 2) if m_.sum() >= 5 else None})
        ah = ap[ho]
        for a in pd.Series(ah).value_counts().index:
            mk = (ah == a) & core
            if mk.sum() >= 500:
                out["per_airport_core"][str(a)] = {"n": int(mk.sum()),
                                                    "rmse": round(r(mk), 2)}
        log(f"  {tag}: full={full:.2f} core={cr:.2f} Jan={out['jan']:.2f} Jul={out['jul']:.2f}")
        return out, P, yt

    res = {}
    base_cols = [c for c in X.columns if c not in NEW_FEATS]
    lap(f"=== A 基线 {len(base_cols)} 列 ===")
    Pb, ib = run(base_cols, "base")
    res["A_base"], _, _ = report(Pb, ib, "A_base")
    lap(f"=== B 基线+12 反演量 {X.shape[1]} 列 ===")
    Pq, iq = run(list(X.columns), "plus")
    res["B_plus"], _, _ = report(Pq, iq, "B_plus")

    # ---- 对比 ----
    d_full = res["B_plus"]["full"] - res["A_base"]["full"]
    d_core = res["B_plus"]["core"] - res["A_base"]["core"]
    pa, pb = res["A_base"]["per_airport_core"], res["B_plus"]["per_airport_core"]
    common = sorted(set(pa) & set(pb))
    better = sum(1 for a in common if pb[a]["rmse"] < pa[a]["rmse"])
    diffs = {a: round(pb[a]["rmse"] - pa[a]["rmse"], 2) for a in common}
    log("")
    log("=" * 96)
    log("A/B 对比")
    log("=" * 96)
    log(f"{'档':>14} {'A基线':>10} {'B+反演':>10} {'Δ':>10}")
    for ba, bb in zip(res["A_base"]["bins"], res["B_plus"]["bins"]):
        if ba["rmse"] is None:
            continue
        log(f"{ba['bin']:>14} {ba['rmse']:>10.2f} {bb['rmse']:>10.2f} {bb['rmse']-ba['rmse']:>+10.2f}")
    log(f"\nfull  {res['A_base']['full']:>8.2f} → {res['B_plus']['full']:>8.2f}  Δ={d_full:+.2f}")
    log(f"core  {res['A_base']['core']:>8.2f} → {res['B_plus']['core']:>8.2f}  Δ={d_core:+.2f}")
    log(f"Jan   {res['A_base']['jan']:>8.2f} → {res['B_plus']['jan']:>8.2f}")
    log(f"Jul   {res['A_base']['jul']:>8.2f} → {res['B_plus']['jul']:>8.2f}")
    log(f"\n分机场 core (B 更优 {better}/{len(common)}):")
    for a in sorted(diffs, key=lambda x: diffs[x]):
        log(f"  {a:>5} {pa[a]['rmse']:>9.2f} → {pb[a]['rmse']:>9.2f}  Δ={diffs[a]:>+9.2f}")
    seed_core = [i["best_iter"] for i in iq]
    log(f"\nB 三 seed best_iter: {seed_core}   A 三 seed best_iter: {[i['best_iter'] for i in ib]}")

    verdict = ("有稳定增益" if (d_full < 0 and d_core < 0 and better >= 6) else "无增益")
    log(f"\n判定: {verdict}  (Δfull={d_full:+.2f}, Δcore={d_core:+.2f}, 机场同号 {better}/{len(common)})")
    res["delta"] = {"full": round(d_full, 2), "core": round(d_core, 2),
                    "ap_better": better, "ap_total": len(common),
                    "ap_diff": diffs, "verdict": verdict,
                    "seed_std_note": "三 seed 同模型, std≈2.9s (历史)"}
    res["new_feats"] = NEW_FEATS
    res["n_cols"] = {"base": len(base_cols), "plus": int(X.shape[1])}
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
    lap(f"→ {OUT}")

if __name__ == "__main__":
    main()
