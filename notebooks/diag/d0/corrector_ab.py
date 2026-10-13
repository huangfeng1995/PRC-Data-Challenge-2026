"""
corrector_ab.py — 按月折 OOF corrector 验证 (D 残差校正为主, C/B 为辅)

红线:
  不改 label / 不 clip 真值 / 不建单机场模型 / TE+LUT 只用训练折(2-12月, 绝不含1,7)
  时间戳先转 datetime64[ns] / 分支隔离(只写 corrector_*.json)
  不据 holdout 回头调参: 所有超参用常规值, 一次跑完

变体:
  A 基线   : 直接 LGB (holdout 预测)
  B 纯stk  : 只喂 OOF预测 -> 真值        (对照, 预期无增益)
  D 残差校正: OOF预测 + 71特征 -> 残差     (重点, LGB + CAT)
  C 直接校正: OOF预测 + 71特征 -> 真值     (辅助对照)
"""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "corrector_ab.json"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))

from src.data_loader import load_month, parse_time_columns
from src.features import build_features as build_basic
from src.features_queue import add_queue_features
import _r7_lib as L
import _r8_drift_lib as D
from v15_clean_retrain import (FEATS71, LOW_CARD_CATS, HIGH_CARD_CATS,
                              build_features, encode_cat_columns, encode_for_catboost,
                              TRAIN_COLS, TIME_COLS)

CLIP_LO, CLIP_HI = 30.0, 10800.0
CAP = 2000                 # 用户指定 2000 轮控成本
ES_ROUND = 80
SEEDS = [42, 123, 2024]
# MAIN split: holdout = 1月 + 7月, 训练 = 2-6,8-12 (10 个月, 绝不含 1/7)
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
assert not (set(TRAIN_MONTHS) & set(HOLD_MONTHS)), "训练月与 holdout 月重叠"
# 按月折 OOF: 10 个月切 5 折, 严格留出 (每月只出现在一个折里)
OOF_FOLDS = [[2, 3], [4, 5], [6, 8], [9, 10], [11, 12]]
_flat = [m for f in OOF_FOLDS for m in f]
assert sorted(_flat) == sorted(TRAIN_MONTHS), f"折未覆盖全部训练月: {sorted(_flat)}"

LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
    "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
CAT_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "learning_rate": 0.0147555939449363, "depth": 11,
    "l2_leaf_reg": 1.9936339729896335, "random_strength": 0.668064577502325,
    "border_count": 127, "thread_count": -1, "allow_writing_files": False, "verbose": 0,
}

_t0 = time.time()
def log(*a):
    print(f"[{time.time()-_t0:7.0f}s]", " ".join(str(x) for x in a), flush=True)


# ============================ 1. 载数据 + 建特征 ============================
def load_all():
    cache = D0 / "_corr_feat_cache.pkl"
    if cache.exists():
        log("复用特征缓存")
        return pd.read_pickle(cache)
    log("载入 12 个月原始数据 ...")
    import glob
    frames = []
    for p in sorted(glob.glob(str(ROOT / "data/raw/training_2025*.parquet"))):
        frames.append(pd.read_parquet(p, columns=TRAIN_COLS))
    raw = pd.concat(frames, ignore_index=True)
    log(f"  raw {len(raw):,} 行")
    # R4: LUT 只用训练月拟合
    raw["month"] = raw["MVT_TIME_UTC_mvt"].dt.month
    tr_raw = raw[raw["month"].isin(TRAIN_MONTHS)]
    lut = L.build_t_unimp_lut(tr_raw)
    log(f"  LUT 用训练月 {len(tr_raw):,} 行拟合")
    df = build_features(raw, ref_dep_for_d=tr_raw, lut=lut, label="all")
    dep = df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.month
    dep = dep[dep["month"].isin(TRAIN_MONTHS + HOLD_MONTHS)].reset_index(drop=True)
    cat_map = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
    dep = encode_cat_columns(dep, cat_map)
    for f in FEATS71:
        if f not in dep.columns:
            dep[f] = np.nan
    dep.to_pickle(cache)
    log(f"  特征表 {dep.shape} 已缓存")
    return dep


# ============================ 2. 按月折 OOF 基础预测 ============================
def gen_oof(dep):
    """LGB 2000轮, 按月折外推 -> 真折外 OOF。返回 oof_pred (仅训练月有值)。"""
    cache = D0 / "_corr_oof.npy"
    if cache.exists():
        log("复用 OOF 缓存")
        return np.load(cache)
    n = len(dep)
    oof = np.full(n, np.nan)
    mo = dep["month"].to_numpy()
    X = dep[FEATS71]
    y = dep["TAXITIME_SEC_mvt"].to_numpy(np.float64)
    for fi, fm in enumerate(OOF_FOLDS):
        tr = ~np.isin(mo, fm)
        va = np.isin(mo, fm)
        dtr = lgb.Dataset(X[tr], label=y[tr], categorical_feature=LOW_CARD_CATS)
        dva = lgb.Dataset(X[va], label=y[va], categorical_feature=LOW_CARD_CATS, reference=dtr)
        b = lgb.train(dict(LGB_PARAMS, seed=SEEDS[0], bagging_seed=SEEDS[0],
                          feature_fraction_seed=SEEDS[0], data_random_seed=SEEDS[0]),
                      dtr, num_boost_round=CAP, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(ES_ROUND, verbose=False), lgb.log_evaluation(0)])
        bi = b.best_iteration or CAP
        oof[va] = np.clip(b.predict(X[va], num_iteration=bi).astype(np.float64), CLIP_LO, CLIP_HI)
        log(f"  OOF fold{fi+1} 月{fm}: n_va={int(va.sum()):,} best_iter={bi}")
    np.save(cache, oof)
    return oof


# ============================ 3. 变体训练 ============================
def fit_predict_lgb(Xtr, ytr, Xva, yva, Xho, seed, cap=CAP):
    # 只有当帧里确实存在 category 列时才声明 categorical_feature
    cats = [c for c in LOW_CARD_CATS if c in Xtr.columns]
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats or "auto")
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats or "auto", reference=dtr)
    b = lgb.train(dict(LGB_PARAMS, seed=seed, bagging_seed=seed,
                      feature_fraction_seed=seed, data_random_seed=seed),
                  dtr, num_boost_round=cap, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(ES_ROUND, verbose=False), lgb.log_evaluation(0)])
    bi = b.best_iteration or cap
    p = np.clip(b.predict(Xho, num_iteration=bi).astype(np.float64), CLIP_LO, CLIP_HI)
    return p, bi


def fit_predict_cat(Xtr, ytr, Xva, yva, Xho, seed, cap=CAP):
    m = CatBoostRegressor(**dict(CAT_PARAMS, random_seed=seed))
    m.fit(Xtr, ytr, cat_features=LOW_CARD_CATS, eval_set=(Xva, yva),
          early_stopping_rounds=ES_ROUND, verbose=0)
    bi = m.get_best_iteration() + 1
    p = np.clip(m.predict(Xho).astype(np.float64), CLIP_LO, CLIP_HI)
    return p, bi


# ============================ 4. 评估 ============================
def evaluate(pred, y, ap, mo, base_full, base_core, base_ap, name):
    core = (y > 0) & (y <= 10800)
    R = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.sum() else float("nan")
    full, cr = R(pred, np.ones(len(y), bool)), R(pred, core)
    pa = {}
    for a in pd.unique(ap):
        m = ap == a
        pa[str(a)] = {"n": int(m.sum()), "full": round(R(pred, m), 2),
                      "base_full": round(base_ap[a], 2), "delta": round(R(pred, m) - base_ap[a], 2)}
    ds = [v["delta"] for v in pa.values()]
    imp = sum(1 for d in ds if d < 0); wor = sum(1 for d in ds if d > 0)
    return {
        "name": name, "full": round(full, 2), "core": round(cr, 2),
        "d_full": round(full - base_full, 2), "d_core": round(cr - base_core, 2),
        "jan": round(R(pred, mo == 1), 2), "jul": round(R(pred, mo == 7), 2),
        "d_jan": round(R(pred, mo == 1) - base_ap["_jan"], 2),
        "d_jul": round(R(pred, mo == 7) - base_ap["_jul"], 2),
        "per_airport": pa, "n_ap_improve": imp, "n_ap_worse": wor,
        "max_worse": round(max([d for d in ds if d > 0], default=0.0), 2),
    }


# ============================ main ============================
def main():
    dep = load_all()
    mo = dep["month"].to_numpy()
    y = dep["TAXITIME_SEC_mvt"].to_numpy(np.float64)
    ap = dep["ADEP_mvt"].to_numpy()
    tr = np.isin(mo, TRAIN_MONTHS)
    ho = np.isin(mo, HOLD_MONTHS)
    log(f"训练月 {int(tr.sum()):,} 行 | holdout {int(ho.sum()):,} 行")

    log("生成按月折 OOF 基础预测 (LGB, 5 折, 2000 轮) ...")
    oof = gen_oof(dep)
    assert not np.isnan(oof[tr]).any(), "OOF 覆盖不完整"
    assert np.isnan(oof[ho]).all(), "holdout 不应有 OOF"
    core_tr = (y[tr] > 0) & (y[tr] <= 10800)
    log(f"  OOF 训练月 RMSE: full={np.sqrt(np.mean((oof[tr]-y[tr])**2)):.2f} "
        f"core={np.sqrt(np.mean((oof[tr][core_tr]-y[tr][core_tr])**2)):.2f}")

    # 基线 A: 在全训练月上训, 预测 holdout (3 seed 平均 = P0 的做法)
    log("训练基线 A (全训练月 -> holdout) ...")
    # 注意: fit_idx/es_idx 是【全表】位置, 必须在未子集化的帧上 .iloc
    Xall = dep[FEATS71]
    ts = dep["MVT_TIME_UTC_mvt"].astype("int64").to_numpy()[tr]
    order = np.argsort(ts, kind="stable"); cut = int(len(order) * 0.9)
    es_pos = order[cut:]
    tr_idx = np.where(tr)[0]
    fit_idx, es_idx = tr_idx[order[:cut]], tr_idx[es_pos]
    es_months = sorted(set(mo[es_idx].tolist()))
    assert 1 not in es_months and 7 not in es_months, f"R4 违规 {es_months}"
    log(f"  基线 ES: n={len(es_idx):,} 月份={es_months} ✓ 不含 1/7")
    Xho = dep[FEATS71][ho]
    bcache = D0 / "_corr_basepred.npy"
    if bcache.exists():
        base = np.load(bcache)
        log(f"  复用基线预测缓存 {base.shape}")
    else:
        base_preds = []
        for sd in SEEDS:
            p, bi = fit_predict_lgb(Xall.iloc[fit_idx], y[fit_idx],
                                    Xall.iloc[es_idx], y[es_idx], Xho, sd)
            base_preds.append(p)
            log(f"  A seed{sd}: best_iter={bi}")
        base = np.mean(base_preds, 0)
        np.save(bcache, base)

    # 基线指标
    yho, apho, moho = y[ho], ap[ho], mo[ho]
    # ADEP_mvt 是整数编码; 用 cat_map 还原机场代码, 否则分机场报告全是数字
    _cm = json.loads((ROOT / "models/feature_lists/_rebuild_12m_cat.json").read_text())
    _rev = {int(v): k for k, v in _cm.get("ADEP_mvt", {}).items()}
    apho = np.array([_rev.get(int(a), f"UNK{int(a)}") for a in apho], dtype=object)
    core = (yho > 0) & (yho <= 10800)
    R = lambda p, m: float(np.sqrt(np.mean((p[m] - yho[m]) ** 2)))
    base_full = R(base, np.ones(len(yho), bool)); base_core = R(base, core)
    base_ap = {a: R(base, apho == a) for a in pd.unique(apho)}
    base_ap["_jan"] = R(base, moho == 1); base_ap["_jul"] = R(base, moho == 7)
    log(f"基线 A: full={base_full:.2f} core={base_core:.2f} "
        f"Jan={base_ap['_jan']:.2f} Jul={base_ap['_jul']:.2f}")
    log("  (对照 P0 ENS: full 567.61 / core 304.94; 本次 cap=2000 且单库, 预期略高)")

    res = {"base_A": {"full": round(base_full, 2), "core": round(base_core, 2),
                      "jan": round(base_ap['_jan'], 2), "jul": round(base_ap['_jul'], 2),
                      "n_seeds": len(SEEDS), "cap": CAP,
                      "per_airport": {str(k): round(v, 2) for k, v in base_ap.items()}},
           "oof_quality": {"full": round(float(np.sqrt(np.mean((oof[tr]-y[tr])**2))), 2),
                           "core": round(float(np.sqrt(np.mean((oof[tr][core_tr]-y[tr][core_tr])**2))), 2),
                           "folds": OOF_FOLDS}}
    variants = {}

    # ---- B: 纯 stacking (只喂 OOF 预测 -> 真值) ----
    log("变体 B: 纯 stacking (只喂 OOF 预测 -> 真值) ...")
    Xb_fit = pd.DataFrame({"oof": oof[fit_idx]})
    Xb_es = pd.DataFrame({"oof": oof[es_idx]})
    Xb_ho = pd.DataFrame({"oof": base})
    bs = []
    for sd in SEEDS:
        p, bi = fit_predict_lgb(Xb_fit, y[fit_idx], Xb_es, y[es_idx], Xb_ho, sd)
        bs.append(p); log(f"  B seed{sd}: best_iter={bi}")
    variants["B_stacking_oof_only"] = evaluate(np.mean(bs, 0), yho, apho, moho,
                                               base_full, base_core, base_ap, "B")
    variants["B_stacking_oof_only"]["per_seed_full"] = [round(R(p, np.ones(len(yho), bool)), 2) for p in bs]

    # ---- D / C: OOF + 全特征 ----
    log("构建 D/C 的特征矩阵 (OOF预测 + 71特征) ...")
    def mk(with_feats):
        if not with_feats:
            return (pd.DataFrame({"oof": oof[fit_idx]}), pd.DataFrame({"oof": oof[es_idx]}),
                    pd.DataFrame({"oof": base}))
        A = dep[FEATS71].reset_index(drop=True)
        f = A.iloc[fit_idx].copy(); f.insert(0, "oof_pred", oof[fit_idx])
        e = A.iloc[es_idx].copy(); e.insert(0, "oof_pred", oof[es_idx])
        h = A.iloc[np.where(ho)[0]].copy(); h.insert(0, "oof_pred", base)
        return f, e, h
    Xf, Xe, Xh = mk(True)
    cat_cols = LOW_CARD_CATS

    # ---- D: 残差校正 (重点) ----
    log("变体 D: 残差校正 (OOF + 71特征 -> 残差) ...")
    resid = y - oof
    for sd in SEEDS:
        p, bi = fit_predict_lgb(Xf, resid[fit_idx], Xe, resid[es_idx], Xh, sd)
        final = np.clip(base + p, CLIP_LO, CLIP_HI)
        variants.setdefault("D_resid_lgb", []).append(
            {"seed": sd, "best_iter": bi, **evaluate(final, yho, apho, moho,
                                            base_full, base_core, base_ap, f"D_lgb_s{sd}")})
        log(f"  D-LGB seed{sd}: best_iter={bi} full={variants['D_resid_lgb'][-1]['full']:.2f} "
            f"Δ={variants['D_resid_lgb'][-1]['d_full']:+.2f}")
    # D + CAT
    Xf_c, Xe_c, Xh_c = encode_for_catboost(Xf), encode_for_catboost(Xe), encode_for_catboost(Xh)
    for sd in (SEEDS if os.environ.get("SKIP_CAT", "0") != "1" else []):
        p, bi = fit_predict_cat(Xf_c, resid[fit_idx], Xe_c, resid[es_idx], Xh_c, sd)
        final = np.clip(base + p, CLIP_LO, CLIP_HI)
        variants.setdefault("D_resid_cat", []).append(
            {"seed": sd, "best_iter": bi, **evaluate(final, yho, apho, moho,
                                            base_full, base_core, base_ap, f"D_cat_s{sd}")})
        log(f"  D-CAT seed{sd}: best_iter={bi} full={variants['D_resid_cat'][-1]['full']:.2f} "
            f"Δ={variants['D_resid_cat'][-1]['d_full']:+.2f}")

    # ---- C: 直接校正 (辅助) ----
    log("变体 C: 直接校正 (OOF + 71特征 -> 真值) ...")
    for sd in SEEDS[:2]:   # 辅助, 2 seed 控成本
        p, bi = fit_predict_lgb(Xf, y[fit_idx], Xe, y[es_idx], Xh, sd)
        variants.setdefault("C_direct_lgb", []).append(
            {"seed": sd, "best_iter": bi, **evaluate(p, yho, apho, moho,
                                            base_full, base_core, base_ap, f"C_lgb_s{sd}")})
        log(f"  C-LGB seed{sd}: best_iter={bi} full={variants['C_direct_lgb'][-1]['full']:.2f} "
            f"Δ={variants['C_direct_lgb'][-1]['d_full']:+.2f}")

    res["variants"] = variants
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"→ {OUT}")
    return res, base_full, base_core, base_ap


if __name__ == "__main__":
    main()
