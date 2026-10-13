"""
v24_calib_rounds.py — 阶段1: 干净口径标定 LGB/XGB/CAT 的固定训练轮数

目的
----
v24 要做全年固定轮数训练 (不用 early stopping), 必须先知道各库需要多少轮。
历史教训: LGB 在全年 ES 上 3 个 seed 全部撞 3000 轮上限 (best_iteration=3000),
说明轮数不足。这里用**干净口径**重标, 不碰真榜月份。

严格 split
----------
  训练: 2025-02..12 (10 个月)
  ES  : 2025-11..12 内的最后 10% (时间靠后) —— 与训练集不重叠
  另附: holdout 2025-01+07 仅作内部参考, **不用于选轮数**

已知工程 bug 一并修掉
--------------------
  * CatBoost: P0 曾因硬编码 1000 轮上限导致 3 个 seed 全部停在整数 1000。
    本脚本**不设 iterations 上限**, 交给 od_type="Iter" + early_stopping_rounds。
  * XGBoost: P0 曾把模型存成 .txt dump 导致不可 load。
    本脚本统一存 .json 并**实际 reload 验证**。

红线: 只读原始数据, 不改 label, 不上传, 不直推 main。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
MODEL_DIR = D0 / "v24_models"
OUT = D0 / "v24_rounds.json"
SEEDS = [42]
CLIP_LO, CLIP_HI = 30.0, 10800.0
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
ES_MONTHS = [11, 12]                 # ES 取自训练集内部的最后两个月
HOLD_MONTHS = [1, 7]                 # 仅内部参考
ES_ROUND = 100
BUF = 1.10                          # 固定轮数 = ceil(best_iter * 1.10)
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


BASE_LGB = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363, "num_leaves": 127,
    "max_depth": 11, "min_data_in_leaf": 5,
    "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
BASE_XGB = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "eta": 0.0147555939449363, "max_depth": 11, "min_child_weight": 5,
    "subsample": 0.9813515130184908, "colsample_bytree": 0.6720708449966734,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "nthread": -1,
}
# P0 修复: 不设 iterations 上限, 由 early_stopping_rounds 决定
BASE_CAT = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "learning_rate": 0.0147555939449363, "depth": 11,
    "l2_leaf_reg": 1.9936339729896335,
    "random_strength": 0.668064577502325,
    "border_count": 127, "thread_count": -1,
    "allow_writing_files": False, "verbose": 0,
    "od_type": "Iter", "od_wait": ES_ROUND,
}
CAPS = {"lgb": 12000, "xgb": 12000, "cat": 12000}   # 给足上限, 靠 ES 收敛


def load():
    import pickle
    blob = pickle.load(open(D0 / "_qinv_feats.pkl", "rb"))
    X = blob["X"]; y = blob["y"]; mo = blob["mo"]; mvt_s = blob["mvt_s"]
    X = X[[c for c in X.columns if not c.startswith("q_")]]
    assert len(X) == 2_085_047, f"特征行数异常 {len(X)}"
    return X, y, mo, mvt_s


def cat_prep(X: pd.DataFrame) -> pd.DataFrame:
    """CAT 的 cat_features 不能是 NaN / category dtype。"""
    Z = X.copy()
    for c in [c for c in Z.columns if str(Z[c].dtype) == "category"]:
        Z[c] = Z[c].astype(str)
    Z = Z.fillna(-999999)
    for c in Z.columns:
        if Z[c].dtype == object:
            Z[c] = Z[c].astype(str)
    return Z


def run_lgb(Xtr, ytr, Xes, yes, seed, cap):
    p = dict(BASE_LGB, seed=seed, bagging_seed=seed,
             feature_fraction_seed=seed, data_random_seed=seed)
    d = lgb.Dataset(Xtr, label=ytr, categorical_feature=LOW_CARD)
    dv = lgb.Dataset(Xes, label=yes, categorical_feature=LOW_CARD, reference=d)
    b = lgb.train(p, d, num_boost_round=cap, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(ES_ROUND, verbose=False),
                             lgb.log_evaluation(0)])
    return b, int(b.best_iteration or cap)


def run_xgb(Xtr, ytr, Xes, yes, seed, cap):
    p = dict(BASE_XGB, seed=seed, random_state=seed)
    d = xgb.DMatrix(Xtr, label=ytr, enable_categorical=True)
    dv = xgb.DMatrix(Xes, label=yes, enable_categorical=True)
    bst = xgb.train(p, d, num_boost_round=cap, evals=[(dv, "es")],
                    early_stopping_rounds=ES_ROUND, verbose_eval=False)
    bi = int(bst.best_iteration) + 1 if hasattr(bst, "best_iteration") else cap
    return bst, bi


def run_cat(Xtr, ytr, Xes, yes, seed, cap):
    m = CatBoostRegressor(**dict(BASE_CAT, random_seed=seed,
                                  iterations=cap))
    m.fit(Xtr, ytr, eval_set=(Xes, yes), use_best_model=True)
    bi = int(m.get_best_iteration() or cap) + 1
    return m, bi


def save_reload(fam, m, bi, path):
    """落盘并实际 reload 验证 (P0 教训: XGB .txt dump 不可 load)。"""
    if fam == "lgb":
        m.save_model(str(path), num_iteration=bi)
    elif fam == "xgb":
        m.save_model(str(path))                 # xgb 3.x 按内容自判 json/ubj
        assert path.suffix == ".json", "XGB 必须存 .json (P0: .txt 是文本 dump)"
    else:
        m.save_model(str(path))
    # reload 自检
    if fam == "lgb":
        r = lgb.Booster(model_file=str(path))
        assert r.num_trees() > 0, "LGB reload 后无树"
        return {"file": path.name, "bytes": path.stat().st_size, "reload": "OK",
                "n_trees": int(r.num_trees())}
    if fam == "xgb":
        r = xgb.Booster(); r.load_model(str(path))
        assert r.num_boosted_rounds() > 0, "XGB reload 后无轮次"
        return {"file": path.name, "bytes": path.stat().st_size, "reload": "OK",
                "n_rounds": int(r.num_boosted_rounds())}
    r = CatBoostRegressor(); r.load_model(str(path))
    assert r.tree_count_ > 0, "CAT reload 后无树"
    return {"file": path.name, "bytes": path.stat().st_size, "reload": "OK",
            "n_trees": int(r.tree_count_)}


LOW_CARD = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]


def main():
    global LOW_CARD
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lap("加载 FEATS71")
    X, y, mo, mvt_s = load()
    LOW_CARD = [c for c in LOW_CARD if c in X.columns]

    tr = np.isin(mo, TRAIN_MONTHS)
    ho = np.isin(mo, HOLD_MONTHS)
    i_tr = np.where(tr)[0]
    # ES = 训练折内按 MVT 时间最后的 10% (与 fit 严格不重叠, 且不含 1/7 月)。
    # 与 seq_ab.py 的 ES 口径完全一致, 保证轮数标定与 A/B 可比。
    o = i_tr[np.argsort(mvt_s[i_tr], kind="stable")]
    cut = int(len(o) * 0.9)
    i_fit, i_es = o[:cut], o[cut:]
    assert not np.isin(mo[i_es], HOLD_MONTHS).any(), "ES 泄漏到 holdout 月"
    assert not np.intersect1d(i_fit, i_es).size, "fit 与 ES 重叠"
    es_m = sorted(set(mo[i_es].tolist()))
    lap(f"fit {len(i_fit):,} | ES {len(i_es):,} (月 {es_m}) | "
        f"内部参考 holdout {int(ho.sum()):,} | ES 占训练折 {100*len(i_es)/len(i_fit):.1f}%")

    # 标签清洗口径与 v15 一致
    tt = y.astype(np.float64)
    keep = np.ones(len(y), bool)
    for i in i_fit:
        if not (tt[i] > 0 and tt[i] <= 10800):
            keep[i] = False
    i_fit = i_fit[keep[i_fit]]
    lap(f"fit 清洗后 {len(i_fit):,}")

    Xtr, ytr = X.iloc[i_fit], tt[i_fit]
    Xes, yes = X.iloc[i_es], tt[i_es]
    Xc_tr, Xc_es = cat_prep(Xtr), cat_prep(Xes)

    res = {"n_fit": int(len(i_fit)), "n_es": int(len(i_es)),
           "es_months": sorted(set(mo[i_es].tolist())),
           "train_months": TRAIN_MONTHS, "holdout_months": HOLD_MONTHS,
           "es_round": ES_ROUND, "buf": BUF, "caps": CAPS, "fams": {}}

    for fam, fn in (("lgb", run_lgb), ("xgb", run_xgb), ("cat", run_cat)):
        for seed in SEEDS:
            lap(f"\n--- {fam} seed{seed} (cap={CAPS[fam]}) ---")
            t0 = time.time()
            # CAT 需要 cat_prep 后的 X (cat_features 不能是 category/NaN)
            a_tr = Xc_tr if fam == "cat" else Xtr
            a_es = Xc_es if fam == "cat" else Xes
            m, bi = fn(a_tr, ytr, a_es, yes, seed, CAPS[fam])
            hit_cap = bi >= CAPS[fam]
            path = MODEL_DIR / f"calib_{fam}_seed{seed}.json"
            ck = save_reload(fam, m, bi, path)
            fixed = int(np.ceil(bi * BUF))
            res["fams"][f"{fam}_seed{seed}"] = {
                "best_iter": bi, "hit_cap": hit_cap, "fixed_rounds": fixed,
                "secs": round(time.time() - t0, 1), "save": ck}
            lap(f"  best_iter={bi}  hit_cap={hit_cap}  → 固定轮数 {fixed}  "
                f"({time.time()-t0:.0f}s)  reload {ck['reload']}")
            del m

    print("\n" + "=" * 84)
    print("阶段1 轮数标定结果")
    print("=" * 84)
    print(f"{'库':<6} {'seed':>5} {'best_iter':>10} {'撞顶':>6} {'固定轮数':>9} {'秒':>7} {'reload':>7}")
    for k, v in res["fams"].items():
        print(f"{k.split('_')[0]:<6} {k.split('_')[1][4:]:>5} {v['best_iter']:>10,} "
              f"{'是' if v['hit_cap'] else '否':>6} {v['fixed_rounds']:>9,} {v['secs']:>7.0f} "
              f"{v['save']['reload']:>7}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    lap(f"→ {OUT}")


if __name__ == "__main__":
    main()
