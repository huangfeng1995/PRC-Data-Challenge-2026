"""
v24_3lib_probe.py — 串行标定 XGB / Cat, 并验证三库简单平均是否有增益

严格串行: 一次只跑一个训练 (--fam xgb 或 --fam cat), 避免再被系统杀。
只用 LGB 单库 (已完成, best_iter=10423) 作为对照基准。

口径 (与已完成的 LGB 阶段1 完全一致)
--------------------------------------
  fit    : 2025-02..12 DEP, 清洗 (剔 TAXITIME<=0 或 >10800)
  ES     : fit 按 MVT 排序最后 10% (月份 ⊆ {11,12}, 与 fit 零重叠)
  holdout: 2025 Jan + Jul (344,419 行)

评估: **只看 full RMSE** (core 口径已两次误导: P0 泄漏、P50 高估)

阶段划分 (每步单独跑, 结果落盘复用)
  --fam xgb : 训练 XGB, 落盘模型 + holdout 预测 + best_iter
  --fam cat : 训练 Cat, 同上
  --fam all : 读三库预测, 计算 full RMSE / 分机场 / 分月 Δ 与判定

红线: 只读原始数据, 不改 label, 不 clip 真值, 不上传, 严格串行。
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
MDIR = D0 / "v24_models"
PRED = D0 / "v24_preds"
OUT = D0 / "v24_3lib_probe.json"
CAP = 14000
SEED = 42
CLIP_LO, CLIP_HI = 30.0, 10800.0
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
BASE_XGB = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "eta": 0.0147555939449363, "max_depth": 11, "min_child_weight": 5,
    "subsample": 0.9813515130184908, "colsample_bytree": 0.6720708449966734,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "nthread": -1,
}
BASE_CAT = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "learning_rate": 0.0147555939449363, "depth": 11,
    "l2_leaf_reg": 1.9936339729896335,
    "random_strength": 0.668064577502325,
    "border_count": 127, "thread_count": -1,
    "allow_writing_files": False, "verbose": 0,
    "od_type": "Iter", "od_wait": 100,     # P0 修复: 无 1000 轮硬上限
}
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def jdump(d, p):
    old = {}
    if p.exists():
        try:
            old = json.load(open(p))
        except Exception:
            old = {}
    old.update(d)
    json.dump(old, open(p, "w"), ensure_ascii=False, indent=2)
    return old


def load_split():
    """重建与 LGB 阶段1 完全一致的 fit / ES / holdout 划分。"""
    import pickle
    blob = pickle.load(open(D0 / "_qinv_feats.pkl", "rb"))
    X = blob["X"]; y = blob["y"].astype(np.float64)
    mo = blob["mo"].astype(np.int64); mvt_s = blob["mvt_s"]
    X = X[[c for c in X.columns if not c.startswith("q_")]]
    assert len(X) == 2_085_047

    tr = np.isin(mo, TRAIN_MONTHS)
    i_tr = np.where(tr)[0]
    o = i_tr[np.argsort(mvt_s[i_tr], kind="stable")]
    cut = int(len(o) * 0.9)
    i_fit, i_es = o[:cut], o[cut:]
    assert not np.isin(mo[i_es], HOLD_MONTHS).any(), "ES 泄漏到 holdout"
    assert not np.intersect1d(i_fit, i_es).size
    keep = (y[i_fit] > 0) & (y[i_fit] <= 10800)
    n_raw = len(i_fit); i_fit = i_fit[keep]
    i_ho = np.where(np.isin(mo, HOLD_MONTHS))[0]
    es_m = sorted(set(mo[i_es].tolist()))
    lap(f"fit {len(i_fit):,} (清洗前 {n_raw:,}) | ES {len(i_es):,} 月 {es_m} | "
        f"holdout {len(i_ho):,}")
    # holdout 机场标签 (分机场评估用, 只读) —— 跨 12 个月文件按行序拼接
    if not (D0 / "_v24_holdout_ap.parquet").exists():
        parts_ap = []
        for mm in range(1, 13):
            nxt = "2026-01-01" if mm == 12 else f"2025-{mm + 1:02d}-01"
            fp = ROOT / f"data/raw/training_2025-{mm:02d}-01_{nxt}.parquet"
            dd = pd.read_parquet(fp, columns=["PHASE_mvt", "ADEP_mvt"])
            parts_ap.append(dd[dd["PHASE_mvt"] == "DEP"]
                            .reset_index(drop=True)["ADEP_mvt"].astype(str))
        ap_all = pd.concat(parts_ap, ignore_index=True)
        assert len(ap_all) == len(X), f"机场表行数 {len(ap_all)} != {len(X)}"
        pd.DataFrame({"ap": ap_all.to_numpy()[i_ho]}).to_parquet(
            D0 / "_v24_holdout_ap.parquet", index=False)
        lap(f"  已保存 holdout 机场标签 n={len(i_ho):,} (机场 {ap_all.iloc[i_ho].nunique()} 个)")
    return X, y, mo, i_fit, i_es, i_ho


def cat_prep(X):
    """CAT 的 cat_features 不能是 category dtype / NaN。

    注意 (实测踩坑): `category.astype(str)` 会把 NaN 变成字符串 "nan",
    之后 fillna 对 object 列无效, CatBoost 读该列时报
        Bad value for num_feature[...]= "M": Cannot convert 'M' to float
    正确做法: 类别列一律转 **整数编码** (cat.codes, NaN → -1), 与 LGB/XGB 的
    整数编码口径一致; 数值列的 NaN 填 -999999。
    """
    Z = X.copy()
    for c in Z.columns:
        if str(Z[c].dtype) == "category":
            # codes 是 int8/int16, NaN 处为 -1 —— 已是合法整数, 无需 fillna
            Z[c] = Z[c].cat.codes.astype(np.int32)
        elif Z[c].dtype == object:
            Z[c] = pd.factorize(Z[c], sort=True)[0].astype(np.int32)
    # 数值列的 NaN 用哨兵整数 (CatBoost 数值列不接受 NaN)
    num = [c for c in Z.columns if str(Z[c].dtype) != "category"
           and str(Z[c].dtype) != "object"]
    Z[num] = Z[num].fillna(-999999)
    # 最终自检: 不允许残留 object / category / NaN
    for c in Z.columns:
        assert str(Z[c].dtype) in ("int8", "int16", "int32", "int64",
                                   "float32", "float64"), \
            f"{c} dtype={Z[c].dtype} 未转成数值"
        assert not Z[c].isna().any(), f"{c} 残留 NaN"
    return Z


def do_xgb(X, y, i_fit, i_es, i_ho):
    import xgboost as xgb
    p = dict(BASE_XGB, seed=SEED, random_state=SEED)
    Xf, Xe, Xh = X.iloc[i_fit], X.iloc[i_es], X.iloc[i_ho]
    d = xgb.DMatrix(Xf, label=y[i_fit], enable_categorical=True)
    dv = xgb.DMatrix(Xe, label=y[i_es], enable_categorical=True)
    lap(f"  DMatrix 构建完成, 开始训练 (cap={CAP})")
    bst = xgb.train(p, d, num_boost_round=CAP, evals=[(dv, "es")],
                    early_stopping_rounds=100, verbose_eval=False)
    bi = int(bst.best_iteration) + 1 if hasattr(bst, "best_iteration") else CAP
    mp = MDIR / "calib_xgb_seed42.json"
    bst.save_model(str(mp))                      # P0 教训: 存 .json, 禁 .txt dump
    r = xgb.Booster(); r.load_model(str(mp))     # reload 自检
    assert r.num_boosted_rounds() > 0, "XGB reload 后无轮次"
    lap(f"  reload OK: {r.num_boosted_rounds():,} 轮")
    pr = np.clip(bst.predict(xgb.DMatrix(Xh, enable_categorical=True),
                             iteration_range=(0, bi)).astype(np.float64), CLIP_LO, CLIP_HI)
    PRED.mkdir(exist_ok=True)
    np.save(PRED / "xgb_holdout.npy", pr)
    jdump({"xgb": {"best_iter": bi, "hit_cap": bi >= CAP, "seed": SEED,
                   "model": mp.name, "model_bytes": mp.stat().st_size,
                   "reload": "OK", "n_rounds_after_reload": int(r.num_boosted_rounds())}}, OUT)
    lap(f"  best_iter={bi}  hit_cap={bi >= CAP}  → 固定轮数 {int(np.ceil(bi*1.1))}")
    del bst, r, d, dv


def do_cat(X, y, i_fit, i_es, i_ho):
    from catboost import CatBoostRegressor
    Xf, Xe, Xh = cat_prep(X.iloc[i_fit]), cat_prep(X.iloc[i_es]), cat_prep(X.iloc[i_ho])
    m = CatBoostRegressor(**dict(BASE_CAT, random_seed=SEED, iterations=CAP))
    lap(f"  cat_prep 完成, 开始训练 (cap={CAP}, od_type=Iter, od_wait=100)")
    m.fit(Xf, y[i_fit], eval_set=(Xe, y[i_es]), use_best_model=True)
    bi = int(m.get_best_iteration() or CAP) + 1
    mp = MDIR / "calib_cat_seed42.json"
    m.save_model(str(mp))
    r = CatBoostRegressor(); r.load_model(str(mp))
    assert r.tree_count_ > 0, "CAT reload 后无树"
    lap(f"  reload OK: {r.tree_count_:,} 树")
    pr = np.clip(m.predict(Xh).astype(np.float64), CLIP_LO, CLIP_HI)
    PRED.mkdir(exist_ok=True)
    np.save(PRED / "cat_holdout.npy", pr)
    jdump({"cat": {"best_iter": bi, "hit_cap": bi >= CAP, "seed": SEED,
                   "model": mp.name, "model_bytes": mp.stat().st_size,
                   "reload": "OK", "n_trees_after_reload": int(r.tree_count_)}}, OUT)
    lap(f"  best_iter={bi}  hit_cap={bi >= CAP}  → 固定轮数 {int(np.ceil(bi*1.1))}")
    del m, r


def do_lgb_pred(X, i_ho):
    """用已落盘的 LGB 模型 (10423 树) 重建 holdout 预测, 作为对照基准。"""
    mp = MDIR / "calib_lgb_seed42.json"
    b = lgb.Booster(model_file=str(mp))
    bi = b.num_trees()
    lap(f"  LGB reload OK: {bi:,} 树")
    pr = np.clip(b.predict(X.iloc[i_ho], num_iteration=bi).astype(np.float64), CLIP_LO, CLIP_HI)
    PRED.mkdir(exist_ok=True)
    np.save(PRED / "lgb_holdout.npy", pr)
    jdump({"lgb": {"best_iter": bi, "hit_cap": False, "seed": SEED,
                   "model": mp.name, "reload": "OK"}}, OUT)
    lap("  LGB holdout 预测已保存")
    del b


def do_all(X, y, mo, i_ho):
    yt = y[i_ho]; moh = mo[i_ho]
    ap = pd.read_parquet(D0 / "_v24_holdout_ap.parquet")["ap"].to_numpy()
    preds = {}
    for fam in ("lgb", "xgb", "cat"):
        f = PRED / f"{fam}_holdout.npy"
        assert f.exists(), f"缺 {f}, 请先跑 --fam {fam}"
        preds[fam] = np.load(f)
        lap(f"  载入 {fam} holdout 预测 n={len(preds[fam]):,}")
    ens3 = (preds["lgb"] + preds["xgb"] + preds["cat"]) / 3.0

    def ev(P):
        r = lambda m: float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))
        return {"full": round(r(np.ones(len(yt), bool)), 2),
                "jan": round(r(moh == 1), 2), "jul": round(r(moh == 7), 2)}

    r = {k: ev(v) for k, v in preds.items()}
    r["ens3"] = ev(ens3)
    print("\n" + "=" * 92)
    print(f"Jan+Jul holdout ({len(yt):,} 行) · seed{SEED} · full RMSE 为主")
    print("=" * 92)
    print(f"{'模型':<16} {'full':>9} {'Jan':>9} {'Jul':>9}")
    for k in ("lgb", "xgb", "cat", "ens3"):
        print(f"{k:<16} {r[k]['full']:>9.2f} {r[k]['jan']:>9.2f} {r[k]['jul']:>9.2f}")
    print("-" * 92)
    print(f"{'Δ ens3 - lgb':<16} {r['ens3']['full']-r['lgb']['full']:>+9.2f} "
          f"{r['ens3']['jan']-r['lgb']['jan']:>+9.2f} {r['ens3']['jul']-r['lgb']['jul']:>+9.2f}")

    # 分机场
    per = {}
    for a in np.unique(ap):
        m = ap == a
        if m.sum() < 200:
            continue
        r2 = lambda P: float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))
        per[str(a)] = {"n": int(m.sum()),
                       "lgb": round(r2(preds["lgb"]), 2),
                       "ens3": round(r2(ens3), 2),
                       "delta": round(r2(ens3) - r2(preds["lgb"]), 2)}
    ds = np.array([v["delta"] for v in per.values()])
    npos = int((ds > 0).sum()); nneg = int((ds < 0).sum())
    same_sign = (npos == 0) or (nneg == 0)
    print(f"\n分机场 (≥200 行, 共 {len(per)} 个机场):")
    print(f"  ens3 更差的机场 {npos} 个, 更好的 {nneg} 个 → "
          f"{'同号' if same_sign else '不同号'}")
    worst = sorted(per.items(), key=lambda kv: -kv[1]["delta"])[:5]
    print("  劣化最多的 5 个机场:")
    for a, v in worst:
        print(f"    {a:<5} n={v['n']:>7,}  lgb={v['lgb']:>9.2f}  ens3={v['ens3']:>9.2f}  "
              f"Δ={v['delta']:>+9.2f}")

    # 判定门槛: Δ 超 seed 抖动 + 分月/分机场同号
    d_full = r["ens3"]["full"] - r["lgb"]["full"]
    d_jan = r["ens3"]["jan"] - r["lgb"]["jan"]
    d_jul = r["ens3"]["jul"] - r["lgb"]["jul"]
    month_ok = (d_jan < 0) == (d_jul < 0)
    verdict = "ENS3_VALID" if (d_full < 0 and month_ok and same_sign) else "ENS3_REJECT"
    print("\n" + "=" * 92)
    print(f"判定: Δfull={d_full:+.2f}  ΔJan={d_jan:+.2f}  ΔJul={d_jul:+.2f}  "
          f"分月同号={month_ok}  分机场同号={same_sign}")
    print(f"  → {verdict}")
    print("  注: 单 seed 口径下 seed 抖动未知; 若 Δ 幅度 <2s 一律视为噪声")

    jdump({"probe": {"per_model": r, "per_airport": per,
                     "delta_ens3_minus_lgb": {"full": round(d_full, 2),
                                              "jan": round(d_jan, 2),
                                              "jul": round(d_jul, 2)},
                     "airport_same_sign": bool(same_sign),
                     "airport_n_worse": npos, "airport_n_better": nneg,
                     "verdict": verdict}}, OUT)
    lap(f"→ {OUT}")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--fam", required=True, choices=["lgb", "xgb", "cat", "all"])
    args = ap_.parse_args()
    MDIR.mkdir(exist_ok=True); PRED.mkdir(exist_ok=True)
    if args.fam == "all":
        X, y, mo, _, _, i_ho = load_split()
        do_all(X, y, mo, i_ho)
        return
    X, y, mo, i_fit, i_es, i_ho = load_split()
    lap(f"=== {args.fam.upper()} seed{SEED} · cap={CAP} ===")
    if args.fam == "lgb":
        do_lgb_pred(X, i_ho)
    elif args.fam == "xgb":
        do_xgb(X, y, i_fit, i_es, i_ho)
    else:
        do_cat(X, y, i_fit, i_es, i_ho)


if __name__ == "__main__":
    main()