"""
v24_febjun_gate.py — final 决策点: Feb/Jun holdout 泛化验证 (LGB 单库)

背景
----
阶段0 的 Jan/Jul holdout 显示序列特征 full RMSE 改善 34.91s, 但那是单次划分、
单 seed。Jan(1月) 与 Jul(7月) 都是**年初/年中**月份, 可能对序列特征特别有利。
本脚本用**全新的 Feb/Jun 两个月**做独立验证 —— 这两个月从未参与过任何选型。

口径 (严格)
------------
  fit    : 2025 除 Feb(2)、Jun(6) 外的月份 DEP
  ES     : fit 训练折按 MVT 排序最后 10% (断言 ES 月份不含 2、6)
  holdout: 2025 Feb + Jun DEP, 报告各月行数
  A = FEATS71 (基线)   B = FEATS71 + 3 序列特征
  同 ES 划分、同 seed、同超参、同 cap → Δ 只反映特征差异

判定 (按用户要求, 不强行纳入)
------------------------------
  * 先 1 seed 快筛
  * 两月同号且超 seed 抖动 → 保留, 阶段2 按 FEATS74 跑
  * 任一月份恶化/无增益 → 如实报告, 可退到 FEATS71 或只保留稳定子集

红线: 只用 LGB 单库快筛 (不跑三库), 不改 label, 不 clip 真值, 不上传。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(D0))
from mvtseq_feats import SEQ_FEATS  # noqa: E402

OUT = D0 / "v24_febjun_gate.json"
CAP = 12000                       # 与阶段1 同 cap, 保证可比
SEED = 42
FIT_MONTHS = [1, 3, 4, 5, 7, 8, 9, 10, 11, 12]     # 除 2、6
HOLD_MONTHS = [2, 6]
ES_FRAC = 0.10
CLIP_LO, CLIP_HI = 30.0, 10800.0
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, float("inf"))]
BLAB = ["[0,600)", "[600,1800)", "[1800,3600)", "[3600,10800]", ">10800"]
PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363, "num_leaves": 127,
    "max_depth": 11, "min_data_in_leaf": 5,
    "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def load():
    import pickle
    blob = pickle.load(open(D0 / "_qinv_feats.pkl", "rb"))
    X = blob["X"]; y = blob["y"].astype(np.float64); mo = blob["mo"]; mvt_s = blob["mvt_s"]
    X = X[[c for c in X.columns if not c.startswith("q_")]]
    assert len(X) == 2_085_047, f"特征行数异常 {len(X)}"
    return X, y, mo, mvt_s


def fit_lgb(X, y, i_fit, i_es, seed, tag):
    p = dict(PARAMS, seed=seed, bagging_seed=seed,
             feature_fraction_seed=seed, data_random_seed=seed)
    d = lgb.Dataset(X.iloc[i_fit], label=y[i_fit], categorical_feature=LOW_CARD_CATS)
    dv = lgb.Dataset(X.iloc[i_es], label=y[i_es], categorical_feature=LOW_CARD_CATS, reference=d)
    b = lgb.train(p, d, num_boost_round=CAP, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    bi = int(b.best_iteration or CAP)
    lap(f"  {tag}: best_iter={bi}")
    return b, bi


def ev(P, yt, mo):
    r = lambda m: float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))
    o = {"full": round(r(np.ones(len(yt), bool)), 2)}
    for mm in HOLD_MONTHS:
        o[f"m{mm}"] = round(r(mo == mm), 2)
    o["bins"] = []
    for (lo, hi), lab in zip(BINS, BLAB):
        m = (yt >= lo) & (yt < hi) if hi != float("inf") else (yt >= lo)
        e = {"bin": lab, "n": int(m.sum()),
             "rmse": round(r(m), 2) if m.sum() >= 5 else None,
             "pred_med": round(float(np.median(P[m])), 1) if m.sum() >= 5 else None,
             "true_med": round(float(np.median(yt[m])), 1) if m.sum() >= 5 else None}
        o["bins"].append(e)
    return o


def main(seeds=(SEED,)):
    lap("加载 FEATS71")
    X, y, mo, mvt_s = load()

    lap("加载序列特征 (seq_feats_2025.parquet)")
    sq = pd.read_parquet(ROOT / "data/interim/seq_feats_2025.parquet")
    assert len(sq) == len(X), f"序列特征行数 {len(sq)} != {len(X)}"
    mo_c = mo.astype(np.int64); mo_sq = sq["month"].to_numpy().astype(np.int64)
    mism = int((mo_c != mo_sq).sum())
    assert mism <= 10, f"月份序列偏移过大 {mism}"
    # 按行取 Feb/Jun 的序列特征 (缓存已覆盖全年)
    lap(f"  行数一致, 逐行月份仅 {mism} 行不同 → 同源同序")

    fit_m = np.isin(mo_c, FIT_MONTHS)
    ho_m = np.isin(mo_c, HOLD_MONTHS)
    i_tr = np.where(fit_m)[0]
    o = i_tr[np.argsort(mvt_s[i_tr], kind="stable")]
    cut = int(len(o) * (1 - ES_FRAC))
    i_fit, i_es = o[:cut], o[cut:]
    es_m = sorted(set(mo_c[i_es].tolist()))
    assert not np.isin(mo_c[i_es], HOLD_MONTHS).any(), f"ES 泄漏到 holdout 月: {es_m}"
    assert not np.intersect1d(i_fit, i_es).size, "fit 与 ES 重叠"
    i_ho = np.where(ho_m)[0]
    counts = {int(m): int((mo_c == m).sum()) for m in HOLD_MONTHS}
    lap(f"fit {len(i_fit):,} | ES {len(i_es):,} (月 {es_m})")
    lap(f"holdout Feb={counts[2]:,}  Jun={counts[6]:,}  合计 {len(i_ho):,}")

    # 标签清洗与 v15 一致, 但**只清洗 fit**(不改 holdout 真值)
    keep = (y[i_fit] > 0) & (y[i_fit] <= 10800)
    n_raw = len(i_fit); i_fit = i_fit[keep]
    lap(f"fit 清洗: {n_raw:,} → {len(i_fit):,} (剔 {n_raw - len(i_fit):,})")

    Xb = X.copy()
    for c in SEQ_FEATS:
        Xb[c] = sq[c].to_numpy(np.float32)
    assert list(Xb.columns[-3:]) == SEQ_FEATS

    yt = y[i_ho]; moh = mo_c[i_ho]
    res = {"fit_months": FIT_MONTHS, "holdout_months": HOLD_MONTHS,
           "es_months": es_m, "cap": CAP, "seeds": list(seeds),
           "n_fit": int(len(i_fit)), "n_es": int(len(i_es)),
           "holdout_rows": counts, "n_holdout": int(len(i_ho)),
           "fit_clean": {"raw": int(n_raw), "kept": int(len(i_fit))},
           "runs": {}}

    for seed in seeds:
        lap(f"\n===== seed {seed} =====")
        bA, biA = fit_lgb(X, y, i_fit, i_es, seed, "A FEATS71")
        pA = np.clip(bA.predict(X.iloc[i_ho], num_iteration=biA).astype(np.float64), CLIP_LO, CLIP_HI)
        bB, biB = fit_lgb(Xb, y, i_fit, i_es, seed, "B FEATS74")
        pB = np.clip(bB.predict(Xb.iloc[i_ho], num_iteration=biB).astype(np.float64), CLIP_LO, CLIP_HI)

        rA, rB = ev(pA, yt, moh), ev(pB, yt, moh)
        imp = sorted(zip(Xb.columns, bB.feature_importance("gain")), key=lambda x: -x[1])
        rank = {c: i + 1 for i, (c, _) in enumerate(imp)}
        run = {"A": rA, "B": rB, "best_iter": {"A": biA, "B": biB},
               "delta": {"full": round(rB["full"] - rA["full"], 2),
                         "m2": round(rB["m2"] - rA["m2"], 2),
                         "m6": round(rB["m6"] - rA["m6"], 2)},
               "seq_gain": [{"feat": c, "gain": round(float(bB.feature_importance("gain")[Xb.columns.get_loc(c)])),
                             "rank": rank[c]} for c in SEQ_FEATS],
               "n_features": len(Xb.columns)}
        res["runs"][f"seed{seed}"] = run

        print("\n" + "=" * 96)
        print(f"Feb/Jun holdout · seed{seed} · LGB 单库 · L2")
        print("=" * 96)
        print(f"{'组':<20} {'full':>9} {'Feb':>9} {'Jun':>9} {'best_iter':>10}")
        print(f"{'A FEATS71':<20} {rA['full']:>9.2f} {rA['m2']:>9.2f} {rA['m6']:>9.2f} {biA:>10}")
        print(f"{'B FEATS74':<20} {rB['full']:>9.2f} {rB['m2']:>9.2f} {rB['m6']:>9.2f} {biB:>10}")
        print("-" * 96)
        print(f"{'Δ (B-A)':<20} {run['delta']['full']:>+9.2f} {run['delta']['m2']:>+9.2f} "
              f"{run['delta']['m6']:>+9.2f}   (负=改善)")
        print("\n分档 RMSE (重点看尾段):")
        print(f"{'档':>14} {'n':>8} {'A':>10} {'B':>10} {'Δ':>9} {'A真值中位':>10} {'B预测中位':>10}")
        for e1, e2 in zip(rA["bins"], rB["bins"]):
            if e1["rmse"] is None:
                print(f"{e1['bin']:>14} {e1['n']:>8,}   (样本不足)")
                continue
            print(f"{e1['bin']:>14} {e1['n']:>8,} {e1['rmse']:>10.2f} {e2['rmse']:>10.2f} "
                  f"{e2['rmse']-e1['rmse']:>+9.2f} {e1['true_med']:>10.1f} {e2['pred_med']:>10.1f}")
        print("\n3 个序列特征 gain 排名 (共 %d 特征):" % len(Xb.columns))
        for s in sorted(run["seq_gain"], key=lambda x: x["rank"]):
            print(f"  #{s['rank']:<3} {s['feat']:<28} gain={s['gain']:.4g}")
        del bA, bB

    # ---------- 多 seed 判定 ----------
    print("\n" + "=" * 96)
    print("判定")
    print("=" * 96)
    if len(seeds) == 1:
        d = res["runs"][f"seed{seeds[0]}"]["delta"]
        same_sign = (d["m2"] < 0) == (d["m6"] < 0)
        print(f"  单 seed 快筛: Feb Δ={d['m2']:+.2f}  Jun Δ={d['m6']:+.2f}  "
              f"两月同号={'是' if same_sign else '否'}")
        res["verdict_1seed"] = ("CANDIDATE_NEEDS_MORE_SEEDS" if same_sign and
                                min(-d["m2"], -d["m6"]) > 0 else "REJECT")
        print(f"  → {res['verdict_1seed']}")
    else:
        ds = [res["runs"][f"seed{s}"] for s in seeds]
        fm = np.array([d["full"] for d in ds]); f2 = np.array([d["m2"] for d in ds])
        f6 = np.array([d["m6"] for d in ds])
        print(f"  Δ full: {np.round(fm,2).tolist()}  均值 {fm.mean():+.2f}  std {fm.std():.2f}")
        print(f"  Δ Feb : {np.round(f2,2).tolist()}  均值 {f2.mean():+.2f}  std {f2.std():.2f}")
        print(f"  Δ Jun : {np.round(f6,2).tolist()}  均值 {f6.mean():+.2f}  std {f6.std():.2f}")
        jitter = max(f2.std(), f6.std())
        ok = (f2.mean() < 0 and f6.mean() < 0 and
              abs(f2.mean()) > jitter and abs(f6.mean()) > jitter)
        res["verdict_multi"] = "KEEP_SEQ_FEATS" if ok else "REJECT_SEQ_FEATS"
        res["jitter"] = round(float(jitter), 3)
        res["delta_mean"] = {"full": round(float(fm.mean()), 2),
                             "m2": round(float(f2.mean()), 2),
                             "m6": round(float(f6.mean()), 2)}
        print(f"  seed 抖动 std = {jitter:.2f}s (阈值: 均值绝对值须大于抖动)")
        print(f"  → {res['verdict_multi']}")

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    lap(f"→ {OUT}")


if __name__ == "__main__":
    main()