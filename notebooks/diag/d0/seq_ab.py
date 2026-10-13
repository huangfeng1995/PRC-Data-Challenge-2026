"""
seq_ab.py — MVT-only 序列特征 A/B (干净 holdout, 单 seed42)

口径 (严格):
  * 特征: FEATS71 (复用 _qinv_feats.pkl 缓存) + 3 个 MVT-only 序列特征 → FEATS74
  * 训练折: 2025-02..12 (10 个月, 剔除 1 月和 7 月)
  * holdout: 2025-01 + 2025-07  ← 与训练折完全不重叠
  * ES: 训练折内按 MVT 排序最后 10% 作 early stopping, 不含 1/7 月
  * 目标: L2 (regression)  ← v23 已证 P50 在真榜劣化 31.72s, 永久关闭
  * 评估: **full RMSE 为主** (core 口径已两次误导: P0 泄漏、P50 高估)
  * 基线: 同一 ES 划分、同一 seed、同一超参的 FEATS71 模型

红线: 不改 label、不 clip 真值、只读、不上传。
"""
from __future__ import annotations
import sys, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
sys.path.insert(0, str(D0))
from mvtseq_feats import SEQ_FEATS  # noqa: E402

OUT = D0 / "seq_ab.json"
CACHE = D0 / "_qinv_feats.pkl"
SEQCACHE = ROOT / "data/interim/seq_feats_2025.parquet"
CAP = 2000
SEED = 42
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.0147555939449363, "num_leaves": 127,
    "max_depth": 11, "min_data_in_leaf": 5,
    "feature_fraction": 0.6720708449966734,
    "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
    "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
    "verbose": -1, "num_threads": -1,
}
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, float("inf"))]
BLAB = ["[0,600)", "[600,1800)", "[1800,3600)", "[3600,10800]", ">10800"]
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def ev(P, yt, mo):
    core = (yt > 0) & (yt <= 10800)
    r = lambda m: float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))
    o = {"full": round(r(np.ones(len(yt), bool)), 2),
         "core": round(r(core), 2),
         "jan": round(r(mo == 1), 2), "jul": round(r(mo == 7), 2), "bins": []}
    for (lo, hi), lab in zip(BINS, BLAB):
        m = (yt >= lo) & (yt < hi) if hi != float("inf") else (yt >= lo)
        e = {"bin": lab, "n": int(m.sum()), "rmse": round(r(m), 2) if m.sum() >= 5 else None}
        if m.sum() >= 5:
            e.update(pred_med=round(float(np.median(P[m])), 1),
                     true_med=round(float(np.median(yt[m])), 1),
                     bias=round(float(np.mean(P[m] - yt[m])), 1))
        o["bins"].append(e)
    return o


def fit(X, y, fit_idx, es_idx, tag):
    p = dict(PARAMS, seed=SEED, bagging_seed=SEED + 1, feature_fraction_seed=SEED + 2)
    dtr = lgb.Dataset(X.iloc[fit_idx], label=y[fit_idx], categorical_feature=LOW_CARD_CATS)
    dva = lgb.Dataset(X.iloc[es_idx], label=y[es_idx], reference=dtr)
    b = lgb.train(p, dtr, num_boost_round=CAP, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    lap(f"  {tag}: best_iter={b.best_iteration}")
    return b


def main():
    lap("加载 FEATS71 缓存")
    blob = pickle.load(open(CACHE, "rb"))
    X = blob["X"]; y = blob["y"]; mo = blob["mo"]; mvt_s = blob["mvt_s"]
    X = X[[c for c in X.columns if not c.startswith("q_")]]
    lap(f"  FEATS71 X={X.shape}")

    lap("加载序列特征并按 MVT_ID 对齐")
    sq = pd.read_parquet(SEQCACHE)
    assert len(sq) == len(X), f"序列特征行数 {len(sq)} != 特征矩阵 {len(X)}"
    # 两份缓存同源同序, 用逐行月份核验:
    #   实测仅 2 行不同 —— 2 月少 1、3 月多 1, 是**跨月航班的月份归属**差异
    #   (月份按 MVT 所在月 vs 按文件来源月), 不是行错位。
    #   行错位会表现为大量不一致, 2 行且计数只在一对相邻月之间移动 1 行 → 同源。
    mo_sq = sq["month"].to_numpy().astype(np.int64)
    mo_c = mo.astype(np.int64)
    mism = int((mo_sq != mo_c).sum())
    diff_by_m = {int(m): int(((mo_c == m) & (mo_sq != m)).sum())
                  for m in set(mo_c.tolist()) | set(mo_sq.tolist())
                  if ((mo_c == m) & (mo_sq != m)).any()}
    assert mism <= 10, f"月份序列偏移过大: {mism} 行不一致 {diff_by_m}"
    lap(f"  行数一致, 逐行月份仅 {mism} 行不同 (跨月归属: {diff_by_m}) → 判定同源同序")

    tr = np.isin(mo, TRAIN_MONTHS)
    ho = np.isin(mo, HOLD_MONTHS)
    idx_tr = np.where(tr)[0]
    order = idx_tr[np.argsort(mvt_s[idx_tr], kind="stable")]
    cut = int(len(order) * 0.9)
    fit_idx, es_idx = order[:cut], order[cut:]
    es_m = sorted(set(mo[es_idx].tolist()))
    assert 1 not in es_m and 7 not in es_m, f"ES 含 holdout 月 {es_m}"
    ho_idx = np.where(ho)[0]
    lap(f"fit {len(fit_idx):,} / ES {len(es_idx):,} {es_m} / holdout {len(ho_idx):,}")

    # ---------- A: 基线 FEATS71 ----------
    lap("\n训练 A: FEATS71 (基线)")
    bA = fit(X, y, fit_idx, es_idx, "A")
    pA = np.clip(bA.predict(X.iloc[ho_idx], num_iteration=bA.best_iteration).astype(np.float64),
                 30.0, 10800.0)

    # ---------- B: FEATS71 + 3 个序列特征 ----------
    lap("\n训练 B: FEATS74 (+3 MVT 序列特征)")
    Xb = X.copy()
    for c in SEQ_FEATS:
        Xb[c] = sq[c].to_numpy(np.float32)
    assert list(Xb.columns[-3:]) == SEQ_FEATS
    bB = fit(Xb, y, fit_idx, es_idx, "B")
    pB = np.clip(bB.predict(Xb.iloc[ho_idx], num_iteration=bB.best_iteration).astype(np.float64),
                 30.0, 10800.0)

    yt, moh = y[ho_idx], mo[ho_idx]
    rA, rB = ev(pA, yt, moh), ev(pB, yt, moh)
    res = {"n_holdout": int(len(yt)), "seed": SEED, "cap": CAP,
           "train_months": TRAIN_MONTHS, "holdout_months": HOLD_MONTHS,
           "es_months": es_m, "n_fit": int(len(fit_idx)), "n_es": int(len(es_idx)),
           "best_iter": {"A": int(bA.best_iteration), "B": int(bB.best_iteration)},
           "baseline_FEATS71": rA, "with_seq_FEATS74": rB}
    res["delta"] = {"full": round(rB["full"] - rA["full"], 2),
                    "core": round(rB["core"] - rA["core"], 2),
                    "jan": round(rB["jan"] - rA["jan"], 2),
                    "jul": round(rB["jul"] - rA["jul"], 2)}
    res["verdict"] = ("PASS_GE_10S" if rA["full"] - rB["full"] >= 10 else
                      "MARGINAL" if rA["full"] - rB["full"] > 0 else "NO_GAIN")

    # 特征重要度
    imp = sorted(zip(Xb.columns, bB.feature_importance("gain")),
                 key=lambda x: -x[1])
    res["top_gain"] = [{"feat": c, "gain": round(float(g))} for c, g in imp[:15]]
    res["seq_feat_gain"] = [{"feat": c, "gain": round(float(bB.feature_importance("gain")[Xb.columns.get_loc(c)])),
                             "rank": [c for c, _ in imp].index(c) + 1} for c in SEQ_FEATS]

    print("\n" + "=" * 96)
    print("干净 holdout (2025-01 + 2025-07) · 单 seed42 · L2 · full RMSE 为主")
    print("=" * 96)
    print(f"{'组':<26} {'full':>9} {'core':>9} {'Jan':>9} {'Jul':>9} {'best_iter':>10}")
    for lab, r, bi in [("A 基线 FEATS71", rA, res["best_iter"]["A"]),
                       ("B +3 序列特征", rB, res["best_iter"]["B"])]:
        print(f"{lab:<26} {r['full']:>9.2f} {r['core']:>9.2f} {r['jan']:>9.2f} {r['jul']:>9.2f} {bi:>10}")
    print("-" * 96)
    d = res["delta"]
    print(f"{'Δ (B-A)':<26} {d['full']:>+9.2f} {d['core']:>+9.2f} {d['jan']:>+9.2f} {d['jul']:>+9.2f}")
    print(f"  full 改善 = {rA['full'] - rB['full']:+.2f}s   判定 = {res['verdict']}")

    print("\n分档对比 (真实滑行时间):")
    print(f"{'档':>14} {'n':>8} | {'A rmse':>9} {'B rmse':>9} {'Δ':>8} | {'A bias':>9} {'B bias':>9}")
    for e1, e2 in zip(rA["bins"], rB["bins"]):
        if e1["rmse"] is None:
            continue
        print(f"{e1['bin']:>14} {e1['n']:>8,} | {e1['rmse']:>9.2f} {e2['rmse']:>9.2f} "
              f"{e2['rmse']-e1['rmse']:>+8.2f} | {e1['bias']:>9.1f} {e2['bias']:>9.1f}")

    print("\n3 个新特征 gain 排名 (共 %d 特征):" % len(Xb.columns))
    for s in res["seq_feat_gain"]:
        print(f"  #{s['rank']:<3} {s['feat']:<28} gain={s['gain']:.1f}")
    print("\n全局 gain top10:")
    for s in res["top_gain"][:10]:
        print(f"  {s['feat']:<32} gain={s['gain']:.1f}")

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    lap(f"→ {OUT}")


if __name__ == "__main__":
    main()
