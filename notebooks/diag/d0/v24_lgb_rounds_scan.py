"""
v24_lgb_rounds_scan.py — LGB 单库 seed42 轮数-增益扫描

目的: 量化"补足轮数"相对 3000 轮的 full RMSE 净增益, 给出全年固定轮数建议。

口径 (与阶段1 LGB 标定完全一致)
--------------------------------
  fit    : 2025-02..12 DEP, 清洗 (剔 TAXITIME<=0 或 >10800) → FEATS71
  ES     : fit 按 MVT 排序最后 10% (月份 ⊆ {11,12}), 与 fit 零重叠
  holdout: 2025 Jan+Jul (344,419 行) —— 加入 valid_sets 但 **不用于早停**

做法
----
一次训练到 13000 轮 (无早停), save_model 后用 num_iteration=k 对各检查点
重新预测。这样所有检查点来自**同一次训练**, 曲线自洽, 且能得到 Jan/Jul 与
分机场的逐检查点表现 (record_evaluation 只给整体 RMSE, 给不了分月)。
record_evaluation 同时记录整体 holdout 分数作为交叉校验。

红线: 不改 label, 不 clip 真值, 统计只用 fit, holdout 不参与选型, 不上传。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
MDIR = D0 / "v24_models"
OUT = D0 / "v24_lgb_rounds_scan.json"
MP = MDIR / "scan_lgb_seed42.json"

CAP = 13000
SEED = 42
CKPTS = [3000, 6000, 9000, 10423, 11000, 12000, 13000]
BASE_ROUND = 3000                       # Δ 的参照点 (v15/v24 现状)
SAT_STEP = 1000                         # 饱和判定步长
SAT_EPS = 2.0                           # 改善 < 2s 视为饱和
CLIP_LO, CLIP_HI = 30.0, 10800.0
TRAIN_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD_MONTHS = [1, 7]
LOW_CARD_CATS = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
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


def main():
    import pickle
    lap("加载 FEATS71 缓存")
    blob = pickle.load(open(D0 / "_qinv_feats.pkl", "rb"))
    X = blob["X"]; y = blob["y"].astype(np.float64)
    mo = blob["mo"].astype(np.int64); mvt_s = blob["mvt_s"]
    X = X[[c for c in X.columns if not c.startswith("q_")]]
    assert len(X) == 2_085_047 and len(X.columns) == 71

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

    ap = pd.read_parquet(D0 / "_v24_holdout_ap.parquet")["ap"].to_numpy()
    assert len(ap) == len(i_ho)
    yt = y[i_ho]; moh = mo[i_ho]

    # ---------------- 训练 (无早停, holdout 仅记录) ----------------
    hist = {}
    if MP.exists():
        lap(f"复用已训模型 {MP.name}")
        b = lgb.Booster(model_file=str(MP))
        assert b.num_trees() == CAP, f"树数 {b.num_trees()} != {CAP}"
    else:
        p = dict(PARAMS, seed=SEED, bagging_seed=SEED,
                 feature_fraction_seed=SEED, data_random_seed=SEED)
        d = lgb.Dataset(X.iloc[i_fit], label=y[i_fit], categorical_feature=LOW_CARD_CATS)
        dv_es = lgb.Dataset(X.iloc[i_es], label=y[i_es],
                            categorical_feature=LOW_CARD_CATS, reference=d)
        dv_ho = lgb.Dataset(X.iloc[i_ho], label=yt, categorical_feature=LOW_CARD_CATS,
                            reference=d)
        lap(f"训练中: cap={CAP}, 无早停, valid_sets=[ES, holdout] (holdout 不参与选型)")
        b = lgb.train(p, d, num_boost_round=CAP, valid_sets=[dv_es, dv_ho],
                      valid_names=["es", "holdout"],
                      callbacks=[lgb.record_evaluation(hist),
                                 lgb.log_evaluation(0)])
        b.save_model(str(MP))
        lap(f"训练完成 {b.num_trees():,} 树 → {MP.name} ({MP.stat().st_size/1e6:.0f}MB)")
        if hist:
            hv = hist.get("valid_1", {})
            if hv:
                lap(f"  record_evaluation 记录 {len(hv)} 轮 holdout 分数 (交叉校验用)")
    Xh = X.iloc[i_ho]

    def rm(P, m=None):
        m = np.ones(len(yt), bool) if m is None else m
        return float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))

    # ---------------- 逐检查点预测 ----------------
    rows = []
    for k in CKPTS:
        P = np.clip(b.predict(Xh, num_iteration=k).astype(np.float64), CLIP_LO, CLIP_HI)
        rows.append({"rounds": k, "full": round(rm(P), 2),
                     "jan": round(rm(P, moh == 1), 2),
                     "jul": round(rm(P, moh == 7), 2),
                     "_P": P})
        lap(f"  iter {k:>6,}: full {rows[-1]['full']:>8.2f}  "
            f"Jan {rows[-1]['jan']:>8.2f}  Jul {rows[-1]['jul']:>8.2f}")

    base = next(r for r in rows if r["rounds"] == BASE_ROUND)
    for r in rows:
        r["d_full"] = round(r["full"] - base["full"], 2)
        r["d_jan"] = round(r["jan"] - base["jan"], 2)
        r["d_jul"] = round(r["jul"] - base["jul"], 2)
        r.pop("_P")

    # ---------------- 饱和点 ----------------
    curve = {r["rounds"]: r["full"] for r in rows}
    sat = None
    ks = sorted(curve)
    for i, k in enumerate(ks):
        nxt = k + SAT_STEP
        if nxt in curve:
            imp = curve[k] - curve[nxt]
            lap(f"  饱和检查 {k:,} → {nxt:,}: 改善 {imp:+.2f}s")
            if imp < SAT_EPS:
                sat = k
                break
    if sat is None:
        sat = ks[-1]

    # ---------------- 分机场 Δ (饱和点 vs 3000) ----------------
    Psat = np.clip(b.predict(Xh, num_iteration=sat).astype(np.float64), CLIP_LO, CLIP_HI)
    Pbase = np.clip(b.predict(Xh, num_iteration=BASE_ROUND).astype(np.float64),
                    CLIP_LO, CLIP_HI)
    per = {}
    for a in np.unique(ap):
        m = ap == a
        if m.sum() < 200:
            continue
        per[str(a)] = {"n": int(m.sum()),
                       "base": round(float(np.sqrt(np.mean(
                           (Pbase[m] - yt[m]) ** 2))), 2),
                       "sat": round(float(np.sqrt(np.mean((Psat[m] - yt[m]) ** 2))), 2)}
        per[str(a)]["delta"] = round(per[str(a)]["sat"] - per[str(a)]["base"], 2)
    ds = np.array([v["delta"] for v in per.values()])
    npos = int((ds > 0).sum()); nneg = int((ds < 0).sum())
    same_sign = (npos == 0) or (nneg == 0)

    # ---------------- 报告 ----------------
    print("\n" + "=" * 94)
    print(f"LGB seed{SEED} 轮数-增益扫描 · Jan+Jul holdout {len(yt):,} 行 · FEATS71 · L2")
    print("=" * 94)
    print(f"{'轮数':>8} {'full':>9} {'Δfull':>9} {'Jan':>9} {'ΔJan':>9} "
          f"{'Jul':>9} {'ΔJul':>9}")
    for r in rows:
        print(f"{r['rounds']:>8,} {r['full']:>9.2f} {r['d_full']:>+9.2f} "
              f"{r['jan']:>9.2f} {r['d_jan']:>+9.2f} {r['jul']:>9.2f} {r['d_jul']:>+9.2f}")
    print("-" * 94)
    print(f"增益饱和点: {sat:,} 轮 (再增 {SAT_STEP:,} 轮改善 < {SAT_EPS}s)")
    print(f"补轮数净增益: full {rows[-1]['d_full']:+.2f}s  (3000 → {rows[-1]['rounds']:,})")
    print(f"                饱和点处 {next(r['d_full'] for r in rows if r['rounds']==sat):+.2f}s "
          f"({BASE_ROUND:,} → {sat:,})")
    print(f"\n分机场 (≥200 行, 共 {len(per)} 个):  饱和点 vs {BASE_ROUND:,}")
    print(f"  变差 {npos} 个, 变好 {nneg} 个 → {'同号' if same_sign else '不同号'}")
    for a, v in sorted(per.items(), key=lambda kv: kv[1]["delta"]):
        print(f"    {a:<5} n={v['n']:>7,}  {BASE_ROUND:,}={v['base']:>8.2f}  "
              f"{sat:,}={v['sat']:>8.2f}  Δ={v['delta']:>+8.2f}")

    res = {"cap": CAP, "seed": SEED, "n_fit": int(len(i_fit)),
           "fit_clean_raw": int(n_raw), "n_es": int(len(i_es)), "es_months": es_m,
           "n_holdout": int(len(i_ho)), "checkpoints": rows,
           "base_rounds": BASE_ROUND, "sat_point": sat,
           "sat_rule": f"再增 {SAT_STEP} 轮 full RMSE 改善 < {SAT_EPS}s",
           "per_airport": per, "airport_n_worse": npos, "airport_n_better": nneg,
           "airport_same_sign": bool(same_sign),
           "net_gain_full": rows[-1]["d_full"],
           "net_gain_at_sat": next(r["d_full"] for r in rows if r["rounds"] == sat),
           "suggest_fixed_rounds": sat}
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    lap(f"→ {OUT}")


if __name__ == "__main__":
    main()