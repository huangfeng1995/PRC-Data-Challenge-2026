"""
ngb_probe.py — NGBoost (分布回归) vs LGB GBDT 基线, 只读复用缓存

目的
----
NGBoost 不只给点预测 (loc/μ), 还给不确定性的 scale (σ)。本脚本检验:
  1. μ 的 full RMSE 是否能打平/超过 LGB GBDT;
  2. σ 是否真的随残差增大 (即不确定性估计是否有信号)。

口径 (与 seq_ab.py / v24 系列一致)
------------------------------------
  fit      : mo ∈ {2,3,4,5,6,8,9,10,11,12}, 清洗 (剔 TAXITIME<=0 或 >10800)
  ES       : fit 按 MVT 排序最后 10%, 断言月份 ⊆ {11,12} 且不含 1、7
  holdout1 : mo ∈ {1,7}  ← 主对比
  holdout2 : mo ∈ {2,6}  ← 泛化验证

红线: 只读缓存, 不改 label, 不写 submissions/, 不上传, 严格串行。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# NGBoost 训练时树会试探极端 scale, 触发 overflow 瞬时警告 (实测 σ 本身正常,
# 中位 801s、无非有限值), 故局部抑制, 不掩盖真实错误。
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*ngboost.*")

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
CACHE = D0 / "_qinv_feats.pkl"
OUT = D0 / "ngb_probe.json"

FIT_MONTHS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
HOLD1 = [1, 7]
HOLD2 = [2, 6]
SEED = 42
LGB_ROUNDS = 3000
CLIP_LO, CLIP_HI = 30.0, 10800.0
BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 10800), (10800, float("inf"))]
BLAB = ["[0,600)", "[600,1800)", "[1800,3600)", "[3600,10800]", ">10800"]
SMOKE_N = 200_000          # 步骤2 小样本
_t0 = time.time()


def lap(m=""):
    print(f"[{time.time() - _t0:7.1f}s] {m}", flush=True)


def load():
    """加载缓存 → FEATS71 (去 q_)。返回两份 X:
       X_raw  保留 category dtype, 供 LGB 基线 (它原生支持类别)
       X_num  类别转整数 + NaN 填 -999, 供 sklearn/NGBoost
    """
    import re
    blob = pickle_load(CACHE)
    X = blob["X"]; y = blob["y"].astype(np.float64)
    mo = blob["mo"].astype(np.int64); mvt_s = blob["mvt_s"].astype(np.float64)
    src = (D0 / "lgb_full_train.py").read_text()
    f71 = re.findall(r'"([^"]+)"',
                     re.search(r"FEATS71\s*=\s*\[(.*?)\]", src, re.S).group(1))
    assert len(f71) == 71, f"FEATS71 解析出 {len(f71)} 列"
    X = X[[c for c in X.columns if not c.startswith("q_")]]
    assert list(X.columns) == f71, "列顺序与 FEATS71 不一致"
    assert len(X) == 2_085_047, f"行数 {len(X)}"
    X_raw = X.copy()

    # 类别 → 整数编码 (sklearn 树不原生处理类别, 这是 NGBoost 的天然劣势)
    X_num = X.copy()
    cats = [c for c in X_num.columns if str(X_num[c].dtype) == "category"]
    for c in cats:
        X_num[c] = X_num[c].cat.codes.astype(np.float32)
    lap(f"缓存载入 {X.shape[0]:,}×{X.shape[1]}")
    nan_cols = [c for c in X_num.columns if X_num[c].isna().any()]
    if nan_cols:
        lap(f"  X_num 含 NaN {len(nan_cols)} 列 → 填 -999")
        X_num[nan_cols] = X_num[nan_cols].fillna(-999.0)
    assert not X_num.isna().any().any(), "X_num 仍有 NaN"
    lap(f"  X_raw 保留 category {cats}")
    lap(f"  X_num 类别转整数, NaN 已填")
    return X_raw, X_num, y, mo, mvt_s


def pickle_load(p):
    import pickle
    return pickle.load(open(p, "rb"))


def split(mo, mvt_s):
    fm = np.isin(mo, FIT_MONTHS)
    i_tr = np.where(fm)[0]
    o = i_tr[np.argsort(mvt_s[i_tr], kind="stable")]
    cut = int(len(o) * 0.9)
    i_fit, i_es = o[:cut], o[cut:]
    es_m = sorted(set(mo[i_es].tolist()))
    assert set(es_m) <= {11, 12}, f"ES 月份异常: {es_m}"
    assert not np.isin(mo[i_es], HOLD1).any(), "ES 泄漏到 holdout1"
    assert not np.intersect1d(i_fit, i_es).size, "fit 与 ES 重叠"
    return i_fit, i_es, es_m


def clean(y, i):
    k = (y[i] > 0) & (y[i] <= 10800)
    return i[k], int((~k).sum())


def rmse(P, yt, m=None):
    m = np.ones(len(yt), bool) if m is None else m
    return float(np.sqrt(np.mean((P[m] - yt[m]) ** 2)))


def binned(P, yt):
    out = []
    for (lo, hi), lab in zip(BINS, BLAB):
        m = (yt >= lo) & (yt < hi) if hi != float("inf") else (yt >= lo)
        out.append({"bin": lab, "n": int(m.sum()),
                    "rmse": round(rmse(P, yt, m), 2) if m.sum() >= 5 else None})
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return round(float(r.statistic), 4), round(float(r.pvalue), 4)


# ---------------------------------------------------------------- LGB 基线
def run_lgb(X, y, i_fit, i_es, ho):
    """X 必须是保留 category dtype 的原始版本。"""
    import lightgbm as lgb
    LOW = ["MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt"]
    assert str(X[LOW[0]].dtype) == "category", "LGB 基线需 category dtype"
    p = {"objective": "regression", "metric": "rmse",
         "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
         "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
         "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
         "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
         "verbose": -1, "num_threads": -1, "seed": SEED, "bagging_seed": SEED,
         "feature_fraction_seed": SEED, "data_random_seed": SEED}
    d = lgb.Dataset(X.iloc[i_fit], label=y[i_fit], categorical_feature=LOW)
    dv = lgb.Dataset(X.iloc[i_es], label=y[i_es], categorical_feature=LOW, reference=d)
    lap(f"LGB 训练 {LGB_ROUNDS} 轮 …")
    b = lgb.train(p, d, num_boost_round=LGB_ROUNDS, valid_sets=[dv],
                  callbacks=[lgb.log_evaluation(0)])
    res = {}
    for nm, ih in ho.items():
        P = np.clip(b.predict(X.iloc[ih], num_iteration=LGB_ROUNDS).astype(np.float64),
                    CLIP_LO, CLIP_HI)
        res[nm] = {"full": round(rmse(P, y[ih]), 2),
                   "bins": binned(P, y[ih]),
                   "pred_p50": round(float(np.median(P)), 1),
                   "pred_max": round(float(P.max()), 1)}
        np.save(D0 / f"_ngb_lgb_{nm}.npy", P)
    del b
    return res


# ---------------------------------------------------------------- NGBoost
def run_ngb(X, y, i_fit, i_es, ho, n_estimators=500, tag=""):
    from ngboost import NGBRegressor
    # ---- 分布选择: 实测 T(StudentT) 在多特征下不可用 ----
    # ngboost 0.5.x 把 StudentT 改名 T, df 改为可学习第三参数 (n_params=3)。
    # 但 TLogScore 的 df 梯度含 digamma(df/2), df→0 时数值发散:
    #   · 3 特征 → OK;  70 特征 → "Input y contains NaN"(树分裂产生极小
    #     var, digamma 参数非法 → 梯度 NaN → 下一轮 y 变 NaN)。
    # Normal 无此问题, 且 NGBoost 官方对回归默认就是 Normal。
    # 故主实验用 Normal; T 仅作对照在 3 特征 smoke 里验证过可用。
    from ngboost.distns import Normal as Dist
    from sklearn.tree import DecisionTreeRegressor

    m = NGBRegressor(Dist=Dist,
                     Base=DecisionTreeRegressor(max_depth=3, random_state=SEED),
                     n_estimators=n_estimators, learning_rate=0.03, verbose=False)

    # 先落盘再评估: NGBoost 全量训练实测 4.8 小时, 若评估阶段崩溃
    # (此前就因一个过严的 sigma 断言白丢一次), 可直接复用模型而不重训。
    import pickle
    mpath = D0 / f"_ngb_model_{tag.strip('[] ')}.pkl"
    if mpath.exists():
        m = pickle.load(open(mpath, "rb"))
        fit_s = 0.0
        lap(f"NGBoost{tag} 复用已训模型 {mpath.name}")
    else:
        t_fit = time.time()
        m.fit(X.iloc[i_fit], y[i_fit], X_val=X.iloc[i_es], Y_val=y[i_es])
        fit_s = time.time() - t_fit
        pickle.dump(m, open(mpath, "wb"), protocol=4)
        lap(f"NGBoost{tag} 训练完成 {fit_s:.1f}s  已落盘 {mpath.name}")
    # 0.5.x 里 best_val_loss_ 改名 best_val_loss_itr, 且只在传了 valid 时才有
    vl = getattr(m, "best_val_loss_itr", None)
    lap(f"NGBoost{tag} 训练完成 {fit_s:.1f}s  best_iter={vl}")
    res = {"fit_seconds": round(fit_s, 1), "n_estimators": n_estimators,
           "dist": "Normal", "best_iter": int(vl) if vl is not None else None,
           "per_holdout": {}}
    for nm, ih in ho.items():
        t0 = time.time()
        D = m.pred_dist(X.iloc[ih])
        loc = D.loc; scale = D.scale
        assert np.isfinite(loc).all(), f"{nm} loc 有非有限值"
        assert np.isfinite(scale).all(), f"{nm} scale 有非有限值"
        assert (scale > 0).all(), f"{nm} scale 有非正值"
        # 注: 早期版本这里断言 scale.max()<1e4, 结果把 max=2.45e4 的长尾行
        # 误判为失败、白白丢掉 4.8 小时训练。σ 大是合法结果 (那些行本就
        # 极度不确定), 只记录不拦截。
        n_ovf = int((scale > 1e4).sum())
        P = np.clip(loc, CLIP_LO, CLIP_HI)
        yt = y[ih]
        r = np.abs(P - yt)
        rho, pv = spearman(scale, r)
        hi = yt > 1800
        lo = yt < 600
        res["per_holdout"][nm] = {
            "n": int(len(ih)),
            "full": round(rmse(P, yt), 2),
            "bins": binned(P, yt),
            "sigma_vs_absresid_spearman": rho,
            "sigma_vs_absresid_p": pv,
            "sigma_median_gt1800": round(float(np.median(scale[hi])), 3),
            "sigma_median_lt600": round(float(np.median(scale[lo])), 3),
            "sigma_ratio": round(float(np.median(scale[hi]) / np.median(scale[lo])), 3),
            "sigma_max": round(float(scale.max()), 1),
            "n_sigma_gt_1e4": n_ovf,
            "loc_median": round(float(np.median(loc)), 1),
            "scale_median": round(float(np.median(scale)), 3),
            "pred_max": round(float(P.max()), 1),
            "predict_seconds": round(time.time() - t0, 1),
        }
        np.save(D0 / f"_ngb_{nm}_loc.npy", loc.astype(np.float32))
        np.save(D0 / f"_ngb_{nm}_scale.npy", scale.astype(np.float32))
    res["model"] = m
    return res


def report(tag, lgb_res, ngb_res):
    print("\n" + "=" * 96)
    print(f"{tag} · full RMSE (主指标) · seed{SEED}")
    print("=" * 96)
    print(f"{'模型':<22} {'holdout1(Jan+Jul)':>19} {'holdout2(Feb+Jun)':>19}")
    print(f"{'LGB GBDT (基线)':<22} {lgb_res['h1']['full']:>19.2f} {lgb_res['h2']['full']:>19.2f}")
    print(f"{'NGBoost loc (μ)':<22} {ngb_res['per_holdout']['h1']['full']:>19.2f} "
          f"{ngb_res['per_holdout']['h2']['full']:>19.2f}")
    d1 = ngb_res["per_holdout"]["h1"]["full"] - lgb_res["h1"]["full"]
    d2 = ngb_res["per_holdout"]["h2"]["full"] - lgb_res["h2"]["full"]
    print(f"{'Δ (NGB - LGB)':<22} {d1:>+19.2f} {d2:>+19.2f}   (负=改善)")

    for nm, lkey in (("h1", "h1"), ("h2", "h2")):
        a = lgb_res[lkey]; b = ngb_res["per_holdout"][nm]
        print(f"\n--- {nm} 分档 RMSE (n={b['n']:,}) ---")
        print(f"{'档':>14} {'n':>8} {'LGB':>10} {'NGB':>10} {'Δ':>9}")
        for e1, e2 in zip(a["bins"], b["bins"]):
            if e1["rmse"] is None:
                print(f"{e1['bin']:>14} {e1['n']:>8,}  (样本不足)"); continue
            print(f"{e1['bin']:>14} {e1['n']:>8,} {e1['rmse']:>10.2f} {e2['rmse']:>10.2f} "
                  f"{e2['rmse']-e1['rmse']:>+9.2f}")
        print(f"  σ vs |残差| Spearman: {b['sigma_vs_absresid_spearman']:+.4f} "
              f"(p={b['sigma_vs_absresid_p']:.2e})")
        print(f"  σ 中位数: >1800 段 {b['sigma_median_gt1800']:.3f} vs "
              f"<600 段 {b['sigma_median_lt600']:.3f}  → 比值 {b['sigma_ratio']:.2f}x")
    print("\n⚠️caveat: sklearn 树不原生处理类别特征, NGBoost 拿到的是整数编码的")
    print("  FEATS71; LGB 基线用的是原生 categorical dtype。此处 NGBoost 天然吃亏。")
    return {"d_h1": round(d1, 2), "d_h2": round(d2, 2)}


def main():
    lap("步骤1: 加载缓存")
    X_raw, X_num, y, mo, mvt_s = load()
    i_fit, i_es, es_m = split(mo, mvt_s)
    i_fit, n_cl = clean(y, i_fit)
    ho = {"h1": np.where(np.isin(mo, HOLD1))[0],
          "h2": np.where(np.isin(mo, HOLD2))[0]}
    lap(f"fit {len(i_fit):,} (清洗剔 {n_cl:,}) | ES {len(i_es):,} 月 {es_m}")
    lap(f"holdout1 {len(ho['h1']):,} | holdout2 {len(ho['h2']):,}")

    res = {"fit_months": FIT_MONTHS, "es_months": es_m, "n_fit": int(len(i_fit)),
           "n_fit_cleaned": n_cl, "n_es": int(len(i_es)),
           "holdout": {k: int(len(v)) for k, v in ho.items()},
           "lgb_rounds": LGB_ROUNDS, "seed": SEED,
           "caveat": "sklearn 树不原生处理类别; NGBoost 用整数编码的 FEATS71, "
                     "LGB 基线用原生 categorical dtype。此处 NGBoost 天然吃亏。"}

    # ---------------- 步骤2: 小样本 smoke ----------------
    lap("\n步骤2: 小样本 smoke (fit 20 万行, 20 estimators)")
    rs = np.random.RandomState(SEED)
    sub = rs.choice(len(i_fit), min(SMOKE_N, len(i_fit)), replace=False)
    i_sub = i_fit[sub]
    sm = run_ngb(X_num, y, i_sub, i_es, {"h1": ho["h1"][:20000]},
                 n_estimators=20, tag=" [smoke]")
    lap(f"  smoke ok: loc 中位 {sm['per_holdout']['h1']['loc_median']}, "
        f"σ 中位 {sm['per_holdout']['h1']['scale_median']}")
    res["smoke"] = {"n_fit": int(len(i_sub)), "n_estimators": 20,
                    "loc_median": sm["per_holdout"]["h1"]["loc_median"],
                    "scale_median": sm["per_holdout"]["h1"]["scale_median"]}
    del sm

    # ---------------- 步骤3: LGB 基线 ----------------
    lap("\n步骤3: LGB 基线 (FEATS71, MSE, 3000 轮, 原生 category)")
    t = time.time()
    lgb_res = run_lgb(X_raw, y, i_fit, i_es, ho)
    lap(f"LGB 完成 {time.time()-t:.1f}s  h1={lgb_res['h1']['full']}  h2={lgb_res['h2']['full']}")
    res["lgb"] = lgb_res
    del X_raw

    # ---------------- 步骤4: NGBoost 全量 ----------------
    lap("\n步骤4: NGBoost 全量 (500 estimators, Normal, Base=max_depth3)")
    ngb_res = run_ngb(X_num, y, i_fit, i_es, ho, n_estimators=500)
    del ngb_res["model"]

    # ---------------- 步骤5: 报告 ----------------
    d = report("NGBoost vs LGB GBDT", lgb_res, ngb_res)
    res["ngb"] = ngb_res
    res["delta"] = d
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    lap(f"→ {OUT}")


if __name__ == "__main__":
    main()