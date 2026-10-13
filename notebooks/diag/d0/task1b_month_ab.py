"""
task1b_month_ab.py — 定位 107s 差距的真实变量
轮数已排除(2000→8000 反而 +2.11s)。本脚本对照"训练月份构成":
  A: train 2-6,8-10 (ES=11,12)          ← corrector_ab / task1 的口径
  B: train 1,2,3,4,8,9,10,11,12 (ES=11,12 内部) ← learning_curve 的口径
同一 FEATS71 / 同一编码 / 同一 seed42 / 同一 cap。只看 Jan / Jul。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "task1b_month_ab.json"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
from v15_clean_retrain import FEATS71, LOW_CARD_CATS

CAP = 8000
LP = {"objective": "regression", "metric": "rmse",
      "learning_rate": 0.0147555939449363, "num_leaves": 127, "max_depth": 11,
      "min_data_in_leaf": 5, "feature_fraction": 0.6720708449966734,
      "bagging_fraction": 0.9813515130184908, "bagging_freq": 5,
      "reg_alpha": 2.4624169162161342, "reg_lambda": 1.9936339729896335,
      "verbose": -1, "num_threads": -1}
_t0 = time.time()
def log(*a): print(f"[{time.time()-_t0:6.0f}s]", " ".join(str(x) for x in a), flush=True)

dep = pd.read_pickle(D0 / "_corr_feat_cache.pkl")
mo = dep["month"].to_numpy(); y = dep["TAXITIME_SEC_mvt"].to_numpy(float)
X = dep[FEATS71]
jul = np.where(mo == 7)[0]
jan = np.where(mo == 1)[0]

def run(tag, fit_months, es_months):
    tr = np.where(np.isin(mo, fit_months))[0]
    es = np.where(np.isin(mo, es_months))[0]
    assert 1 not in fit_months and 7 not in fit_months, f"{tag} 训练月含 holdout!"
    dtr = lgb.Dataset(X.iloc[tr], label=y[tr], categorical_feature=LOW_CARD_CATS)
    dev = lgb.Dataset(X.iloc[es], label=y[es], categorical_feature=LOW_CARD_CATS, reference=dtr)
    log(f"{tag}: 训练月={fit_months} n={len(tr):,} | ES月={es_months} n={len(es):,}")
    b = lgb.train(dict(LP, seed=42, bagging_seed=42, feature_fraction_seed=42,
                       data_random_seed=42),
                  dtr, num_boost_round=CAP, valid_sets=[dev],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    bi = b.best_iteration or CAP
    out = {"tag": tag, "fit_months": fit_months, "n_train": int(len(tr)),
           "es_months": es_months, "best_iter": int(bi)}
    for nm, idx in [("Jan", jan), ("Jul", jul)]:
        p = np.clip(b.predict(X.iloc[idx], num_iteration=bi).astype(np.float64), 30, 10800)
        t = y[idx]; c = (t > 0) & (t <= 10800)
        out[nm] = {"n": int(len(idx)),
                   "full": round(float(np.sqrt(np.mean((p - t) ** 2))), 2),
                   "core": round(float(np.sqrt(np.mean((p[c] - t[c]) ** 2))), 2)}
        log(f"   {nm}: full={out[nm]['full']:7.2f}  core={out[nm]['core']:7.2f}")
    return out

# A: corrector_ab / task1 口径 —— ES 用训练月内时间末10% (11,12)
trA = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
ts = dep["MVT_TIME_UTC_mvt"].astype("int64").to_numpy()
mA = np.isin(mo, trA); idxA = np.where(mA)[0]
o = np.argsort(ts[idxA], kind="stable"); cut = int(len(o) * 0.9)
esA = sorted(set(mo[idxA[o[cut:]]].tolist()))
fitA = sorted(set(mo[idxA[o[:cut]]].tolist()))
resA = run("A_corrector口径", fitA, esA)

# B: learning_curve 口径 —— 训练含 11,12 月, ES 取 11,12 会泄漏, 改用 10 月末 10%
fitB = [1, 2, 3, 4, 8, 9, 10, 11, 12]
mB = np.isin(mo, fitB); idxB = np.where(mB)[0]
oB = np.argsort(ts[idxB], kind="stable"); cutB = int(len(oB) * 0.9)
esB = sorted(set(mo[idxB[oB[cutB:]]].tolist()))
fitB_actual = sorted(set(mo[idxB[oB[:cutB]]].tolist()))
log(f"B 口径: ES 月份={esB} (11,12 在训练内, 与 learning_curve 一致)")
resB = run("B_learningcurve口径", fitB_actual, esB)

log("")
log(f"对照  Jul core:  A={resA['Jul']['core']:.2f}   B={resB['Jul']['core']:.2f}   "
    f"差={resA['Jul']['core']-resB['Jul']['core']:+.2f}s")
log("参照  learning_curve 报告 Jul core≈293-299 (k=3000~8000)")
json.dump({"A": resA, "B": resB,
           "delta_jul_core": round(resA["Jul"]["core"] - resB["Jul"]["core"], 2)},
          open(OUT, "w"), ensure_ascii=False, indent=2)
log(f"→ {OUT}")
