"""
task1d_leak_test.py — 决定性根因检验
假设: p0_holdout_predictions.parquet 的低分(304.94)来自【训练集包含 1/7 月】的模型。
对照:
  LEAK = 训练 12 个月(含 1 和 7), 在 1+7 上评估   → 若 core≈300 则泄漏成立
  CLEAN= 训练 2-6,8-12,      在 1+7 上评估   → 真实泛化(core≈430)
同一特征/同一超参/同一 seed42 / cap2000。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/openclaw/Documents/PRC_data_challenge")
D0 = ROOT / "notebooks" / "diag" / "d0"
OUT = D0 / "task1d_leak_test.json"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D0))
from v15_clean_retrain import FEATS71, LOW_CARD_CATS

CAP = 2000
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
ho = np.where(np.isin(mo, [1, 7]))[0]
t = y[ho]; c = (t > 0) & (t <= 10800)

def train_eval(tag, fit_idx, es_idx, cap=CAP):
    dtr = lgb.Dataset(X.iloc[fit_idx], label=y[fit_idx], categorical_feature=LOW_CARD_CATS)
    dev = lgb.Dataset(X.iloc[es_idx], label=y[es_idx], categorical_feature=LOW_CARD_CATS, reference=dtr)
    b = lgb.train(dict(LP, seed=42, bagging_seed=42, feature_fraction_seed=42,
                       data_random_seed=42), dtr, num_boost_round=cap, valid_sets=[dev],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    bi = b.best_iteration or cap
    p = np.clip(b.predict(X.iloc[ho], num_iteration=bi).astype(np.float64), 30, 10800)
    r = {"tag": tag, "n_fit": int(len(fit_idx)), "best_iter": int(bi),
         "full": round(float(np.sqrt(np.mean((p - t) ** 2))), 2),
         "core": round(float(np.sqrt(np.mean((p[c] - t[c]) ** 2))), 2),
         "fit_months": sorted(set(mo[fit_idx].tolist()))}
    log(f"{tag}: fit={len(fit_idx):,} best_iter={bi} → full={r['full']:.2f} core={r['core']:.2f}")
    return r

def chrono_split(months):
    m = np.isin(mo, months); idx = np.where(m)[0]
    ts = dep["MVT_TIME_UTC_mvt"].astype("int64").to_numpy()[idx]
    o = np.argsort(ts, kind="stable"); cut = int(len(o) * 0.9)
    return idx[o[:cut]], idx[o[cut:]]

# LEAK: 训练集含 1 和 7 月
f_l, e_l = chrono_split([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
r_leak = train_eval("LEAK  (训练含 1+7 月)", f_l, e_l)

# CLEAN: 训练 2-6,8-12
f_c, e_c = chrono_split([2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
r_clean = train_eval("CLEAN (训练 2-6,8-12)", f_c, e_c)

log("")
log(f"差值 core: {r_leak['core'] - r_clean['core']:+.2f}s  (泄漏带来的虚高)")
log(f"P0 parquet 报 core=304.94; 本实验 LEAK={r_leak['core']:.2f} CLEAN={r_clean['core']:.2f}")
json.dump({"leak": r_leak, "clean": r_clean,
           "delta_core": round(r_leak["core"] - r_clean["clean" if False else "core"], 2),
           "p0_parquet_core": 304.94, "cap": CAP}, open(OUT, "w"), ensure_ascii=False, indent=2)
log(f"→ {OUT}")
